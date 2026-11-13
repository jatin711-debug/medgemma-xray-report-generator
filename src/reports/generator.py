"""Render the experimental pneumothorax research report as a PDF."""
import io
import logging
import math
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable, Image as ReportLabImage, KeepTogether, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from src.core.findings import clean_markdown, split_response
from src.core.heatmap import (
    AttributionMap, EvidenceSummary, render_overlay, render_token_grid, summarize_evidence,
)
from src.core.image_quality import intensity_histogram
from src.reports.charts import (
    HEX, PALETTE, DecisionGauge, IntensityHistogram, OverlayLegend, RegionChart,
    format_percent, format_signed,
)
from src.reports.storage import atomic_report, report_path

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
PREVIEW_MAX_SIDE = 1000
NOT_RECORDED = "Not recorded"
MARGIN = 40
REASONING_PREVIEW_CHARS = 320
WHITE_PIXEL_FLAG_PCT = 5.0
NARROW_RANGE_LEVELS = 100

# (background, accent bar, text) for each kind of model output.
OUTCOME_TONES = {
    "danger": (colors.HexColor("#FEF2F2"), colors.HexColor("#DC2626"), colors.HexColor("#991B1B")),
    "success": (colors.HexColor("#F0FDF4"), colors.HexColor("#16A34A"), colors.HexColor("#166534")),
    "neutral": (colors.HexColor("#F3F4F6"), colors.HexColor("#6B7280"), colors.HexColor("#374151")),
}
_OUTCOMES = {"pneumothorax detected": "danger", "no pneumothorax detected": "success"}

POSITIVE_LABEL = "‘Pneumothorax detected’"
NEGATIVE_LABEL = "‘No pneumothorax detected’"
LIMITATIONS = (
    "<b>Experimental research output.</b> Not validated for diagnosis, clinical decisions, or "
    "patient care. The model can be wrong or indeterminate, and no calibrated confidence is "
    "available. Attribution shows which image regions influenced one model score; it is not a "
    "segmentation, a lesion outline or a validated localization. Do not use real patient data "
    "without appropriate authorization and safeguards."
)


# --- formatting helpers -------------------------------------------------------------------

def outcome_tone(impression: str) -> str:
    """Return 'danger', 'success' or 'neutral' for the parsed model output."""
    return _OUTCOMES.get(impression.strip().lower(), "neutral")


def _sentence_case(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:].lower() if text else text


def _preview(text: str, limit: int) -> tuple[str, bool]:
    """Shorten ``text`` at a word boundary; report whether anything was cut."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return cut + "…", True


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _format_bytes(size) -> Optional[str]:
    if not isinstance(size, (int, float)) or size < 0:
        return None
    if size < 1024:
        return f"{int(size)} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024 ** 2:.1f} MB"


def _format_timestamp(value: str, compact: bool = False) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return value or NOT_RECORDED
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%d %b %Y, %H:%M UTC" if compact else "%B %d, %Y, %H:%M:%S UTC")


def answer_preference(score: float) -> float:
    """Share of the two answers' first-token probability that goes to 'Pneumothorax'.

    This is the model's own softmax restricted to the two diverging tokens. It is
    not calibrated against ground truth and is not a probability of disease.
    """
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    odds = math.exp(score)
    return odds / (1.0 + odds)


def score_agreement(impression: str, score: float) -> tuple[str, str]:
    """Compare the decision score's direction with the generated answer.

    Returns a tone ('success', 'danger' or 'neutral') and a sentence for the report.
    """
    tone = outcome_tone(impression)
    if tone == "neutral":
        return "neutral", "Answer not definite; the score compares the two definite answers."
    if score == 0:
        return "neutral", "No preference between the two definite answers."
    agrees = score > 0 if tone == "danger" else score < 0
    if agrees:
        return "success", "✓ Same direction as the generated answer."
    return "danger", "▲ Points away from the generated answer; treat with extra caution."


def region_headline(summary: EvidenceSummary) -> str:
    """One plain-language sentence on how much evidence points each way, and where."""
    parts = []
    for index, share, label in ((1, summary.toward_share, POSITIVE_LABEL),
                                (2, summary.against_share, NEGATIVE_LABEL)):
        text = f"{format_percent(share)} raised {label}"
        region = max(summary.regions, key=lambda item: item[index])
        if region[index] > 0:
            text += f" (largest share: {region[0].lower()})"
        parts.append(text)
    return "Of the total attribution, " + " and ".join(parts) + "."


def border_note(summary: EvidenceSummary) -> Optional[str]:
    """Flag attribution that sits disproportionately on the image edge."""
    if summary.border_share <= 1.5 * summary.border_area:
        return None
    return (f"{format_percent(summary.border_share)} of the attribution lies in the outermost "
            f"ring of tokens, which covers only {format_percent(summary.border_area)} of the "
            "image. Labels, markers and borders usually sit there, so check whether the model "
            "relied on them.")


def image_check_flags(check: dict) -> list[str]:
    """Descriptive notes about the input file, using the thresholds stated in each note."""
    flags = []
    if not check.get("grayscale", True):
        flags.append("Colour channels differ. Chest radiographs are normally grayscale; "
                     "confirm this is the intended image.")
    white = check.get("white_pixel_pct")
    if isinstance(white, (int, float)) and white > WHITE_PIXEL_FLAG_PCT:
        flags.append(f"{white:.1f}% of pixels are pure white (flagged above "
                     f"{WHITE_PIXEL_FLAG_PCT:.0f}%), which can hide detail in bright areas.")
    low, high = check.get("intensity_p1"), check.get("intensity_p99")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if high - low < NARROW_RANGE_LEVELS:
            flags.append(f"The 1st–99th percentile range spans only {high - low:.0f} of 255 "
                         f"levels (flagged below {NARROW_RANGE_LEVELS}); the image may be low "
                         "in contrast.")
    return flags


def _numbered_canvas(font: str, color, right: float, baseline: float):
    """A canvas class that adds 'Page n of N' once the page count is known."""

    class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._page_states = []

        def showPage(self):
            self._page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._page_states)
            for state in self._page_states:
                self.__dict__.update(state)
                self.setFont(font, 8)
                self.setFillColor(color)
                self.drawRightString(right, baseline, f"Page {self._pageNumber} of {total}")
                Canvas.showPage(self)
            Canvas.save(self)

    return NumberedCanvas


# --- report -------------------------------------------------------------------------------

class UltimatePDFGenerator:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self.colors = PALETTE
        self._setup_fonts()
        self._setup_styles()

    def _setup_fonts(self):
        """Use the bundled fonts, falling back to built-in PDF fonts."""
        try:
            for name, filename in (("Montserrat-Bold", "Montserrat-Bold.ttf"),
                                   ("Source", "SourceSansPro-Regular.ttf"),
                                   ("Source-Bold", "SourceSansPro-Bold.ttf")):
                pdfmetrics.registerFont(TTFont(name, str(ASSETS_DIR / filename)))
            pdfmetrics.registerFontFamily("Source", normal="Source", bold="Source-Bold",
                                          italic="Source", boldItalic="Source-Bold")
            self.font_display, self.font_body, self.font_body_bold = (
                "Montserrat-Bold", "Source", "Source-Bold")
        except Exception:
            logger.warning("Bundled fonts not found in %s; using built-in fonts.", ASSETS_DIR)
            self.font_display, self.font_body, self.font_body_bold = (
                "Helvetica-Bold", "Helvetica", "Helvetica-Bold")
        self.fonts = (self.font_body, self.font_body_bold)

    def _setup_styles(self):
        c = self.colors

        def style(name, font, size, leading, color, **extra):
            return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                                  textColor=color, **extra)

        body, bold = self.font_body, self.font_body_bold
        self.styles = {
            "title": style("Title", self.font_display, 20, 24, c["ink"]),
            "subtitle": style("Subtitle", body, 9.5, 13, c["gray_600"], spaceBefore=2,
                              spaceAfter=10),
            "section": style("Section", self.font_display, 11.5, 14, c["primary"],
                             spaceBefore=10),
            "intro": style("Intro", body, 8.5, 11.5, c["gray_600"], spaceAfter=6),
            "label": style("Label", bold, 7, 9, c["gray_500"]),
            "meta_value": style("MetaValue", bold, 9.5, 12, c["gray_800"]),
            "meta_sub": style("MetaSub", body, 7.5, 9.5, c["gray_500"]),
            "outcome": style("Outcome", self.font_display, 16, 20, c["gray_800"]),
            "body": style("Body", body, 9.5, 13.5, c["gray_800"]),
            "body_small": style("BodySmall", body, 8.5, 11.5, c["gray_700"]),
            "score": style("Score", bold, 15, 18, c["gray_800"]),
            "cell_label": style("CellLabel", bold, 8, 10.5, c["gray_700"]),
            "cell": style("Cell", body, 8, 10.5, c["gray_800"]),
            "caption": style("Caption", bold, 8, 10, c["gray_700"], alignment=TA_CENTER),
            "note": style("Note", body, 7.8, 10.5, c["gray_600"], spaceBefore=5),
            "tile_value": style("TileValue", bold, 14, 17, c["ink"]),
            "tile_sub": style("TileSub", body, 7.2, 9.2, c["gray_500"]),
        }

    # --- page furniture -------------------------------------------------------------------

    def _draw_page_template(self, canvas, doc, analysis_id: Optional[str] = None):
        """Draw the header and footer on each page."""
        c = self.colors
        page_width, page_height = doc.pagesize
        canvas.saveState()
        canvas.setFillColor(c["primary"])
        canvas.rect(0, page_height - 5, page_width, 5, stroke=0, fill=1)

        baseline = page_height - 30
        canvas.setFont(self.font_display, 10.5)
        canvas.setFillColor(c["ink"])
        canvas.drawString(MARGIN, baseline, "MedGemma")
        brand_width = pdfmetrics.stringWidth("MedGemma", self.font_display, 10.5)
        canvas.setFont(self.font_body, 10)
        canvas.setFillColor(c["gray_500"])
        canvas.drawString(MARGIN + brand_width + 6, baseline, "Research prototype")

        pill = "RESEARCH USE ONLY · NOT FOR CLINICAL USE"
        pill_width = pdfmetrics.stringWidth(pill, self.font_body_bold, 7) + 16
        pill_x = page_width - MARGIN - pill_width
        canvas.setFillColor(c["amber_50"])
        canvas.setStrokeColor(c["amber_500"])
        canvas.setLineWidth(0.75)
        canvas.roundRect(pill_x, baseline - 4, pill_width, 15, 7.5, stroke=1, fill=1)
        canvas.setFont(self.font_body_bold, 7)
        canvas.setFillColor(c["amber_800"])
        canvas.drawString(pill_x + 8, baseline + 0.8, pill)

        canvas.setStrokeColor(c["gray_200"])
        canvas.setLineWidth(0.75)
        canvas.line(MARGIN, page_height - 44, page_width - MARGIN, page_height - 44)
        canvas.line(MARGIN, 44, page_width - MARGIN, 44)
        canvas.setFont(self.font_body, 7.5)
        canvas.setFillColor(c["gray_500"])
        canvas.drawString(MARGIN, 31, "Experimental model output · Research use only · "
                                      "Not validated for patient care")
        if analysis_id:
            canvas.drawString(MARGIN, 21, f"Analysis {analysis_id}")
        canvas.restoreState()

    # --- building blocks ------------------------------------------------------------------

    def _label(self, text: str) -> Paragraph:
        return Paragraph(escape(text.upper()), self.styles["label"])

    def _section(self, title: str, flowables: list, intro: Optional[str] = None) -> KeepTogether:
        """Keep a heading, its rule and its content on the same page."""
        items = [Paragraph(escape(title), self.styles["section"]),
                 HRFlowable(width="100%", thickness=0.6, color=self.colors["gray_200"],
                            spaceBefore=3, spaceAfter=5)]
        if intro:
            items.append(Paragraph(intro, self.styles["intro"]))
        return KeepTogether(items + flowables)

    @staticmethod
    def _fit_image(image: Image.Image, max_width: float, max_height: float) -> ReportLabImage:
        """Embed a downscaled copy, sized to fit the box while keeping its aspect ratio."""
        preview = image.convert("RGB")
        preview.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)
        scale = min(max_width / image.width, max_height / image.height)
        return ReportLabImage(buffer, width=image.width * scale, height=image.height * scale)

    def _key_value_table(self, rows: list[tuple[str, str]], width: float,
                         label_width: float = 110, padding: float = 4) -> Table:
        data = [[Paragraph(escape(label), self.styles["cell_label"]),
                 Paragraph(escape(value).replace("\n", "<br/>"), self.styles["cell"])]
                for label, value in rows]
        table = Table(data, colWidths=[label_width, width - label_width])
        table.setStyle(self._grid_style(padding=padding))
        return table

    def _grid_style(self, extra: Optional[list] = None, padding: float = 4) -> TableStyle:
        return TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), self.colors["gray_50"]),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, self.colors["gray_200"]),
            ("BOX", (0, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            *(extra or []),
        ])

    def _meta_strip(self, response, width: float) -> Table:
        meta = response.image_metadata or {}
        info = response.model_info or {}
        source = str(meta.get("filename") or meta.get("url") or meta.get("source") or NOT_RECORDED)
        dimensions = meta.get("dimensions")
        image_sub = (f"{dimensions[0]} × {dimensions[1]} px"
                     if isinstance(dimensions, (list, tuple)) and len(dimensions) == 2 else None)
        precision = info.get("precision") or ""
        runtime = [("4-bit NF4" if "4-bit" in precision else precision),
                   str(info.get("device") or "").split(" ")[0]]
        analysis_id = response.analysis_id
        short_id = analysis_id if len(analysis_id) <= 13 else analysis_id[:8]
        cells = [
            ("Analysis ID", short_id, "Full ID in footer" if short_id != analysis_id else None),
            ("Generated", _format_timestamp(response.timestamp, compact=True), None),
            ("Image", _truncate(source.rsplit("/", 1)[-1], 28), image_sub),
            ("Model", _truncate(str(info.get("model_id") or NOT_RECORDED).split("/")[-1], 26),
             " · ".join(part for part in runtime if part) or None),
        ]
        row = []
        for label, value, sub in cells:
            cell = [self._label(label), Spacer(1, 2),
                    Paragraph(escape(value), self.styles["meta_value"])]
            if sub:
                cell.append(Paragraph(escape(sub), self.styles["meta_sub"]))
            row.append(cell)
        table = Table([row], colWidths=[width / 4] * 4)
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), self.colors["gray_50"]),
            ("BOX", (0, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    def _response_rows(self, response) -> list[tuple[str, str]]:
        sections = split_response(response.raw_response)
        finding = sections["finding"]
        # The banner already shows the parsed result; repeat the model's wording only if it differs.
        if finding and finding.lower().rstrip(". ") == response.findings.impression.lower():
            finding = None
        rows = [(label, value) for label, value in (("Finding", finding),
                                                    ("Reasoning", sections["reasoning"])) if value]
        if not rows:
            rows = [("Response", clean_markdown(response.raw_response) or "No text returned")]
        return rows

    def _response_is_long(self, response) -> bool:
        return any(len(value) > REASONING_PREVIEW_CHARS
                   for _, value in self._response_rows(response))

    def _finding_column(self, response, text_color) -> list:
        outcome_style = ParagraphStyle("OutcomeTone", parent=self.styles["outcome"],
                                       textColor=text_color)
        items = [self._label("Model finding"), Spacer(1, 4),
                 Paragraph(escape(_sentence_case(response.findings.impression)), outcome_style),
                 Spacer(1, 7)]
        rows = self._response_rows(response)
        prefixes = {"Finding": "Model wording", "Response": "Model response"}
        if len(rows) > 1:
            prefixes["Reasoning"] = "Reasoning"
        for label, value in rows:
            text, _ = _preview(value, REASONING_PREVIEW_CHARS)
            prefix = prefixes.get(label)
            markup = escape(text) if prefix is None else f"<b>{prefix}:</b> {escape(text)}"
            items.append(Paragraph(markup, self.styles["body"]))
        if self._response_is_long(response):
            items.append(Paragraph("Shortened here; see ‘Full model response’ later in this "
                                   "report.", self.styles["note"]))
        return items

    def _score_column(self, response, attribution: Optional[AttributionMap], width: float) -> list:
        items = [self._label("Decision score · logit difference"), Spacer(1, 3)]
        if attribution is None:
            status = (response.attribution or {}).get("status")
            reason = {
                "disabled": "Attribution is disabled on this server (MEDGEMMA_ATTRIBUTION=0).",
                "unavailable": "The attribution pass failed for this analysis.",
            }.get(status, "No attribution pass was recorded for this analysis.")
            return items + [Paragraph("Not available", self.styles["score"]),
                            Paragraph(escape(reason), self.styles["note"]),
                            Spacer(1, 8), self._label("Confidence"), Spacer(1, 2),
                            Paragraph("Not available (uncalibrated)", self.styles["body_small"])]

        score = attribution.target_score
        leaning = ("favours pneumothorax" if score > 0 else
                   "favours no pneumothorax" if score < 0 else "no preference")
        gray = HEX["gray_500"]
        items += [
            Paragraph(f"{format_signed(score)}"
                      f"<font name='{self.font_body}' size='8.5' color='{gray}'>"
                      f"&nbsp;&nbsp;{escape(leaning)}</font>",
                      self.styles["score"]),
            Spacer(1, 5),
            DecisionGauge(score, width, self.fonts),
            Spacer(1, 4),
        ]
        tone, sentence = score_agreement(response.findings.impression, score)
        tone_color = {"success": HEX["success_text"], "danger": HEX["danger_text"]}.get(
            tone, HEX["gray_600"])
        items.append(Paragraph(f"<font color='{tone_color}'>{escape(sentence)}</font>",
                               self.styles["body_small"]))
        preference = answer_preference(score)
        items.append(Paragraph(
            f"Token preference: pneumothorax {format_percent(preference)}, no pneumothorax "
            f"{format_percent(1 - preference)} (uncalibrated; not the probability of disease).",
            self.styles["note"]))
        return items

    def _result_panel(self, response, attribution: Optional[AttributionMap], width: float) -> Table:
        background, accent, text_color = OUTCOME_TONES[outcome_tone(response.findings.impression)]
        left_width = width * 0.53
        padding = 14
        right_width = width - left_width
        table = Table([[self._finding_column(response, text_color),
                        self._score_column(response, attribution, right_width - 2 * padding)]],
                      colWidths=[left_width, right_width])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), background),
            ("BACKGROUND", (1, 0), (1, 0), self.colors["white"]),
            ("LINEBEFORE", (0, 0), (0, 0), 4, accent),
            ("BOX", (0, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), padding),
            ("RIGHTPADDING", (0, 0), (-1, -1), padding),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        return table

    def _figure(self, image: Image.Image, attribution: Optional[AttributionMap],
                width: float) -> list:
        max_height = 236
        if attribution is None:
            # The decision-score panel already says why there is no map.
            single = Table([[self._fit_image(image, width * 0.55, max_height)]], colWidths=[width])
            single.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            return [single]

        gap = 16
        panel = (width - gap) / 2
        overlay = self._fit_image(render_overlay(image, attribution.grid), panel, max_height)
        rows = [
            [self._fit_image(image, panel, max_height), overlay],
            [Paragraph("Input image", self.styles["caption"]),
             Paragraph("Attribution overlay", self.styles["caption"])],
            ["", OverlayLegend(max(overlay.drawWidth, 200), self.fonts)],
        ]
        grid = Table(rows, colWidths=[width / 2, width / 2])
        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        rows_count, columns_count = attribution.grid.shape
        if attribution.is_empty:
            note = ("Every image token contributed zero to the score, so the overlay is blank "
                    "and there is no regional breakdown.")
        else:
            note = (f"Red areas raised the score for {POSITIVE_LABEL}; blue areas raised "
                    f"{NEGATIVE_LABEL}. Colour strength is relative to the strongest token in "
                    f"this image. {rows_count} × {columns_count} token resolution, smoothed for "
                    "display (the unsmoothed grid follows). Not a lesion outline or a validated "
                    "localization.")
        return [grid, Paragraph(note, self.styles["note"])]

    def _stat_tiles(self, tiles: list[tuple[str, str, str]], width: float) -> Table:
        row = [[Paragraph(label, self.styles["label"]), Spacer(1, 3),
                Paragraph(escape(value), self.styles["tile_value"]),
                Paragraph(escape(sub), self.styles["tile_sub"])] for label, value, sub in tiles]
        table = Table([row], colWidths=[width / len(tiles)] * len(tiles))
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.5, self.colors["gray_200"]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    def _region_section(self, image: Image.Image, attribution: AttributionMap,
                        summary: EvidenceSummary, width: float) -> list:
        map_width, map_height, caption_height = 142, 164, 13
        token_image = self._fit_image(render_token_grid(image, attribution.grid),
                                      map_width, map_height)
        column = map_width + 18
        chart = RegionChart(summary.regions, width - column - 8,
                            token_image.drawHeight + caption_height, self.fonts)
        layout = Table(
            [[[token_image, Spacer(1, 3),
               Paragraph("Token grid, unsmoothed", self.styles["caption"])], chart]],
            colWidths=[column, width - column])
        layout.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        swatch = "<font size='9' color='{}'>■</font>&nbsp;{}"
        tiles = self._stat_tiles([
            (swatch.format(HEX["toward"], "TOWARD PNEUMOTHORAX"),
             format_percent(summary.toward_share), "of the total attribution"),
            (swatch.format(HEX["against"], "TOWARD NO PNEUMOTHORAX"),
             format_percent(summary.against_share), "of the total attribution"),
            (f"STRONGEST {summary.top_token_count} TOKENS",
             format_percent(summary.top_token_share),
             "an even spread would give "
             f"{format_percent(summary.top_token_count / summary.token_count)}"),
            ("OUTER RING OF TOKENS", format_percent(summary.border_share),
             f"the ring covers {format_percent(summary.border_area)} of the image"),
        ], width)
        items = [layout, Spacer(1, 8), tiles]
        note = border_note(summary)
        if note:
            items.append(Paragraph("<b>Note:</b> " + escape(note), self.styles["note"]))
        return items

    def _image_check_rows(self, response) -> list[tuple[str, str]]:
        check = (response.image_metadata or {}).get("image_check")
        if not isinstance(check, dict) or "mean_intensity" not in check:
            return [("Image check", NOT_RECORDED)]
        rows = [
            ("Colour", "Grayscale" if check.get("grayscale") else "Colour (channels differ)"),
            ("Mean intensity", f"{check['mean_intensity']:.0f} / 255"),
            ("Contrast (SD)", f"{check.get('contrast_sd', 0):.0f}"),
            ("1st–99th percentile",
             f"{check.get('intensity_p1')}–{check.get('intensity_p99')} (shaded)"),
        ]
        if "black_pixel_pct" in check:
            rows.append(("Pure black / white", f"{check['black_pixel_pct']:.1f}% / "
                                               f"{check.get('white_pixel_pct', 0):.1f}%"))
        rows.append(("Flags", " ".join(image_check_flags(check)) or "None"))
        return rows

    def _image_check_section(self, response, image: Image.Image, width: float) -> list:
        check = (response.image_metadata or {}).get("image_check")
        check = check if isinstance(check, dict) else {}
        histogram_width = width * 0.48
        histogram = IntensityHistogram(intensity_histogram(image), histogram_width - 14, 66,
                                       self.fonts, check.get("intensity_p1"),
                                       check.get("intensity_p99"))
        table_width = width - histogram_width
        layout = Table(
            [[[histogram, Spacer(1, 3),
               Paragraph("Pixel intensity distribution", self.styles["caption"])],
              self._key_value_table(self._image_check_rows(response), table_width, 96, 2.5)]],
            colWidths=[histogram_width, table_width])
        layout.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return [layout]

    def _detail_rows(self, response) -> list[tuple[str, str]]:
        meta = response.image_metadata or {}
        info = response.model_info or {}
        timings = response.timings or {}
        attribution = response.attribution or {}

        source = meta.get("filename") or meta.get("url") or meta.get("source")
        image_parts = [_truncate(str(source), 48)] if source else []
        dimensions = meta.get("dimensions")
        if isinstance(dimensions, (list, tuple)) and len(dimensions) == 2:
            image_parts.append(f"{dimensions[0]} × {dimensions[1]} px")
        if meta.get("format"):
            image_parts.append(str(meta["format"]))
        if _format_bytes(meta.get("size")):
            image_parts.append(_format_bytes(meta.get("size")))

        runtime = " on ".join(part for part in (info.get("precision"), info.get("device")) if part)

        timing_parts = []
        if "generation_seconds" in timings:
            timing_parts.append(f"generation {timings['generation_seconds']:.2f} s")
        if "attribution_seconds" in timings:
            timing_parts.append(f"attribution {timings['attribution_seconds']:.2f} s")

        status = attribution.get("status")
        if status == "available":
            score = format_signed(attribution.get("target_score", 0), 3)
            attribution_text = (f"{attribution.get('method', 'Gradient × activation')} · "
                                f"target score {score} (uncalibrated)")
        elif status == "disabled":
            attribution_text = "Disabled"
        elif status == "unavailable":
            attribution_text = "Could not be computed for this analysis"
        else:
            attribution_text = NOT_RECORDED

        software_parts = [f"{name} {info[key]}" for key, name in
                          (("app_version", "app"), ("torch", "torch"),
                           ("transformers", "transformers")) if info.get(key)]

        return [
            ("Analysis ID", response.analysis_id),
            ("Generated", _format_timestamp(response.timestamp)),
            ("Image", " · ".join(image_parts) or NOT_RECORDED),
            ("Timing", " · ".join(timing_parts) or NOT_RECORDED),
            ("Model", info.get("model_id") or NOT_RECORDED),
            ("Precision and device", runtime or NOT_RECORDED),
            ("Software", " · ".join(software_parts) or NOT_RECORDED),
            ("Attribution", attribution_text),
        ]

    def _detail_grid(self, rows: list[tuple[str, str]], width: float) -> Table:
        """Two label/value columns side by side, filled top to bottom."""
        half = (len(rows) + 1) // 2
        left, right = rows[:half], rows[half:]
        label_width, column = 90, width / 2

        def cells(pair):
            if pair is None:
                return ["", ""]
            label, value = pair
            return [Paragraph(escape(label), self.styles["cell_label"]),
                    Paragraph(escape(value), self.styles["cell"])]

        data = [cells(left[index]) + cells(right[index] if index < len(right) else None)
                for index in range(half)]
        table = Table(data, colWidths=[label_width, column - label_width,
                                       label_width, column - label_width])
        table.setStyle(self._grid_style([("LINEBEFORE", (2, 0), (2, -1), 0.5,
                                          self.colors["gray_200"])]))
        return table

    def _method(self, attribution: AttributionMap) -> Paragraph:
        rows, columns = attribution.grid.shape
        return Paragraph(
            f"Gradient × activation on the {rows * columns} image tokens ({rows} × {columns}) "
            f"that MedGemma's language model receives, taken at the token where "
            f"{POSITIVE_LABEL} and {NEGATIVE_LABEL} first differ. The decision score is the "
            "logit difference between those two tokens; positive favours pneumothorax. Regions "
            "are thirds and halves of the image frame, not anatomical zones. Image left is "
            "usually the patient's right on a PA view, but orientation is not checked.",
            self.styles["cell"])

    def _limitations(self, width: float) -> Table:
        box = Table([[Paragraph(LIMITATIONS, self.styles["note"])]], colWidths=[width])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.colors["amber_50"]),
            ("BOX", (0, 0), (-1, -1), 0.75, self.colors["amber_500"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return box

    # --- public API -----------------------------------------------------------------------

    def generate_pdf_report(self, analysis_response, original_image: Image.Image,
                            attribution: Optional[AttributionMap] = None) -> str:
        """Build a research PDF and publish it only after rendering succeeds."""
        destination = report_path(self.reports_dir, analysis_response.analysis_id)
        with atomic_report(destination) as temporary:
            self._build_pdf_report(analysis_response, original_image, attribution, temporary)
        logger.info("Successfully generated experimental PDF report")
        return str(destination)

    def _story(self, response, image: Image.Image, attribution: Optional[AttributionMap],
               width: float) -> list:
        summary = summarize_evidence(attribution.grid) if attribution is not None else None
        story = [
            Paragraph("Pneumothorax Research Report", self.styles["title"]),
            Paragraph("Single chest X-ray analysed by an experimental MedGemma pipeline. "
                      "For research only; not for clinical use.", self.styles["subtitle"]),
            self._meta_strip(response, width),
            Spacer(1, 10),
            self._result_panel(response, attribution, width),
            self._section("Attribution map" if attribution is not None else "Input image",
                          self._figure(image, attribution, width)),
            Spacer(1, 10),
            self._limitations(width),
        ]
        if summary is not None:
            story.append(self._section(
                "Attribution by region",
                self._region_section(image, attribution, summary, width),
                escape(region_headline(summary))
                + " Bars show each region's share; all twelve sum to 100%."))
        story.append(self._section(
            "Input image check", self._image_check_section(response, image, width),
            "Descriptive statistics of the uploaded file. They are not a diagnostic quality "
            "assessment."))
        if self._response_is_long(response):
            text = clean_markdown(response.raw_response) or "No text returned"
            story.append(self._section("Full model response", [
                Paragraph(escape(text).replace("\n", "<br/>"), self.styles["body_small"])]))
        story.append(self._section("Technical details",
                                   [self._detail_grid(self._detail_rows(response), width)]))
        if attribution is not None:
            story.append(self._section("Method", [self._method(attribution)]))
        return story

    def _build_pdf_report(self, response, original_image: Image.Image,
                          attribution: Optional[AttributionMap], filepath: str) -> None:
        """Render the document to a private temporary path before publication."""
        doc = SimpleDocTemplate(
            filepath, pagesize=letter, rightMargin=MARGIN, leftMargin=MARGIN, topMargin=58,
            bottomMargin=56, title="Pneumothorax research report",
            author="MedGemma research prototype", subject="Experimental model output",
        )
        on_page = partial(self._draw_page_template, analysis_id=response.analysis_id)
        doc.build(self._story(response, original_image, attribution, doc.width),
                  onFirstPage=on_page, onLaterPages=on_page,
                  canvasmaker=_numbered_canvas(self.font_body_bold, self.colors["primary"],
                                               letter[0] - MARGIN, 26))
