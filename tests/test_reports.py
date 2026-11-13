"""Local report publication checks without importing the model runtime."""
import itertools
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Table

from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.core.decision import DETECTED, INDETERMINATE, NEGATIVE, REVIEW, Calibration, decide
from src.core.findings import answer_agrees
from src.core.heatmap import AttributionMap, relevance_grid, summarize_evidence
from src.core.image_quality import image_statistics
from src.reports.charts import DecisionGauge
from src.reports.generator import (BOTTOM_MARGIN, MARGIN, NOT_RECORDED, PAGE_ONE_HEIGHT,
                                   TOP_MARGIN, UltimatePDFGenerator, band_evidence, border_note,
                                   calibration_table, decision_score, fit_text, image_check_flags,
                                   outcome_tone, region_headline, text_answer_note)
from src.reports.storage import atomic_report, list_local_reports, report_path

DOCUMENT_WIDTH = letter[0] - 2 * MARGIN


def page_count(path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", Path(path).read_bytes()))


CALIBRATION = Calibration(-4.0, 2.0, {
    "model_id": "google/medgemma-1.5-4b-it", "precision": "4-bit NF4 weights, bfloat16 compute",
    "prompt_fingerprint": "0123456789abcdef", "created": "2026-11-13", "prevalence": 0.0544,
    "reference": {"n_images": 4376, "n_positive": 238, "n_patients": 1695},
    "cross_validated": {
        "review": {"sensitivity": [0.9, 0.86, 0.94], "specificity": [0.61, 0.58, 0.64],
                   "flagged_share": 0.41},
        "detected": {"sensitivity": [0.52, 0.45, 0.6], "specificity": [0.95, 0.94, 0.96],
                     "flagged_share": 0.08},
        "bands": {"detected": {"positive_share": [0.357, 0.277, 0.442], "image_share": 0.071},
                  "review": {"positive_share": [0.074, 0.057, 0.093], "image_share": 0.314},
                  "negative": {"positive_share": [0.009, 0.006, 0.013], "image_share": 0.614}}}})


def decision_for(score, answer="No pneumothorax detected", calibration=CALIBRATION) -> dict:
    """The decision dict exactly as the analyzer records it."""
    decision = decide(score, calibration)
    return {**decision.as_dict(), "prompt_fingerprint": "0123456789abcdef",
            "text_answer": answer, "text_agrees": answer_agrees(answer, decision.band)}


def make_response(impression="Indeterminate", raw="Uncertain <finding>", **extra):
    findings = RadiologyFindings(anatomical_structures=[], abnormalities=[], impression=impression,
                                 recommendations=[], confidence_level="Not available")
    return AnalysisResponse(analysis_id="test-id", timestamp="2026-01-01T00:00:00+00:00",
                            findings=findings, raw_response=raw,
                            image_metadata={"filename": "<scan & input>.png"}, **extra)


class ReportStorageTests(unittest.TestCase):
    def test_report_is_visible_only_after_successful_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = report_path(directory, "test-id")
            with atomic_report(destination) as temporary:
                Path(temporary).write_bytes(b"complete report")
                self.assertEqual(list_local_reports(directory), [])
            self.assertEqual(destination.read_bytes(), b"complete report")
            reports = list_local_reports(directory)
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0]["analysis_id"], "test-id")
            self.assertIsNotNone(datetime.fromisoformat(reports[0]["created_at"]).tzinfo)
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_failed_replacement_preserves_existing_report_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = report_path(directory, "test-id")
            destination.write_bytes(b"previous report")
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                with atomic_report(destination) as temporary:
                    Path(temporary).write_bytes(b"partial report")
                    raise RuntimeError("build failed")
            self.assertEqual(destination.read_bytes(), b"previous report")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_rename_failure_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.reports.storage.os.replace", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    with atomic_report(report_path(directory, "test-id")) as temporary:
                        Path(temporary).write_bytes(b"report")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rejects_unsafe_or_unbounded_identifiers(self):
        for identifier in ("", "../escape", "..\\escape", "/absolute", "a/b", "a:b", "a" * 129):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                report_path("reports", identifier)

    def test_listing_ignores_directories_and_other_files_and_sorts_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path(directory, "folder").mkdir()
            (Path(directory) / ".report-temporary.tmp").write_bytes(b"unfinished")
            for identifier in ("z", "a"):
                report_path(directory, identifier).write_bytes(b"report")
            self.assertEqual([item["analysis_id"] for item in list_local_reports(directory)], ["a", "z"])
            self.assertEqual(list_local_reports(str(Path(directory) / "missing")), [])

    def test_listing_ignores_symlink_entries_and_disappearing_files(self):
        link = Mock(name="link entry")
        link.name = "premium_medical_report_link.pdf"
        link.stat.return_value.st_mode = 0o120777  # Symbolic link, not a regular file.
        removed = Mock(name="removed entry")
        removed.name = "premium_medical_report_removed.pdf"
        removed.stat.side_effect = FileNotFoundError
        with patch("src.reports.storage.os.scandir") as scan:
            scan.return_value.__iter__.return_value = iter([link, removed])
            self.assertEqual(list_local_reports("reports"), [])
        link.stat.assert_called_once_with(follow_symlinks=False)


def paragraph_texts(flowables):
    """Plain text of every paragraph in a story, including inside sections and tables."""
    for item in flowables:
        if isinstance(item, Paragraph):
            yield item.getPlainText()
        elif isinstance(item, KeepTogether):
            yield from paragraph_texts(item._content)
        elif isinstance(item, Table):
            for row in item._cellvalues:
                for cell in row:
                    # A wrapped table stores list cells as tuples.
                    nested = cell if isinstance(cell, (list, tuple)) else [cell]
                    yield from paragraph_texts(nested)


def signed_attribution(score: float) -> AttributionMap:
    """Evidence toward pneumothorax at the upper left, against it lower down."""
    yy, xx = np.mgrid[0:16, 0:16]
    tokens = (np.exp(-(((xx - 4) / 2.0) ** 2 + ((yy - 3) / 2.0) ** 2))
              - 0.8 * np.exp(-(((xx - 11) / 3.0) ** 2 + ((yy - 10) / 3.0) ** 2)))
    return AttributionMap(relevance_grid(tokens), score, "Gradient × activation on image tokens",
                          "target")


def full_response(impression: str, raw: str, score: float, image: Image.Image, **extra):
    """A response with every field the analyzer records, using long realistic values."""
    response = make_response(
        impression, raw,
        model_info={"model_id": "google/medgemma-1.5-4b-it",
                    "precision": "4-bit NF4 weights, bfloat16 compute",
                    "device": "cuda (NVIDIA GeForce RTX 3050 Laptop GPU)",
                    "app_version": "dfa0d3ac820b", "torch": "2.8.0+cu126",
                    "transformers": "5.20.1"},
        timings={"generation_seconds": 3.2, "attribution_seconds": 1.1},
        attribution={"status": "available", "method": "Gradient × activation on image tokens",
                     "target": "t", "target_score": score},
        **extra,
    )
    response.image_metadata.update({
        "url": "https://example.org/datasets/chest-xray/" + "very-long-folder-name/" * 3 + "a.png",
        "filename": None, "dimensions": list(image.size), "format": "PNG", "size": 2_400_000,
        "image_check": image_statistics(image),
    })
    return response


ONE_SENTENCE = ("**Finding**: {finding}\n**Brief Reasoning**: Lung markings appear to extend to "
                "the periphery on both sides, and no visceral pleural line or deep sulcus sign is "
                "visible on this frontal projection.")


class ReportGenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.generator = UltimatePDFGenerator(self.directory.name)
        self.response = make_response()

    def test_real_pdf_build_publishes_complete_document(self):
        path = self.generator.generate_pdf_report(self.response, Image.new("RGB", (16, 16)))
        content = Path(path).read_bytes()
        self.assertTrue(content.startswith(b"%PDF"))
        self.assertTrue(content.rstrip().endswith(b"%%EOF"))
        self.assertEqual(len(list(Path(self.directory.name).iterdir())), 1)

    def test_failed_pdf_build_does_not_publish_a_report(self):
        def fail_build(doc, *args, **kwargs):
            Path(doc.filename).write_bytes(b"%PDF partial")
            raise RuntimeError("renderer failed")
        with patch("src.reports.generator.SimpleDocTemplate.build", new=fail_build):
            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                self.generator.generate_pdf_report(self.response, Image.new("RGB", (16, 16)))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_page_template_labels_output_as_research(self):
        canvas, document = Mock(), Mock(pagesize=(612, 792), page=1)
        self.generator._draw_page_template(canvas, document, analysis_id="abc")
        text = " ".join(str(call.args[-1]) for call in canvas.drawString.call_args_list
                        + canvas.drawRightString.call_args_list)
        self.assertIn("Research use only", text)
        self.assertIn("Analysis abc", text)
        self.assertNotIn("HIPAA", text)
        self.assertNotIn("Encrypted", text)
        self.assertNotIn("Clinical Decision Support", text)

    def test_full_reports_fit_on_two_pages_for_every_outcome(self):
        cases = ((DETECTED, "No pneumothorax detected", 2.4, CALIBRATION),
                 (REVIEW, "No pneumothorax detected", -2.3, CALIBRATION),
                 (NEGATIVE, "Pneumothorax detected", -7.5, CALIBRATION),
                 (INDETERMINATE, "Possible small apical pneumothorax", 0.4, None))
        for size in ((900, 1100), (1400, 1000)):
            for impression, finding, score, calibration in cases:
                with self.subTest(size=size, impression=impression):
                    image = Image.new("RGB", size, (90, 90, 90))
                    response = full_response(impression, ONE_SENTENCE.format(finding=finding),
                                             score, image,
                                             decision=decision_for(score, None, calibration))
                    response.analysis_id = f"test-{size[0]}-{impression.split()[0].lower()}"
                    path = self.generator.generate_pdf_report(response, image,
                                                              signed_attribution(score))
                    self.assertEqual(page_count(path), 2)

    def test_summary_page_keeps_the_limitations_on_page_one(self):
        # A real run put the limitations box (and then Method) a page late: a portrait image,
        # the two-line review title and a long file name together overflowed page 1.
        long_name = "05-Spontanpneumothorax-follow-up-film-with-an-unusually-long-name.jpg"
        cases = ((REVIEW, -2.3, "Pneumothorax detected"),
                 (DETECTED, 2.4, "No pneumothorax detected"),
                 (NEGATIVE, -7.5, "Pneumothorax detected"))
        # 'shorter' takes 40 pt off the page so that the figure has to shrink to fit.
        for size, (impression, score, answer), shorter in itertools.product(
                ((600, 1100), (1060, 1227), (952, 956), (1920, 1557)), cases, (0, 40)):
            with self.subTest(size=size, impression=impression, shorter=shorter):
                image = Image.new("RGB", size, (90, 90, 90))
                response = full_response(impression, ONE_SENTENCE.format(finding=answer),
                                         score, image, decision=decision_for(score, answer))
                response.image_metadata["filename"] = long_name
                response.analysis_id = f"fit-{size[0]}x{size[1]}-{impression.split()[0]}"
                story = self.generator._story(response, image, signed_attribution(score),
                                              DOCUMENT_WIDTH, PAGE_ONE_HEIGHT - shorter)
                page_one = next(index for index, item in enumerate(story)
                                if "Experimental research output" in
                                " ".join(paragraph_texts([item])))
                path = Path(self.directory.name) / f"{response.analysis_id}-{shorter}-p1.pdf"
                SimpleDocTemplate(str(path), pagesize=letter, leftMargin=MARGIN,
                                  rightMargin=MARGIN, topMargin=TOP_MARGIN + shorter,
                                  bottomMargin=BOTTOM_MARGIN).build(story[:page_one + 1])
                self.assertEqual(page_count(path), 1)
                if not shorter:
                    report = self.generator.generate_pdf_report(response, image,
                                                                signed_attribution(score))
                    self.assertEqual(page_count(report), 2)

    def test_generated_text_is_kept_in_full_and_labelled_as_not_deciding(self):
        reasoning = "Long reasoning sentence number {}. "
        raw = "**Finding**: No pneumothorax detected\n**Brief Reasoning**: " + "".join(
            reasoning.format(index) for index in range(30))
        image = Image.new("RGB", (900, 1100), (90, 90, 90))
        response = full_response(REVIEW, raw, -1.5, image, decision=decision_for(-1.5))
        story = self.generator._story(response, image, signed_attribution(-1.5), 532)
        text = " ".join(paragraph_texts(story))
        self.assertIn("Generated text (not used for the outcome)", text)
        self.assertIn("written to fit it", text)
        self.assertIn("number 29.", text)
        panel = self.generator._finding_column(response, "#000000")
        self.assertNotIn("number 0.", " ".join(paragraph_texts(panel)))
        path = self.generator.generate_pdf_report(response, image, signed_attribution(-1.5))
        self.assertLessEqual(page_count(path), 3)

    def test_reports_render_without_attribution_and_with_a_blank_map(self):
        blank = AttributionMap(np.zeros((16, 16), dtype=np.float32), -0.5, "m", "t")
        for status, attribution in (("disabled", None), ("unavailable", None),
                                    ("available", blank)):
            with self.subTest(status=status):
                response = make_response(attribution={"status": status})
                response.analysis_id = f"test-{status}"
                path = self.generator.generate_pdf_report(response, Image.new("RGB", (64, 48)),
                                                          attribution)
                self.assertLessEqual(page_count(path), 2)


class ReportContentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.generator = UltimatePDFGenerator(directory.name)

    def test_outcome_colour_follows_the_band(self):
        self.assertEqual(outcome_tone(DETECTED), "danger")
        self.assertEqual(outcome_tone(REVIEW), "warning")
        self.assertEqual(outcome_tone(NEGATIVE), "success")
        self.assertEqual(outcome_tone(INDETERMINATE), "neutral")
        # Older, text-only outcomes keep their colours.
        self.assertEqual(outcome_tone("Pneumothorax Detected"), "danger")
        self.assertEqual(outcome_tone("No Pneumothorax Detected"), "success")

    def _panel_text(self, response) -> str:
        columns = (self.generator._finding_column(response, "#000000")
                   + self.generator._score_column(response, 200))
        return " ".join(paragraph_texts(columns))

    def test_review_band_asks_for_review_and_reports_held_out_performance(self):
        response = make_response(REVIEW, "**Finding**: No pneumothorax detected",
                                 decision=decision_for(-1.5))
        text = self._panel_text(response)
        self.assertIn("Possible pneumothorax: review required", text)
        self.assertIn("Have a radiologist review", text)
        self.assertIn("7.4% (5.7%–9.3%) of images in this band had a pneumothorax, against 5.4% "
                      "overall", text)
        self.assertIn("Generated text answer: ‘No pneumothorax detected’", text)
        self.assertIn("review band", text)
        self.assertIn("4,376 radiologist-labelled", text)
        self.assertIn("Review ≥ −4.00 90% (86%–94%) 39% (36%–42%)", text)
        self.assertIn("Suspected ≥ +2.00 52% (45%–60%) 5% (4%–6%)", text)

    def test_contradicting_text_is_flagged_and_negative_band_needs_no_targeted_review(self):
        response = make_response(NEGATIVE, "**Finding**: Pneumothorax detected",
                                 decision=decision_for(-7.0, "Pneumothorax detected"))
        text = self._panel_text(response)
        self.assertIn("contradicts this outcome", text)
        self.assertNotIn("Have a radiologist review", text)

    def test_uncalibrated_and_unscored_panels_explain_themselves(self):
        uncalibrated = self._panel_text(make_response(decision=decision_for(3.0, None, None)))
        self.assertIn("No calibration matches", uncalibrated)
        self.assertIn("uncalibrated", uncalibrated)
        unscored = self._panel_text(make_response(decision=decision_for(None, None)))
        self.assertIn("Not available", unscored)
        legacy = self._panel_text(make_response("No Pneumothorax Detected"))
        self.assertIn("generated text alone", legacy)

    def test_detail_rows_describe_model_runtime_and_image(self):
        response = make_response(
            model_info={"model_id": "model/x", "precision": "4-bit NF4 weights, bfloat16 compute",
                        "device": "cuda (GPU)", "app_version": "abc123", "torch": "2.8.0",
                        "transformers": "5.20.1"},
            timings={"generation_seconds": 2.5, "scoring_seconds": 0.2,
                     "attribution_seconds": 0.75},
            attribution={"status": "available", "method": "Gradient × activation",
                         "target": "logit difference", "target_score": 1.25},
            decision=decision_for(1.25),
        )
        response.image_metadata.update({"dimensions": (640, 480), "format": "PNG", "size": 2048})
        rows = dict(self.generator._detail_rows(response))
        self.assertEqual(rows["Model"], "model/x")
        self.assertEqual(rows["Precision and device"],
                         "4-bit NF4 weights, bfloat16 compute on cuda (GPU)")
        self.assertEqual(rows["Timing"], "generation 2.50 s · scoring 0.20 s · attribution 0.75 s")
        self.assertEqual(rows["Calibration"], "review −4.000 / suspected +2.000 · "
                                              "prompt 0123456789abcdef · fitted 2026-11-13")
        self.assertEqual(rows["Image"], "<scan & input>.png · 640 × 480 px · PNG · 2 KB")
        self.assertIn("+1.250", rows["Attribution"])
        self.assertEqual(rows["Software"], "app abc123 · torch 2.8.0 · transformers 5.20.1")
        self.assertEqual(rows["Analysis"], "test-id · January 01, 2026, 00:00:00 UTC")

    def test_missing_details_are_marked_not_recorded(self):
        response = make_response()
        response.image_metadata = {}
        rows = dict(self.generator._detail_rows(response))
        for key in ("Image", "Model", "Precision and device", "Timing", "Attribution", "Software",
                    "Calibration"):
            with self.subTest(key=key):
                self.assertEqual(rows[key], NOT_RECORDED)
        self.assertEqual(self.generator._image_check_rows(response),
                         [("Image check", NOT_RECORDED)])

    def test_image_check_rows_and_flags(self):
        response = make_response()
        response.image_metadata["image_check"] = {
            "grayscale": True, "mean_intensity": 101.4, "contrast_sd": 40.2, "intensity_p1": 3,
            "intensity_p99": 250, "black_pixel_pct": 1.25, "white_pixel_pct": 0.5}
        rows = dict(self.generator._image_check_rows(response))
        self.assertEqual(rows["Colour"], "Grayscale")
        self.assertEqual(rows["Mean intensity"], "101 / 255")
        self.assertEqual(rows["Pure black / white"], "1.2% / 0.5%")
        self.assertEqual(rows["Flags"], "None")

        flags = image_check_flags({"grayscale": False, "white_pixel_pct": 12.0,
                                   "intensity_p1": 90, "intensity_p99": 150})
        self.assertEqual(len(flags), 3)
        self.assertIn("Colour channels differ", flags[0])
        self.assertIn("12.0% of pixels are pure white", flags[1])
        self.assertIn("spans only 60 of 255", flags[2])

    def test_decision_score_prefers_the_recorded_decision(self):
        self.assertEqual(decision_score(make_response(decision={"score": -1.5},
                                                      attribution={"target_score": 9.0})), -1.5)
        self.assertEqual(decision_score(make_response(attribution={"target_score": 9.0})), 9.0)
        self.assertIsNone(decision_score(make_response(decision={"score": float("nan")})))

    def test_text_answer_note_only_warns_on_contradiction(self):
        self.assertEqual(text_answer_note(decision_for(5.0, "No pneumothorax detected"))[0], "danger")
        self.assertEqual(text_answer_note(decision_for(5.0, "Pneumothorax detected"))[0], "neutral")
        self.assertEqual(text_answer_note(decision_for(-1.0, "No pneumothorax detected"))[0],
                         "neutral")
        self.assertIn("no definite answer", text_answer_note(decision_for(-9.0, None))[1])
        self.assertIsNone(text_answer_note(decision_for(1.0, None, None)))

    def test_band_evidence_needs_band_rates_and_prevalence(self):
        self.assertIn("0.9% (0.6%–1.3%)", band_evidence(decision_for(-9.0)))
        self.assertIn("36% (28%–44%)", band_evidence(decision_for(5.0)))
        # The interval takes the estimate's precision rather than mixing '7.4%' with '12%'.
        straddling = {"band": "review", "calibration": {"prevalence": 0.0544, "cross_validated": {
            "bands": {"review": {"positive_share": [0.094, 0.074, 0.116]}}}}}
        self.assertIn("9.4% (7.4%–11.6%)", band_evidence(straddling))
        self.assertIsNone(band_evidence(decision_for(3.0, None, None)))
        self.assertIsNone(band_evidence({"band": "review", "calibration": {"prevalence": 0.05}}))

    def test_long_values_are_shortened_to_one_line_keeping_the_extension(self):
        name = "05-Spontanpneumothorax-follow-up-film.jpg"
        short = fit_text(name, "Helvetica-Bold", 10, 110)
        self.assertLessEqual(pdfmetrics.stringWidth(short, "Helvetica-Bold", 10), 110)
        self.assertTrue(short.startswith("05-Spont") and short.endswith(".jpg"))
        self.assertIn("…", short)
        self.assertEqual(fit_text("a.png", "Helvetica-Bold", 10, 110), "a.png")

    def test_gauge_labels_every_band_for_each_shipped_calibration(self):
        # The report's gauge width: the score column minus its cell padding.
        width = DOCUMENT_WIDTH - DOCUMENT_WIDTH * 0.53 - 2 * 14
        entries = json.loads((Path(__file__).resolve().parents[1] / "src/core/calibration.json")
                             .read_text(encoding="utf-8"))
        for entry in entries["entries"]:
            with self.subTest(precision=entry["precision"]):
                gauge = DecisionGauge(0.0, width, self.generator.fonts,
                                      entry["review_threshold"], entry["detected_threshold"])
                placed = gauge.label_positions()
                self.assertEqual([label for label, _ in placed], ["No flag", "Review", "Suspected"])
                ends = [left + pdfmetrics.stringWidth(label, gauge.font, gauge.LABEL_SIZE)
                        for label, left in placed]
                for (_, left), previous_end in zip(placed[1:], ends):
                    self.assertGreaterEqual(left, previous_end)
                self.assertGreaterEqual(placed[0][1], -gauge.LABEL_OVERHANG)
                self.assertLessEqual(ends[-1], width + gauge.LABEL_OVERHANG)

    def test_calibration_table_reports_both_bands_or_nothing(self):
        caption, rows = calibration_table(decision_for(0.0))
        self.assertIn("238 with pneumothorax", caption)
        self.assertEqual(rows, [("Review ≥ −4.00", "90% (86%–94%)", "39% (36%–42%)"),
                                ("Suspected ≥ +2.00", "52% (45%–60%)", "5% (4%–6%)")])
        self.assertIsNone(calibration_table({"calibration": {"cross_validated": {
            "review": {"sensitivity": [0.9, None, 0.95], "specificity": [0.6, 0.5, 0.7]}}}}))

    def test_region_headline_and_border_note_follow_the_summary(self):
        grid = np.zeros((16, 16))
        grid[0, 0:4] = 1.0
        grid[12, 12] = -4.0
        summary = summarize_evidence(grid)
        self.assertEqual(
            region_headline(summary),
            "Of the total attribution, 50% raised ‘Pneumothorax detected’ (largest share: "
            "upper, image left) and 50% raised ‘No pneumothorax detected’ (largest share: "
            "lower, image right).")
        self.assertIn("outermost ring", border_note(summary))
        self.assertIsNone(border_note(summarize_evidence(np.ones((16, 16)))))

        one_sided = summarize_evidence(-np.ones((16, 16)))
        self.assertIn("0% raised ‘Pneumothorax detected’ and", region_headline(one_sided))
