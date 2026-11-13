"""Chart flowables, colours and number formats shared by the PDF report."""
from typing import Optional

import numpy as np
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Flowable

from src.core.heatmap import AGAINST_COLOR, TOWARD_COLOR, diverging_colors, overlay_colors

GAUGE_LIMIT = 10.0

HEX = {
    "ink": "#0B1426",
    "primary": "#1E3A8A",
    "gray_800": "#1F2937",
    "gray_700": "#374151",
    "gray_600": "#4B5563",
    "gray_500": "#6B7280",
    "gray_400": "#9CA3AF",
    "gray_300": "#D1D5DB",
    "gray_200": "#E5E7EB",
    "gray_100": "#F3F4F6",
    "gray_50": "#F9FAFB",
    "amber_50": "#FFFBEB",
    "amber_500": "#F59E0B",
    "amber_800": "#92400E",
    "histogram": "#64748B",
    "histogram_band": "#E6ECF5",
    "toward": "#%02x%02x%02x" % TOWARD_COLOR,
    "against": "#%02x%02x%02x" % AGAINST_COLOR,
    "success_text": "#166534",
    "danger_text": "#991B1B",
}
PALETTE = {name: colors.HexColor(value) for name, value in HEX.items()}
PALETTE["white"] = colors.white


def format_signed(value: float, digits: int = 2) -> str:
    """Format with an explicit sign and a true minus sign."""
    return f"{value:+.{digits}f}".replace("-", "−")


def format_percent(share: float) -> str:
    """Format a share in [0, 1] as a whole percentage without rounding to 0% or 100%."""
    if share <= 0:
        return "0%"
    if share < 0.005:
        return "<1%"
    if share > 0.995 and share < 1:
        return ">99%"
    return f"{share * 100:.0f}%"


class _Chart(Flowable):
    """A fixed-size drawing that uses the report's fonts."""

    def __init__(self, width: float, height: float, fonts: tuple[str, str]):
        super().__init__()
        self.width, self.height = width, height
        self.font, self.font_bold = fonts

    def wrap(self, available_width, available_height):
        return self.width, self.height

    def _fill(self, rgb) -> None:
        self.canv.setFillColorRGB(*(np.asarray(rgb, dtype=float) / 255.0))


class DecisionGauge(_Chart):
    """Where the decision score falls between the two definite answers."""

    TRACK_Y, TRACK_H = 14.0, 6.0

    def __init__(self, score: float, width: float, fonts, limit: float = GAUGE_LIMIT):
        super().__init__(width, 38, fonts)
        self.score, self.limit = score, limit

    def _x(self, value: float) -> float:
        clamped = max(-self.limit, min(self.limit, value))
        return (clamped + self.limit) / (2 * self.limit) * self.width

    def draw(self):
        canvas, width = self.canv, self.width
        canvas.setFont(self.font, 7)
        canvas.setFillColor(PALETTE["gray_600"])
        canvas.drawString(0, 27, "← No pneumothorax")
        canvas.drawRightString(width, 27, "Pneumothorax →")

        canvas.saveState()
        path = canvas.beginPath()
        path.roundRect(0, self.TRACK_Y, width, self.TRACK_H, self.TRACK_H / 2)
        canvas.clipPath(path, stroke=0, fill=0)
        steps = 72
        step = width / steps
        for index, rgb in enumerate(diverging_colors(np.linspace(-1, 1, steps))):
            self._fill(rgb)
            canvas.rect(index * step, self.TRACK_Y, step + 0.5, self.TRACK_H, stroke=0, fill=1)
        canvas.restoreState()

        canvas.setStrokeColor(PALETTE["gray_500"])
        canvas.setLineWidth(0.75)
        centre = self._x(0)
        canvas.line(centre, self.TRACK_Y - 2, centre, self.TRACK_Y + self.TRACK_H + 2)

        canvas.setFont(self.font, 6.5)
        canvas.setFillColor(PALETTE["gray_500"])
        half = self.limit / 2
        for value in (-self.limit, -half, 0.0, half, self.limit):
            label = "0" if value == 0 else format_signed(value, 0)
            if value == -self.limit:
                canvas.drawString(0, 3, label)
            elif value == self.limit:
                canvas.drawRightString(width, 3, label)
            else:
                canvas.drawCentredString(self._x(value), 3, label)

        x, y = self._x(self.score), self.TRACK_Y + self.TRACK_H / 2
        canvas.setFillColor(PALETTE["white"])
        canvas.circle(x, y, 5.5, stroke=0, fill=1)
        canvas.setFillColor(PALETTE["ink"])
        canvas.circle(x, y, 4, stroke=0, fill=1)


class OverlayLegend(_Chart):
    """Colour key for the overlay, drawn as it appears over a mid-gray X-ray pixel."""

    def __init__(self, width: float, fonts):
        super().__init__(width, 22, fonts)

    def draw(self):
        canvas, width = self.canv, self.width
        steps = 72
        step = width / steps
        for index, rgb in enumerate(overlay_colors(np.linspace(-1, 1, steps))):
            self._fill(rgb)
            canvas.rect(index * step, 12, step + 0.4, 6, stroke=0, fill=1)
        canvas.setFont(self.font, 6.8)
        canvas.setFillColor(PALETTE["gray_600"])
        canvas.drawString(0, 2, "Raised ‘No pneumothorax’")
        canvas.drawCentredString(width / 2, 2, "No effect")
        canvas.drawRightString(width, 2, "Raised ‘Pneumothorax’")


class RegionChart(_Chart):
    """Back-to-back bars: each region's share of the total attribution, by direction."""

    HEADER, LABEL_WIDTH, VALUE_ROOM, BAR = 20.0, 88.0, 24.0, 9.0

    def __init__(self, regions, width: float, height: float, fonts):
        super().__init__(width, height, fonts)
        self.regions = regions
        self.ROW = (height - self.HEADER - 2) / max(1, len(regions))

    def _bar(self, x: float, y: float, length: float, rgb, grows_right: bool) -> None:
        if length < 0.3:
            return
        canvas = self.canv
        radius = min(2.5, length / 2, self.BAR / 2)
        self._fill(rgb)
        path = canvas.beginPath()
        path.roundRect(x, y, length, self.BAR, radius)
        canvas.drawPath(path, stroke=0, fill=1)
        # Square the end that sits on the baseline; only the data end is rounded.
        square_x = x if grows_right else x + length - radius
        canvas.rect(square_x, y, radius, self.BAR, stroke=0, fill=1)

    def draw(self):
        canvas = self.canv
        plot_left = self.LABEL_WIDTH
        centre = plot_left + (self.width - plot_left) / 2
        half = (self.width - plot_left) / 2 - self.VALUE_ROOM
        peak = max([max(toward, against) for _, toward, against in self.regions] + [1e-9])
        top = self.height - self.HEADER

        legend_y = self.height - 9
        canvas.setFont(self.font, 7)
        for text, rgb, right in (("Raised ‘No pneumothorax’", AGAINST_COLOR, False),
                                 ("Raised ‘Pneumothorax’", TOWARD_COLOR, True)):
            text_width = pdfmetrics.stringWidth(text, self.font, 7)
            swatch_x = centre + 6 if right else centre - 6 - text_width - 10
            self._fill(rgb)
            canvas.rect(swatch_x, legend_y - 0.5, 7, 7, stroke=0, fill=1)
            canvas.setFillColor(PALETTE["gray_600"])
            canvas.drawString(swatch_x + 10, legend_y, text)

        for index, (label, toward, against) in enumerate(self.regions):
            middle = top - index * self.ROW - self.ROW / 2
            canvas.setFont(self.font, 7.5)
            canvas.setFillColor(PALETTE["gray_700"])
            canvas.drawString(0, middle - 2.6, label)
            against_length, toward_length = against / peak * half, toward / peak * half
            y = middle - self.BAR / 2
            self._bar(centre - 1 - against_length, y, against_length, AGAINST_COLOR, False)
            self._bar(centre + 1, y, toward_length, TOWARD_COLOR, True)
            canvas.setFont(self.font, 7)
            canvas.setFillColor(PALETTE["gray_600"])
            label_y = middle - 2.4
            canvas.drawRightString(centre - 4 - against_length, label_y, format_percent(against))
            canvas.drawString(centre + 4 + toward_length, label_y, format_percent(toward))

        canvas.setStrokeColor(PALETTE["gray_400"])
        canvas.setLineWidth(0.75)
        canvas.line(centre, 1, centre, top + 2)


class IntensityHistogram(_Chart):
    """Distribution of pixel intensities, with the 1st–99th percentile range shaded."""

    AXIS = 12.0

    def __init__(self, fractions, width: float, height: float, fonts,
                 p1: Optional[float] = None, p99: Optional[float] = None):
        super().__init__(width, height, fonts)
        self.fractions = np.asarray(fractions, dtype=float)
        self.p1, self.p99 = p1, p99

    def _x(self, level: float) -> float:
        return level / 256.0 * self.width

    def draw(self):
        canvas, width = self.canv, self.width
        plot_height = self.height - self.AXIS - 2
        if self.p1 is not None and self.p99 is not None:
            canvas.setFillColor(PALETTE["histogram_band"])
            canvas.rect(self._x(self.p1), self.AXIS, self._x(self.p99 + 1) - self._x(self.p1),
                        plot_height, stroke=0, fill=1)
        peak = float(self.fractions.max()) if self.fractions.size else 0.0
        if peak > 0:
            step = width / self.fractions.size
            canvas.setFillColor(PALETTE["histogram"])
            for index, fraction in enumerate(self.fractions):
                height = fraction / peak * plot_height
                if height >= 0.25:
                    canvas.rect(index * step + 0.3, self.AXIS, max(step - 0.6, 0.3), height,
                                stroke=0, fill=1)
        canvas.setStrokeColor(PALETTE["gray_300"])
        canvas.setLineWidth(0.75)
        canvas.line(0, self.AXIS, width, self.AXIS)
        canvas.setFont(self.font, 6.5)
        canvas.setFillColor(PALETTE["gray_500"])
        canvas.drawString(0, 2, "0 black")
        for level in (64, 128, 192):
            canvas.drawCentredString(self._x(level), 2, str(level))
        canvas.drawRightString(width, 2, "255 white")
