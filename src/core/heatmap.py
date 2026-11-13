"""Attribution map data, summaries and rendering, without importing the model runtime.

MedGemma's image processor resizes every input to a square before encoding,
and the language model receives a square grid of image tokens covering that
whole square. Stretching the grid back over the original image therefore
lines each cell up with the region it was computed from.

Relevance is signed: positive tokens raised the score for "Pneumothorax
detected", negative tokens raised "No pneumothorax detected". Values are scaled
within one image only, so they compare regions of that image, not images.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

# Diverging pair (red raises 'Pneumothorax detected', blue raises 'No pneumothorax
# detected') with a neutral gray midpoint, for light backgrounds.
TOWARD_COLOR = (227, 73, 72)      # #e34948
AGAINST_COLOR = (42, 120, 214)    # #2a78d6
NEUTRAL_COLOR = (240, 239, 236)   # #f0efec
# Lighter steps of the same two hues, for drawing over a dark radiograph.
TOWARD_OVERLAY = (230, 103, 103)  # #e66767
AGAINST_OVERLAY = (57, 135, 229)  # #3987e5
MAX_OVERLAY_ALPHA = 0.65
MAX_CELL_ALPHA = 0.85

# Fixed thirds and halves of the image frame. They are not anatomical lung zones.
REGION_ROWS = ("Upper", "Middle", "Lower")
REGION_COLUMNS = ("image left", "image right")
TOP_TOKEN_FRACTION = 0.10


@dataclass(frozen=True)
class AttributionMap:
    """Signed relevance per image token, scaled so the strongest token has magnitude 1.

    ``target_score`` is the raw, uncalibrated logit difference the map explains.
    """
    grid: np.ndarray
    target_score: float
    method: str
    target: str

    @property
    def toward(self) -> np.ndarray:
        """Evidence that raised 'Pneumothorax detected' (values in [0, 1])."""
        return np.clip(self.grid, 0.0, None)

    @property
    def against(self) -> np.ndarray:
        """Evidence that raised 'No pneumothorax detected' (values in [0, 1])."""
        return np.clip(-self.grid, 0.0, None)

    @property
    def has_toward(self) -> bool:
        return bool(np.any(self.grid > 0))

    @property
    def has_against(self) -> bool:
        return bool(np.any(self.grid < 0))

    @property
    def is_empty(self) -> bool:
        return not bool(np.any(self.grid != 0))


@dataclass(frozen=True)
class EvidenceSummary:
    """Where the attribution lies, as shares of the total absolute relevance.

    ``regions`` holds ``(label, toward_share, against_share)`` for each region;
    all toward and against shares together sum to 1.
    """
    toward_share: float
    against_share: float
    regions: tuple[tuple[str, float, float], ...]
    top_token_share: float
    top_token_count: int
    token_count: int
    border_share: float
    border_area: float


def relevance_grid(token_relevance) -> np.ndarray:
    """Arrange signed per-token relevance into a square grid with values in [-1, 1].

    Values are scaled by the strongest token of either sign in this image, so the
    two directions stay comparable. Non-finite values count as no contribution.
    """
    values = np.nan_to_num(np.asarray(token_relevance, dtype=np.float32).reshape(-1),
                           nan=0.0, posinf=0.0, neginf=0.0)
    side = math.isqrt(values.size)
    if values.size == 0 or side * side != values.size:
        raise ValueError(f"Expected a square number of image tokens, got {values.size}")
    peak = float(np.abs(values).max())
    if peak > 0:
        values = values / peak
    return np.clip(values, -1.0, 1.0).reshape(side, side)


def summarize_evidence(grid: np.ndarray) -> Optional[EvidenceSummary]:
    """Summarise a signed grid by direction, region, concentration and image border.

    Returns ``None`` when every token is zero. Regions are area-weighted, so a
    token split by a region boundary contributes to both regions in proportion.
    """
    values = np.asarray(grid, dtype=np.float64)
    magnitude = np.abs(values)
    total = float(magnitude.sum())
    if total <= 0:
        return None
    rows, columns = len(REGION_ROWS), len(REGION_COLUMNS)

    def region_totals(part: np.ndarray) -> np.ndarray:
        # Repeating each token rows x columns times makes the regions whole cells.
        expanded = np.repeat(np.repeat(part, rows, axis=0), columns, axis=1)
        height, width = expanded.shape[0] // rows, expanded.shape[1] // columns
        cells = expanded.reshape(rows, height, columns, width).sum(axis=(1, 3))
        return cells / (rows * columns * total)

    toward = region_totals(np.clip(values, 0.0, None))
    against = region_totals(np.clip(-values, 0.0, None))
    regions = tuple((f"{REGION_ROWS[row]}, {REGION_COLUMNS[column]}",
                     float(toward[row, column]), float(against[row, column]))
                    for row in range(rows) for column in range(columns))

    flat = np.sort(magnitude.reshape(-1))[::-1]
    top_count = max(1, int(round(TOP_TOKEN_FRACTION * flat.size)))
    border = np.ones(values.shape, dtype=bool)
    border[1:-1, 1:-1] = False
    return EvidenceSummary(
        toward_share=float(np.clip(values, 0.0, None).sum() / total),
        against_share=float(np.clip(-values, 0.0, None).sum() / total),
        regions=regions,
        top_token_share=float(flat[:top_count].sum() / total),
        top_token_count=top_count,
        token_count=int(flat.size),
        border_share=float(magnitude[border].sum() / total),
        border_area=float(border.mean()),
    )


def diverging_colors(values) -> np.ndarray:
    """Map values in [-1, 1] to RGB (uint8): blue below zero, neutral gray at zero, red above."""
    clipped = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)[..., None]
    neutral = np.array(NEUTRAL_COLOR, dtype=np.float32)
    pole = np.where(clipped >= 0, np.array(TOWARD_COLOR, dtype=np.float32),
                    np.array(AGAINST_COLOR, dtype=np.float32))
    return (neutral + (pole - neutral) * np.abs(clipped)).round().astype(np.uint8)


def overlay_colors(values, background: float = 128.0,
                   max_alpha: float = MAX_OVERLAY_ALPHA) -> np.ndarray:
    """RGB (uint8) of the overlay at ``values`` over a flat gray pixel, for legends."""
    values = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    alpha = (np.abs(values) * max_alpha)[..., None]
    pole = np.where(values[..., None] >= 0, np.array(TOWARD_OVERLAY, dtype=np.float32),
                    np.array(AGAINST_OVERLAY, dtype=np.float32))
    return (background * (1.0 - alpha) + pole * alpha).round().astype(np.uint8)


def upsample_grid(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Smoothly resize a signed grid to ``size`` (width, height), clipped to [-1, 1]."""
    source = Image.fromarray(np.clip(np.asarray(grid, dtype=np.float32), -1.0, 1.0))
    resized = source.resize(size, Image.Resampling.BICUBIC)
    return np.clip(np.asarray(resized, dtype=np.float32), -1.0, 1.0)


def render_overlay(image: Image.Image, grid: np.ndarray,
                   max_alpha: float = MAX_OVERLAY_ALPHA) -> Image.Image:
    """Blend the signed relevance map over a grayscale copy of ``image``.

    Opacity follows the magnitude of relevance, so regions with no contribution
    show the unaltered X-ray.
    """
    base = np.asarray(image.convert("L").convert("RGB"), dtype=np.float32)
    values = upsample_grid(grid, (base.shape[1], base.shape[0]))
    alpha = (np.abs(values) * max_alpha)[..., None]
    pole = np.where(values[..., None] >= 0, np.array(TOWARD_OVERLAY, dtype=np.float32),
                    np.array(AGAINST_OVERLAY, dtype=np.float32))
    blended = base * (1.0 - alpha) + pole * alpha
    return Image.fromarray(blended.round().astype(np.uint8))


def render_token_grid(image: Image.Image, grid: np.ndarray, width: int = 600,
                      gap: int = 2, fade: float = 0.72) -> Image.Image:
    """Draw each image token as a flat cell over a faded copy of ``image``.

    No smoothing is applied, so the picture shows the map's true resolution.
    Thin white gaps separate the tokens and darker lines mark the region
    boundaries used by :func:`summarize_evidence`.
    """
    values = np.clip(np.asarray(grid, dtype=np.float32), -1.0, 1.0)
    height = max(1, round(width * image.height / max(1, image.width)))
    base = np.asarray(image.convert("L").resize((width, height), Image.Resampling.BILINEAR),
                      dtype=np.float32)
    faded = 255.0 - (255.0 - base) * (1.0 - fade)
    canvas = np.repeat(faded[..., None], 3, axis=2)

    rows, columns = values.shape
    row_of = np.minimum((np.arange(height) * rows) // height, rows - 1)
    column_of = np.minimum((np.arange(width) * columns) // width, columns - 1)
    cell = values[row_of][:, column_of]
    alpha = (np.abs(cell) * MAX_CELL_ALPHA)[..., None]
    pole = np.where(cell[..., None] >= 0, np.array(TOWARD_COLOR, dtype=np.float32),
                    np.array(AGAINST_COLOR, dtype=np.float32))
    canvas = canvas * (1.0 - alpha) + pole * alpha

    def boundary(index_of: np.ndarray, thickness: int) -> np.ndarray:
        edges = np.flatnonzero(np.diff(index_of)) + 1
        mask = np.zeros(index_of.size, dtype=bool)
        for edge in edges:
            mask[max(0, edge - thickness // 2): edge + (thickness - thickness // 2)] = True
        return mask

    canvas[boundary(row_of, gap), :] = 255.0
    canvas[:, boundary(column_of, gap)] = 255.0

    region_row = np.minimum((np.arange(height) * len(REGION_ROWS)) // height, len(REGION_ROWS) - 1)
    region_column = np.minimum((np.arange(width) * len(REGION_COLUMNS)) // width,
                               len(REGION_COLUMNS) - 1)
    divider = (55.0, 65.0, 81.0)
    canvas[boundary(region_row, gap + 1), :] = divider
    canvas[:, boundary(region_column, gap + 1)] = divider
    return Image.fromarray(canvas.round().astype(np.uint8))
