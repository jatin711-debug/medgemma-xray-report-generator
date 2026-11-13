"""Basic, descriptive input-image statistics (not a diagnostic quality assessment)."""
import numpy as np
from PIL import Image

_STATS_MAX_SIDE = 1024


def _luminance(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return the RGB and 8-bit luminance arrays of a downscaled copy of ``image``."""
    sample = image.convert("RGB")
    sample.thumbnail((_STATS_MAX_SIDE, _STATS_MAX_SIDE))
    return np.asarray(sample, dtype=np.int16), np.asarray(sample.convert("L"), dtype=np.float32)


def image_statistics(image: Image.Image) -> dict:
    """Describe colour, brightness, contrast and clipping of an image on a 0-255 scale.

    Statistics are computed on a downscaled copy so very large uploads stay cheap.
    """
    rgb, luminance = _luminance(image)
    grayscale = bool(np.abs(rgb[..., 0] - rgb[..., 1]).max() <= 2
                     and np.abs(rgb[..., 1] - rgb[..., 2]).max() <= 2)
    low, high = np.percentile(luminance, [1, 99])
    return {
        "grayscale": grayscale,
        "mean_intensity": round(float(luminance.mean()), 1),
        "contrast_sd": round(float(luminance.std()), 1),
        "intensity_p1": int(round(float(low))),
        "intensity_p99": int(round(float(high))),
        "black_pixel_pct": round(float((luminance <= 0).mean() * 100), 2),
        "white_pixel_pct": round(float((luminance >= 255).mean() * 100), 2),
    }


def intensity_histogram(image: Image.Image, bins: int = 64) -> np.ndarray:
    """Return the fraction of pixels in each of ``bins`` equal intensity ranges over 0-255."""
    _, luminance = _luminance(image)
    counts, _ = np.histogram(luminance, bins=bins, range=(0, 256))
    return counts / max(1, counts.sum())
