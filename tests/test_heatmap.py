"""Attribution grid, summary and rendering checks; no PyTorch or model required."""
import unittest

import numpy as np
from PIL import Image

from src.core.heatmap import (AGAINST_COLOR, NEUTRAL_COLOR, TOWARD_COLOR, AttributionMap,
                              diverging_colors, overlay_colors, relevance_grid, render_overlay,
                              render_token_grid, summarize_evidence)


class RelevanceGridTests(unittest.TestCase):
    def test_keeps_both_signs_scaled_to_the_strongest_token(self):
        tokens = np.zeros(256, dtype=np.float32)
        tokens[0], tokens[17], tokens[255] = 4.0, 2.0, -8.0
        grid = relevance_grid(tokens)
        self.assertEqual(grid.shape, (16, 16))
        self.assertEqual(grid[15, 15], -1.0)
        self.assertEqual(grid[0, 0], 0.5)
        self.assertEqual(grid[1, 1], 0.25)
        self.assertEqual(float(np.abs(grid).max()), 1.0)

    def test_direction_views_split_the_signed_grid(self):
        grid = relevance_grid(np.array([1.0, -0.5, 0.0, 0.0]))
        attribution = AttributionMap(grid, 0.3, "m", "t")
        self.assertEqual(attribution.toward.reshape(-1).tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(attribution.against.reshape(-1).tolist(), [0.0, 0.5, 0.0, 0.0])
        self.assertTrue(attribution.has_toward and attribution.has_against)
        self.assertFalse(attribution.is_empty)

    def test_one_sided_evidence_leaves_the_other_direction_blank(self):
        grid = relevance_grid(-np.abs(np.random.default_rng(0).standard_normal(256)))
        attribution = AttributionMap(grid, -1.0, "m", "t")
        self.assertFalse(attribution.has_toward)
        self.assertTrue(attribution.has_against)
        self.assertTrue(AttributionMap(np.zeros((16, 16)), 0.0, "m", "t").is_empty)

    def test_non_finite_values_are_ignored(self):
        tokens = np.full(16, np.nan, dtype=np.float32)
        tokens[3] = 2.0
        tokens[4] = np.inf
        tokens[5] = -np.inf
        grid = relevance_grid(tokens)
        self.assertEqual(grid.reshape(-1).tolist(), [0, 0, 0, 1] + [0] * 12)

    def test_rejects_non_square_token_counts(self):
        for size in (0, 3, 255):
            with self.subTest(size=size), self.assertRaises(ValueError):
                relevance_grid(np.ones(size))


class EvidenceSummaryTests(unittest.TestCase):
    def test_blank_grid_has_no_summary(self):
        self.assertIsNone(summarize_evidence(np.zeros((16, 16))))

    def test_shares_by_direction_and_region_sum_to_one(self):
        grid = np.zeros((16, 16))
        grid[0:3, 0:4] = 1.0     # Upper, image left: toward pneumothorax.
        grid[12:16, 10:16] = -0.5  # Lower, image right: toward no pneumothorax.
        summary = summarize_evidence(grid)
        total = 12 * 1.0 + 24 * 0.5
        self.assertAlmostEqual(summary.toward_share, 12 / total)
        self.assertAlmostEqual(summary.against_share, 12 / total)
        regions = {label: (toward, against) for label, toward, against in summary.regions}
        self.assertEqual(len(regions), 6)
        self.assertAlmostEqual(regions["Upper, image left"][0], 0.5)
        self.assertAlmostEqual(regions["Lower, image right"][1], 0.5)
        self.assertAlmostEqual(sum(t + a for t, a in regions.values()), 1.0)

    def test_tokens_split_by_a_region_boundary_are_shared_by_area(self):
        grid = np.zeros((16, 16))
        grid[5, 0] = 1.0  # Row 5 spans the upper/middle boundary at 16 / 3 rows.
        regions = {label: toward for label, toward, _ in summarize_evidence(grid).regions}
        self.assertAlmostEqual(regions["Upper, image left"], 1 / 3)
        self.assertAlmostEqual(regions["Middle, image left"], 2 / 3)

    def test_concentration_and_border_share(self):
        grid = np.zeros((16, 16))
        grid[0, :] = 1.0  # All evidence on the top edge.
        summary = summarize_evidence(grid)
        self.assertEqual(summary.top_token_count, 26)
        self.assertEqual(summary.token_count, 256)
        self.assertAlmostEqual(summary.top_token_share, 1.0)
        self.assertAlmostEqual(summary.border_share, 1.0)
        self.assertAlmostEqual(summary.border_area, 60 / 256)

        uniform = summarize_evidence(np.ones((16, 16)))
        self.assertAlmostEqual(uniform.top_token_share, 26 / 256)
        self.assertAlmostEqual(uniform.border_share, 60 / 256)


class RenderingTests(unittest.TestCase):
    def test_diverging_colours_run_from_blue_through_gray_to_red(self):
        self.assertEqual(tuple(diverging_colors(np.array(-1.0))), AGAINST_COLOR)
        self.assertEqual(tuple(diverging_colors(np.array(0.0))), NEUTRAL_COLOR)
        self.assertEqual(tuple(diverging_colors(np.array(1.0))), TOWARD_COLOR)
        self.assertEqual(tuple(diverging_colors(np.array(5.0))), TOWARD_COLOR)
        self.assertEqual(tuple(overlay_colors(np.array(0.0))), (128, 128, 128))

    def test_blank_grid_leaves_the_grayscale_image_unchanged(self):
        image = Image.new("RGB", (40, 30), (200, 100, 50))
        overlay = render_overlay(image, np.zeros((16, 16), dtype=np.float32))
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.tobytes(), image.convert("L").convert("RGB").tobytes())

    def test_overlay_colours_each_direction_in_its_own_hue(self):
        image = Image.new("RGB", (160, 160), (128, 128, 128))
        grid = np.zeros((16, 16), dtype=np.float32)
        grid[0, 0], grid[0, 15] = 1.0, -1.0
        overlay = render_overlay(image, grid)
        red, _, blue = overlay.getpixel((3, 3))
        self.assertGreater(red, blue)
        red, _, blue = overlay.getpixel((156, 3))
        self.assertGreater(blue, red)
        self.assertEqual(overlay.getpixel((80, 150)), (128, 128, 128))

    def test_token_grid_keeps_aspect_ratio_and_shows_flat_cells(self):
        image = Image.new("RGB", (300, 400), (0, 0, 0))
        grid = np.zeros((16, 16), dtype=np.float32)
        grid[8, 8] = 1.0
        rendered = render_token_grid(image, grid, width=320)
        self.assertEqual(rendered.size, (320, 427))
        cell_centre = (8 * 20 + 10, int(8.5 * 427 / 16))
        red, green, blue = rendered.getpixel(cell_centre)
        self.assertGreater(red, green)
        self.assertGreater(red, blue)
        # A neighbouring cell with no relevance shows only the faded image.
        self.assertEqual(len(set(rendered.getpixel((5 * 20 + 10, 15)))), 1)


if __name__ == "__main__":
    unittest.main()
