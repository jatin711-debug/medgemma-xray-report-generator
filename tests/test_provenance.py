"""Image statistics and build identification; no PyTorch or model required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.core import version
from src.core.image_quality import image_statistics, intensity_histogram

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class ImageStatisticsTests(unittest.TestCase):
    def test_uniform_grayscale_image(self):
        stats = image_statistics(Image.new("L", (50, 40), 100))
        self.assertEqual(stats, {"grayscale": True, "mean_intensity": 100.0, "contrast_sd": 0.0,
                                 "intensity_p1": 100, "intensity_p99": 100,
                                 "black_pixel_pct": 0.0, "white_pixel_pct": 0.0})

    def test_pure_black_and_white_pixels_are_counted(self):
        image = Image.new("L", (10, 10), 0)
        image.paste(255, (0, 0, 10, 2))
        stats = image_statistics(image)
        self.assertEqual(stats["black_pixel_pct"], 80.0)
        self.assertEqual(stats["white_pixel_pct"], 20.0)

    def test_intensity_histogram_is_a_distribution(self):
        image = Image.new("L", (10, 10), 0)
        image.paste(255, (0, 0, 10, 5))
        histogram = intensity_histogram(image, bins=4)
        self.assertEqual(histogram.tolist(), [0.5, 0.0, 0.0, 0.5])

    def test_colour_image_and_contrast(self):
        image = Image.new("RGB", (20, 20), (255, 0, 0))
        image.paste((0, 0, 0), (0, 0, 20, 10))
        stats = image_statistics(image)
        self.assertFalse(stats["grayscale"])
        self.assertGreater(stats["contrast_sd"], 0)
        self.assertLess(stats["intensity_p1"], stats["intensity_p99"])

    def test_large_images_are_sampled(self):
        stats = image_statistics(Image.new("RGB", (5000, 3000), (30, 30, 30)))
        self.assertTrue(stats["grayscale"])
        self.assertEqual(stats["mean_intensity"], 30.0)


class VersionTests(unittest.TestCase):
    def setUp(self):
        version.app_version.cache_clear()
        self.addCleanup(version.app_version.cache_clear)

    def make_repo(self, head, refs=None, packed=None):
        root = Path(self.directory.name)
        git = root / ".git"
        git.mkdir()
        (git / "HEAD").write_text(head)
        for name, value in (refs or {}).items():
            (git / name).parent.mkdir(parents=True, exist_ok=True)
            (git / name).write_text(value)
        if packed:
            (git / "packed-refs").write_text(packed)
        return root

    def run_in_repo(self, **kwargs):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        return version._git_commit(self.make_repo(**kwargs))

    def test_reads_branch_ref_packed_ref_and_detached_head(self):
        self.assertEqual(self.run_in_repo(head="ref: refs/heads/main\n",
                                          refs={"refs/heads/main": COMMIT + "\n"}), COMMIT)
        self.assertEqual(self.run_in_repo(head="ref: refs/heads/dev\n",
                                          packed=f"# pack-refs\n{COMMIT} refs/heads/dev\n"), COMMIT)
        self.assertEqual(self.run_in_repo(head=COMMIT), COMMIT)

    def test_missing_or_invalid_repository_is_unknown(self):
        self.assertIsNone(self.run_in_repo(head="ref: refs/heads/missing\n"))
        self.assertIsNone(self.run_in_repo(head="not-a-commit"))
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(version._git_commit(Path(empty)))

    def test_environment_override_takes_precedence(self):
        with patch.dict("os.environ", {"APP_VERSION": "v1.2.3"}):
            self.assertEqual(version.app_version(), "v1.2.3")
        version.app_version.cache_clear()
        with patch.dict("os.environ", {"APP_VERSION": ""}), \
             patch.object(version, "_git_commit", return_value=COMMIT):
            self.assertEqual(version.app_version(), COMMIT[:12])
        version.app_version.cache_clear()
        with patch.dict("os.environ", {"APP_VERSION": "unknown"}), \
             patch.object(version, "_git_commit", return_value=None):
            self.assertEqual(version.app_version(), "unknown")


if __name__ == "__main__":
    unittest.main()
