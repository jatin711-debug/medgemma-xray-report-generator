"""Local report publication checks without importing the model runtime."""
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from reportlab.platypus import KeepTogether, Paragraph, Table

from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.core.heatmap import AttributionMap, relevance_grid, summarize_evidence
from src.core.image_quality import image_statistics
from src.reports.generator import (NOT_RECORDED, UltimatePDFGenerator, answer_preference,
                                   border_note, image_check_flags, outcome_tone,
                                   region_headline, score_agreement)
from src.reports.storage import atomic_report, list_local_reports, report_path


def page_count(path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", Path(path).read_bytes()))


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
                    yield from paragraph_texts(cell if isinstance(cell, list) else [cell])


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
        cases = (("Pneumothorax Detected", "Pneumothorax detected", 2.4),
                 ("No Pneumothorax Detected", "No pneumothorax detected", -2.3),
                 ("Indeterminate", "Possible small apical pneumothorax", 0.4))
        for size in ((900, 1100), (1400, 1000)):
            for impression, finding, score in cases:
                with self.subTest(size=size, impression=impression):
                    image = Image.new("RGB", size, (90, 90, 90))
                    response = full_response(impression, ONE_SENTENCE.format(finding=finding),
                                             score, image)
                    response.analysis_id = f"test-{size[0]}-{impression.split()[0].lower()}"
                    path = self.generator.generate_pdf_report(response, image,
                                                              signed_attribution(score))
                    self.assertEqual(page_count(path), 2)

    def test_long_responses_are_shortened_on_page_one_and_kept_in_full(self):
        reasoning = "Long reasoning sentence number {}. "
        raw = "**Finding**: No pneumothorax detected\n**Brief Reasoning**: " + "".join(
            reasoning.format(index) for index in range(30))
        image = Image.new("RGB", (900, 1100), (90, 90, 90))
        response = full_response("No Pneumothorax Detected", raw, -1.5, image)
        story = self.generator._story(response, image, signed_attribution(-1.5), 532)
        text = " ".join(paragraph_texts(story))
        self.assertIn("Shortened here", text)
        self.assertIn("Full model response", text)
        self.assertIn("number 29.", text)
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

    def test_outcome_colour_follows_model_output(self):
        self.assertEqual(outcome_tone("Pneumothorax Detected"), "danger")
        self.assertEqual(outcome_tone("No Pneumothorax Detected"), "success")
        self.assertEqual(outcome_tone("Indeterminate"), "neutral")

    def test_response_rows_skip_repeated_finding_and_strip_markdown(self):
        response = make_response(
            "No Pneumothorax Detected",
            "**Finding**: No pneumothorax detected.\n**Brief Reasoning**: **Clear** lungs.")
        self.assertEqual(self.generator._response_rows(response), [("Reasoning", "Clear lungs.")])

    def test_response_rows_keep_unexpected_wording_and_fall_back_to_raw_text(self):
        response = make_response("Indeterminate", "**Finding**: Possible small pneumothorax")
        self.assertEqual(self.generator._response_rows(response),
                         [("Finding", "Possible small pneumothorax")])
        self.assertEqual(self.generator._response_rows(make_response(raw="**free** text")),
                         [("Response", "free text")])

    def test_detail_rows_describe_model_runtime_and_image(self):
        response = make_response(
            model_info={"model_id": "model/x", "precision": "4-bit NF4 weights, bfloat16 compute",
                        "device": "cuda (GPU)", "app_version": "abc123", "torch": "2.8.0",
                        "transformers": "5.20.1"},
            timings={"generation_seconds": 2.5, "attribution_seconds": 0.75},
            attribution={"status": "available", "method": "Gradient × activation",
                         "target": "logit difference", "target_score": 1.25},
        )
        response.image_metadata.update({"dimensions": (640, 480), "format": "PNG", "size": 2048})
        rows = dict(self.generator._detail_rows(response))
        self.assertEqual(rows["Model"], "model/x")
        self.assertEqual(rows["Precision and device"],
                         "4-bit NF4 weights, bfloat16 compute on cuda (GPU)")
        self.assertEqual(rows["Timing"], "generation 2.50 s · attribution 0.75 s")
        self.assertEqual(rows["Image"], "<scan & input>.png · 640 × 480 px · PNG · 2 KB")
        self.assertIn("+1.250", rows["Attribution"])
        self.assertEqual(rows["Software"], "app abc123 · torch 2.8.0 · transformers 5.20.1")
        self.assertEqual(rows["Generated"], "January 01, 2026, 00:00:00 UTC")

    def test_missing_details_are_marked_not_recorded(self):
        response = make_response()
        response.image_metadata = {}
        rows = dict(self.generator._detail_rows(response))
        for key in ("Image", "Model", "Precision and device", "Timing", "Attribution", "Software"):
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

    def test_answer_preference_is_the_two_way_softmax(self):
        self.assertEqual(answer_preference(0.0), 0.5)
        self.assertAlmostEqual(answer_preference(-2.31), 0.0903, places=4)
        self.assertAlmostEqual(answer_preference(2.31) + answer_preference(-2.31), 1.0)
        self.assertEqual(answer_preference(1000.0), 1.0)
        self.assertEqual(answer_preference(-1000.0), 0.0)

    def test_score_agreement_compares_direction_with_the_answer(self):
        self.assertEqual(score_agreement("Pneumothorax Detected", 1.0)[0], "success")
        self.assertEqual(score_agreement("Pneumothorax Detected", -1.0)[0], "danger")
        self.assertEqual(score_agreement("No Pneumothorax Detected", -1.0)[0], "success")
        self.assertEqual(score_agreement("No Pneumothorax Detected", 1.0)[0], "danger")
        self.assertEqual(score_agreement("No Pneumothorax Detected", 0.0)[0], "neutral")
        self.assertEqual(score_agreement("Indeterminate", 3.0)[0], "neutral")

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
