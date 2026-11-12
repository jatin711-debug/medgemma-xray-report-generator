"""Local report publication checks without importing the model runtime."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.reports.generator import UltimatePDFGenerator
from src.reports.storage import atomic_report, list_local_reports, report_path


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


class ReportGenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.generator = UltimatePDFGenerator(self.directory.name)
        self.response = AnalysisResponse(
            analysis_id="test-id", timestamp="2026-01-01T00:00:00+00:00",
            findings=RadiologyFindings(anatomical_structures=[], abnormalities=[],
                impression="Indeterminate", recommendations=[], confidence_level="Not available"),
            raw_response="Uncertain <finding>", image_metadata={"filename": "<scan & input>.png"},
        )

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
        self.generator._draw_page_template(canvas, document)
        text = " ".join(str(call.args[-1]) for call in canvas.drawString.call_args_list
                        + canvas.drawRightString.call_args_list)
        self.assertIn("Research use only", text)
        self.assertNotIn("HIPAA", text)
        self.assertNotIn("Encrypted", text)
        self.assertNotIn("Clinical Decision Support", text)
