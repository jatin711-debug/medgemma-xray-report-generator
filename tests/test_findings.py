"""Parser regression checks requiring neither PyTorch nor a downloaded model."""
import unittest

from src.core.findings import parse_ai_response


class FindingsTests(unittest.TestCase):
    def test_recognized_findings(self):
        for text, impression in (
            ("**Finding**: No pneumothorax detected", "No Pneumothorax Detected"),
            ("**Finding**: Pneumothorax detected", "Pneumothorax Detected"),
        ):
            with self.subTest(text=text):
                result = parse_ai_response(text)
                self.assertEqual(result.impression, impression)
                self.assertEqual(result.confidence_level, "Not available (uncalibrated)")

    def test_ambiguous_missing_or_duplicate_findings_are_indeterminate(self):
        for text in (
            "", "uncertain", "**Finding**: possible pneumothorax",
            "**Finding**: Indeterminate", "**Finding**: No pneumothorax detected, but unsure",
            "**Finding**: No pneumothorax detected\n**Finding**: Pneumothorax detected",
            "**Finding**: Pneumothorax detected\n**Finding**: Pneumothorax detected",
            "Reasoning mentions **Finding**: Pneumothorax detected",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_ai_response(text).impression, "Indeterminate")

    def test_supported_formatting_and_reasoning_preserve_finding(self):
        result = parse_ai_response(
            "  **Finding**: **NO PNEUMOTHORAX DETECTED.**\r\n"
            "**Brief Reasoning**: Synthetic explanation."
        )
        self.assertEqual(result.impression, "No Pneumothorax Detected")
