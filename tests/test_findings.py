"""Parser regression checks requiring neither PyTorch nor a downloaded model."""
import unittest

from src.core.findings import clean_markdown, parse_ai_response, split_response


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


class ResponseDisplayTests(unittest.TestCase):
    def test_split_response_strips_markdown_and_separates_lines(self):
        sections = split_response(
            "**Finding**: No pneumothorax detected\n"
            "**Brief Reasoning**: Lung markings reach the **periphery**."
        )
        self.assertEqual(sections, {"finding": "No pneumothorax detected",
                                    "reasoning": "Lung markings reach the periphery."})

    def test_split_response_accepts_label_variants_and_missing_sections(self):
        self.assertEqual(split_response("Reasoning: short note")["reasoning"], "short note")
        self.assertEqual(split_response("__Finding:__ Indeterminate")["finding"], "Indeterminate")
        self.assertEqual(split_response("free text only"), {"finding": None, "reasoning": None})
        self.assertEqual(split_response("**Finding**:   ")["finding"], None)

    def test_clean_markdown_removes_formatting_marks(self):
        self.assertEqual(clean_markdown("## Heading\n- **bold** `code`  item"),
                         "Heading\nbold code item")
