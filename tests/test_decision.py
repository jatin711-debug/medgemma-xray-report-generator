import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from src.core import decision as decision_module
from src.core.decision import (DETECTED, INDETERMINATE, NEGATIVE, REVIEW, Calibration, decide,
                               find_calibration, load_calibrations)
from src.core.findings import answer_agrees, findings_from_decision, text_answer
from src.core.medical_analyzer import MedicalImageAnalyzer
from src.core.prompt import USER_PROMPT, build_messages, prompt_fingerprint


def entry(**overrides):
    base = {"model_id": "m", "precision": "bfloat16", "prompt_fingerprint": "f",
            "review_threshold": -4.0, "detected_threshold": 2.0,
            "cross_validated": {"review": {"sensitivity": [0.9, 0.85, 0.94]}}}
    base.update(overrides)
    return base


CALIBRATION = Calibration(-4.0, 2.0, entry())


class DecisionBandTests(unittest.TestCase):
    def test_scores_fall_into_three_bands_with_inclusive_lower_edges(self):
        cases = [(5.0, DETECTED, "detected"), (2.0, DETECTED, "detected"),
                 (1.99, REVIEW, "review"), (-4.0, REVIEW, "review"),
                 (-4.01, NEGATIVE, "negative")]
        for score, outcome, band in cases:
            with self.subTest(score=score):
                result = decide(score, CALIBRATION)
                self.assertEqual((result.outcome, result.band), (outcome, band))

    def test_zero_is_not_a_cutoff(self):
        # The old behaviour reported 'no pneumothorax' for every negative score.
        self.assertEqual(decide(-1.5, CALIBRATION).band, "review")

    def test_missing_score_or_calibration_is_indeterminate(self):
        for score in (None, math.nan, math.inf):
            with self.subTest(score=score):
                self.assertEqual(decide(score, CALIBRATION).band, "unscored")
        uncalibrated = decide(3.0, None)
        self.assertEqual((uncalibrated.outcome, uncalibrated.band), (INDETERMINATE, "uncalibrated"))
        self.assertEqual(uncalibrated.as_dict()["score"], 3.0)

    def test_as_dict_states_thresholds_and_provenance(self):
        result = decide(0.5, CALIBRATION).as_dict()
        self.assertEqual(result["review_threshold"], -4.0)
        self.assertEqual(result["detected_threshold"], 2.0)
        self.assertEqual(result["calibration"]["model_id"], "m")


class CalibrationLookupTests(unittest.TestCase):
    def test_only_an_exact_model_precision_and_prompt_match_is_used(self):
        entries = [entry(precision="4-bit NF4 weights, bfloat16 compute", review_threshold=-9.0),
                   entry()]
        found = find_calibration("m", "bfloat16", "f", entries)
        self.assertEqual(found.review_threshold, -4.0)
        self.assertIsNone(find_calibration("m", "bfloat16", "other-prompt", entries))
        self.assertIsNone(find_calibration("other-model", "bfloat16", "f", entries))
        self.assertIsNone(find_calibration("m", "float32", "f", entries))

    def test_malformed_or_inverted_entries_are_skipped(self):
        bad = [entry(review_threshold="x"), entry(review_threshold=3.0, detected_threshold=2.0),
               entry(detected_threshold=math.nan), {"model_id": "m"}]
        self.assertIsNone(find_calibration("m", "bfloat16", "f", bad))

    def test_missing_or_corrupt_file_yields_no_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            self.assertEqual(load_calibrations(path), [])
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_calibrations(path), [])
            path.write_text(json.dumps({"entries": [entry(), "junk"]}), encoding="utf-8")
            self.assertEqual(load_calibrations(path), [entry()])

    def test_environment_can_point_at_another_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.json"
            path.write_text(json.dumps({"entries": [entry(review_threshold=-1.0)]}), encoding="utf-8")
            with patch.dict("os.environ", {"MEDGEMMA_CALIBRATION": str(path)}):
                self.assertEqual(find_calibration("m", "bfloat16", "f").review_threshold, -1.0)

    def test_shipped_calibration_matches_the_current_prompt(self):
        entries = load_calibrations(decision_module.CALIBRATION_PATH)
        self.assertTrue(entries, "src/core/calibration.json must ship with at least one entry")
        for item in entries:
            with self.subTest(precision=item["precision"]):
                self.assertEqual(item["prompt_fingerprint"], prompt_fingerprint(),
                                 "The prompt changed; re-run the calibration (see README).")
                self.assertLessEqual(item["review_threshold"], item["detected_threshold"])


class PromptTests(unittest.TestCase):
    def test_fingerprint_tracks_the_prompt_text(self):
        original = prompt_fingerprint()
        with patch("src.core.prompt.USER_PROMPT", USER_PROMPT + " "):
            self.assertNotEqual(prompt_fingerprint(), original)

    def test_messages_place_the_image_after_the_instructions(self):
        image = object()
        messages = build_messages(image)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIs(messages[1]["content"][1]["image"], image)


class FindingsFromDecisionTests(unittest.TestCase):
    def test_outcome_comes_from_the_score_not_the_generated_text(self):
        raw = "**Finding**: No pneumothorax detected\n**Brief Reasoning**: Clear."
        findings = findings_from_decision(decide(-1.0, CALIBRATION), raw)
        self.assertEqual(findings.impression, REVIEW)
        self.assertEqual(findings.model_text_answer, "No pneumothorax detected")
        self.assertIn("radiologist", findings.recommendations[0])
        self.assertIn("90%", findings.confidence_level)

    def test_negative_band_does_not_ask_for_targeted_review(self):
        findings = findings_from_decision(decide(-9.0, CALIBRATION), "")
        self.assertEqual(findings.impression, NEGATIVE)
        self.assertEqual(findings.recommendations, ["Research use only; expert review required."])
        self.assertIsNone(findings.model_text_answer)

    def test_uncalibrated_findings_say_why(self):
        findings = findings_from_decision(decide(1.0, None), "")
        self.assertEqual(findings.impression, INDETERMINATE)
        self.assertIn("no matching calibration", findings.confidence_level)

    def test_text_answer_and_agreement(self):
        self.assertEqual(text_answer("**Finding**: Pneumothorax detected."), "Pneumothorax detected")
        self.assertIsNone(text_answer("**Finding**: Indeterminate"))
        self.assertTrue(answer_agrees("Pneumothorax detected", "detected"))
        self.assertFalse(answer_agrees("No pneumothorax detected", "detected"))
        self.assertFalse(answer_agrees("Pneumothorax detected", "negative"))
        self.assertTrue(answer_agrees("No pneumothorax detected", "review"))
        self.assertIsNone(answer_agrees(None, "detected"))
        self.assertIsNone(answer_agrees("Pneumothorax detected", "uncalibrated"))


class AnalyzerDecisionTests(unittest.TestCase):
    def _analyzer(self, precision="bfloat16"):
        analyzer = MedicalImageAnalyzer()
        analyzer.model_id = "m"
        analyzer.runtime = {"precision": precision, "device": "cpu"}
        analyzer.processor = MagicMock()
        analyzer.processor.decode.return_value = "**Finding**: No pneumothorax detected"
        analyzer.model = MagicMock()
        return analyzer

    def _run(self, analyzer, score):
        entries = [entry(prompt_fingerprint=prompt_fingerprint())]
        with patch.dict("os.environ", {"MEDGEMMA_ATTRIBUTION": "0"}), \
             patch("src.core.medical_analyzer.compute_pneumothorax_score", **score), \
             patch("src.core.decision.load_calibrations", return_value=entries):
            result = analyzer._analyze_image(Image.new("RGB", (8, 8)))
        return result, analyzer._make_response(result, {}, Image.new("L", (8, 8)))

    def test_review_band_overrides_a_negative_text_answer(self):
        result, response = self._run(self._analyzer(), {"return_value": -1.0})
        self.assertEqual(response.findings.impression, REVIEW)
        self.assertEqual(response.decision["band"], "review")
        self.assertEqual(response.decision["text_answer"], "No pneumothorax detected")
        self.assertTrue(response.decision["text_agrees"])
        self.assertIn("scoring_seconds", response.timings)

    def test_scoring_failure_is_indeterminate_not_an_error(self):
        result, response = self._run(self._analyzer(), {"side_effect": RuntimeError("OOM")})
        self.assertEqual(response.findings.impression, INDETERMINATE)
        self.assertEqual(response.decision["band"], "unscored")

    def test_other_precision_is_uncalibrated(self):
        result, response = self._run(self._analyzer("float32"), {"return_value": 5.0})
        self.assertEqual(response.decision["band"], "uncalibrated")
        self.assertEqual(response.decision["score"], 5.0)


if __name__ == "__main__":
    unittest.main()
