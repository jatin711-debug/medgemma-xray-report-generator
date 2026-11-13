"""Turn MedGemma's decision score into a reported outcome, without model imports.

The score is logit('Pneumothorax detected') - logit('No pneumothorax detected')
at the answer token. Its zero point is not meaningful on its own: on
radiologist-labelled data the model leans heavily one way, so a fixed cutoff
at 0 misses real cases. Thresholds therefore come from src/core/calibration.json,
fitted for one model, numeric precision and prompt. When nothing matches the
running configuration the score cannot be interpreted and the outcome is
indeterminate.
"""
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

DETECTED = "Pneumothorax Suspected"
REVIEW = "Possible Pneumothorax: Review Required"
NEGATIVE = "No Pneumothorax Flagged"
INDETERMINATE = "Indeterminate"


@dataclass(frozen=True)
class Calibration:
    """One fitted pair of thresholds and the evidence behind them."""
    review_threshold: float
    detected_threshold: float
    data: dict = field(repr=False)

    @property
    def summary(self) -> dict[str, Any]:
        """The fields a report needs to state how the thresholds were set."""
        keys = ("model_id", "precision", "prompt_fingerprint", "targets", "reference", "auc",
                "prevalence", "cross_validated", "created")
        return {key: self.data[key] for key in keys if key in self.data}


@dataclass(frozen=True)
class Decision:
    outcome: str
    band: str  # 'detected', 'review', 'negative', 'unscored' or 'uncalibrated'
    score: Optional[float]
    calibration: Optional[Calibration]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"outcome": self.outcome, "band": self.band,
                                  "score": None if self.score is None else round(self.score, 4),
                                  "reason": self.reason}
        if self.calibration is not None:
            result.update(review_threshold=self.calibration.review_threshold,
                          detected_threshold=self.calibration.detected_threshold,
                          calibration=self.calibration.summary)
        return result


def calibration_path() -> Path:
    override = os.getenv("MEDGEMMA_CALIBRATION")
    return Path(override) if override else CALIBRATION_PATH


def load_calibrations(path: Optional[Path] = None) -> list[dict]:
    """Read every calibration entry; a missing or unreadable file yields none."""
    try:
        content = json.loads((path or calibration_path()).read_text(encoding="utf-8"))
        entries = content["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def find_calibration(model_id: str, precision: str, fingerprint: str,
                     entries: Optional[list[dict]] = None) -> Optional[Calibration]:
    """Return the entry fitted for exactly this model, precision and prompt."""
    for entry in load_calibrations() if entries is None else entries:
        if (entry.get("model_id"), entry.get("precision"), entry.get("prompt_fingerprint")) != (
                model_id, precision, fingerprint):
            continue
        try:
            review = float(entry["review_threshold"])
            detected = float(entry["detected_threshold"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(review) and math.isfinite(detected) and review <= detected:
            return Calibration(review, detected, entry)
    return None


def decide(score: Optional[float], calibration: Optional[Calibration]) -> Decision:
    """Place a score in the detected, review or negative band."""
    if score is None or not math.isfinite(score):
        return Decision(INDETERMINATE, "unscored", None, calibration,
                        "The decision score could not be computed.")
    if calibration is None:
        return Decision(INDETERMINATE, "uncalibrated", score, None,
                        "No calibration matches this model, precision and prompt, so the score "
                        "cannot be interpreted.")
    if score >= calibration.detected_threshold:
        return Decision(DETECTED, "detected", score, calibration,
                        "The score is at or above the suspected threshold.")
    if score >= calibration.review_threshold:
        return Decision(REVIEW, "review", score, calibration,
                        "The score is in the review band: below the suspected threshold but high "
                        "enough that a pneumothorax cannot be ruled out.")
    return Decision(NEGATIVE, "negative", score, calibration,
                    "The score is below the review threshold.")
