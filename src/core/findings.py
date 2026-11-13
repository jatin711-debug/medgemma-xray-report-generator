"""Interpret the requested finding format without importing the model runtime."""
import re

from src.api.schemas import RadiologyFindings
from src.core.decision import Decision


_SECTION_LABELS = {"finding": r"Finding", "reasoning": r"(?:Brief\s+)?Reasoning"}
_ANSWERS = {"pneumothorax detected": "Pneumothorax detected",
            "no pneumothorax detected": "No pneumothorax detected"}


def clean_markdown(text: str) -> str:
    """Remove Markdown emphasis, headings, bullets and code marks from model text."""
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text)
    text = re.sub(r"(?m)^[ \t]*[-*][ \t]+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_response(raw_response: str) -> dict[str, str | None]:
    """Return the model's finding and reasoning lines as plain text, when present."""
    sections: dict[str, str | None] = {}
    for key, label in _SECTION_LABELS.items():
        match = re.search(rf"(?im)^[ \t]*[*_]*{label}[*_]*[ \t]*:[*_]*[ \t]*(\S[^\r\n]*)$",
                          raw_response)
        value = clean_markdown(match.group(1)) if match else ""
        sections[key] = value or None
    return sections


def text_answer(raw_response: str) -> str | None:
    """Return the generated answer when it is exactly one of the two definite options."""
    matches = re.findall(r"(?im)^[ \t]*\*\*Finding\*\*[ \t]*:[ \t]*([^\r\n]*)", raw_response)
    value = matches[0].strip().strip("* ").lower().rstrip(".") if len(matches) == 1 else ""
    return _ANSWERS.get(value)


def answer_agrees(answer: str | None, band: str) -> bool | None:
    """Whether the generated answer points the same way as the calibrated band.

    The review band agrees with either answer, since it means neither can be relied on.
    """
    if answer is None or band not in {"detected", "review", "negative"}:
        return None
    if band == "review":
        return True
    return (answer == "Pneumothorax detected") == (band == "detected")


def findings_from_decision(decision: Decision, raw_response: str) -> RadiologyFindings:
    """Report the calibrated outcome; the generated text is recorded but does not decide."""
    abnormality = {
        "detected": "Calibrated decision score is in the suspected band for pneumothorax.",
        "review": "Calibrated decision score is in the review band; pneumothorax is not ruled out.",
        "negative": "Calibrated decision score is below the review threshold.",
    }.get(decision.band, decision.reason)
    recommendations = ["Research use only; expert review required."]
    if decision.band in {"detected", "review"}:
        recommendations.insert(0, "Have a radiologist review this image for pneumothorax.")
    sensitivity = None
    if decision.calibration is not None:
        review = decision.calibration.data.get("cross_validated", {}).get("review", {})
        sensitivity = review.get("sensitivity")
    if sensitivity and len(sensitivity) == 3 and None not in sensitivity:
        confidence = (f"Calibrated bands; the review threshold flagged {sensitivity[0]:.0%} of "
                      f"pneumothoraces in held-out reference data (95% CI {sensitivity[1]:.0%}–"
                      f"{sensitivity[2]:.0%})")
    elif decision.calibration is not None:
        confidence = "Calibrated bands"
    else:
        confidence = "Not available (no matching calibration)"
    return RadiologyFindings(
        anatomical_structures=["Chest X-ray"], abnormalities=[abnormality],
        impression=decision.outcome, recommendations=recommendations,
        confidence_level=confidence, model_text_answer=text_answer(raw_response),
    )


def parse_ai_response(raw_response: str) -> RadiologyFindings:
    """Classify the generated text alone, without a calibrated decision score.

    Only exact recognized findings are classified; all other responses are indeterminate.
    """
    answer = text_answer(raw_response)
    if answer == "No pneumothorax detected":
        impression = "No Pneumothorax Detected"
        abnormalities = ["Model output: no pneumothorax detected."]
    elif answer == "Pneumothorax detected":
        impression = "Pneumothorax Detected"
        abnormalities = ["Model output: pneumothorax detected."]
    else:
        impression = "Indeterminate"
        abnormalities = ["The model response could not be classified reliably."]
    return RadiologyFindings(
        anatomical_structures=["Chest X-ray"], abnormalities=abnormalities,
        impression=impression, recommendations=["Research use only; expert review required."],
        confidence_level="Not available (uncalibrated)", model_text_answer=answer,
    )
