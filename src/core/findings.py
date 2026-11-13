"""Interpret the requested finding format without importing the model runtime."""
import re

from src.api.schemas import RadiologyFindings


_SECTION_LABELS = {"finding": r"Finding", "reasoning": r"(?:Brief\s+)?Reasoning"}


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


def parse_ai_response(raw_response: str) -> RadiologyFindings:
    """
    Parse raw AI model output for pneumothorax detection.
    
    Only exact recognized findings are classified; all other responses are indeterminate.
    
    Args:
        raw_response: Raw text output from the MedGemma model

    Returns:
        RadiologyFindings: Structured medical findings for pneumothorax.
    """
    # Accept only the exact requested finding line; everything else is uncertain.
    matches = re.findall(r"(?im)^[ \t]*\*\*Finding\*\*[ \t]*:[ \t]*([^\r\n]*)", raw_response)
    value = matches[0].strip().strip("* ").lower().rstrip(".") if len(matches) == 1 else ""
    if value == "no pneumothorax detected":
        impression = "No Pneumothorax Detected"
        abnormalities = ["Model output: no pneumothorax detected."]
    elif value == "pneumothorax detected":
        impression = "Pneumothorax Detected"
        abnormalities = ["Model output: pneumothorax detected."]
    else:
        impression = "Indeterminate"
        abnormalities = ["The model response could not be classified reliably."]
    return RadiologyFindings(
        anatomical_structures=["Chest X-ray"], abnormalities=abnormalities,
        impression=impression, recommendations=["Research use only; expert review required."],
        confidence_level="Not available (uncalibrated)",
    )

