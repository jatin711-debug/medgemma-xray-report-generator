"""Interpret the requested finding format without importing the model runtime."""
import re

from src.api.schemas import RadiologyFindings


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

