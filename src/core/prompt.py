"""The pneumothorax prompt, kept free of model imports so tools can fingerprint it.

Decision thresholds are calibrated for this exact wording. Any edit changes the
fingerprint, and the app then reports the calibration as not matching.
"""
import hashlib

SYSTEM_PROMPT = ("You are a highly experienced radiologist. You analyze chest X-rays for critical "
                 "findings, especially pneumothorax.")
USER_PROMPT = (
    "Review this chest X-ray carefully for signs of pneumothorax, including loss of lung markings, "
    "pleural line visibility, and increased radiolucency.\n\n"
    "Respond strictly in the following format:\n\n"
    "**Finding**: [Pneumothorax detected | No pneumothorax detected | Indeterminate]\n"
    "**Brief Reasoning**: [One-sentence justification]"
)


def build_messages(image) -> list[dict]:
    """Chat messages for one image; generation, scoring and attribution all use these."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT},
                                     {"type": "image", "image": image}]},
    ]


def prompt_fingerprint() -> str:
    """Short SHA-256 of the prompt text."""
    return hashlib.sha256(f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}".encode("utf-8")).hexdigest()[:16]
