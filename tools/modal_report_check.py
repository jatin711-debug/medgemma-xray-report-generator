"""One-shot real-model report check: modal run tools/modal_report_check.py.

Uses the local Hugging Face login through an ephemeral Modal Secret. Only source
and font assets are uploaded. No API is deployed. Outputs stay in reports/.
"""
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("medgemma-report-check")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.lock"),
                                   extra_options="--require-hashes --only-binary=:all:")
    .add_local_dir(ROOT / "src", "/app/src")
    .add_local_dir(ROOT / "assets", "/app/assets")
)
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"


def download_example():
    import base64
    import requests
    # Google's notebook embeds the unmodified example input, with CC0 credit.
    response = requests.get("https://raw.githubusercontent.com/google-health/medgemma/main/notebooks/quick_start_with_hugging_face.ipynb", timeout=60)
    response.raise_for_status()
    for cell in response.json()["cells"]:
        if "display(IPImage(filename=image_filename" in "".join(cell.get("source", [])):
            for output in cell.get("outputs", []):
                encoded = output.get("data", {}).get("image/png")
                if encoded:
                    return base64.b64decode(encoded)
    raise RuntimeError("Google's notebook no longer contains the example input")


def hf_secret():
    if not modal.is_local():
        return modal.Secret.from_dict({})
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        raise RuntimeError("Sign in to Hugging Face with access to MedGemma first")
    return modal.Secret.from_dict({"HF_TOKEN": token})


@app.function(image=image, gpu="A10", memory=16384, cpu=2,
              secrets=[hf_secret()], max_containers=1, timeout=600,
              startup_timeout=600, retries=0,
              env={"MEDGEMMA_QUANTIZE": "0", "MEDGEMMA_ATTRIBUTION": "1"})
async def check(image_bytes: bytes, version: str):
    import io
    import os
    import sys
    import time
    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.environ["APP_VERSION"] = version
    from fastapi import UploadFile
    from src.core.medical_analyzer import MedicalImageAnalyzer
    from src.reports.generator import UltimatePDFGenerator
    import torch

    started = time.monotonic()
    analyzer = MedicalImageAnalyzer()
    response, original, attribution = await analyzer.analyze_image_from_upload(
        UploadFile(filename="public-example.png", file=io.BytesIO(image_bytes)))
    pdf = UltimatePDFGenerator("reports").generate_pdf_report(response, original, attribution)
    diagnostics = {"elapsed_seconds": round(time.monotonic() - started, 2),
                   "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                   "sample_url": SAMPLE_URL,
                   "sample_credit": "Stillwaterising, CC0, via Wikimedia Commons",
                   "model_revision": getattr(analyzer.model.config, "_commit_hash", None)}
    return Path(pdf).read_bytes(), response.model_dump(mode="json"), diagnostics


@app.local_entrypoint()
def main():
    import json
    from datetime import datetime, timezone
    print("Downloading the public example from Google's quickstart notebook...", flush=True)
    image_bytes = download_example()
    version = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True).strip()
    print("Submitting one A10 job (maximum 10 minutes including queue time)...", flush=True)
    call = check.spawn(image_bytes, version)
    try:
        pdf, response, diagnostics = call.get(timeout=600)
    except BaseException:
        call.cancel()
        raise
    output = ROOT / "reports" / ("modal-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.pdf").write_bytes(pdf)
    (output / "input.png").write_bytes(image_bytes)
    (output / "analysis.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "finding": response["findings"],
                      "attribution": response["attribution"], **diagnostics}, indent=2))
