"""Evaluate documented public case panels without sending their labels to the model."""
from pathlib import Path
import modal
from tools.modal_report_check import image, hf_secret, ROOT

app = modal.App("medgemma-known-case-check")


@app.function(image=image, gpu="A10", memory=16384, cpu=2,
              secrets=[hf_secret()], max_containers=1, timeout=600,
              startup_timeout=600, retries=0,
              env={"MEDGEMMA_QUANTIZE": "0", "MEDGEMMA_ATTRIBUTION": "1"})
async def evaluate(samples: list[tuple[str, bytes]], version: str):
    import io
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.environ["APP_VERSION"] = version
    from fastapi import UploadFile
    from src.core.medical_analyzer import MedicalImageAnalyzer
    from src.reports.generator import UltimatePDFGenerator
    import torch
    analyzer = MedicalImageAnalyzer()
    results = []
    for name, data in samples:
        response, original, attribution = await analyzer.analyze_image_from_upload(
            UploadFile(filename=name + ".png", file=io.BytesIO(data)))
        pdf = UltimatePDFGenerator("reports").generate_pdf_report(response, original, attribution)
        results.append({"id": name, "pdf": Path(pdf).read_bytes(),
                        "analysis": response.model_dump(mode="json"),
                        "grid": attribution.grid.tolist() if attribution else None})
        print(name, response.findings.impression, response.attribution, flush=True)
    return results, round(torch.cuda.max_memory_allocated() / 1024**3, 2)


@app.local_entrypoint()
def main():
    import json
    import subprocess
    from datetime import datetime, timezone
    inputs = ROOT / "reports/known-case"
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    # Only opaque IDs and image bytes reach the remote inference function.
    samples = [(c["id"], (inputs / c["file"]).read_bytes()) for c in manifest["cases"]]
    version = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True).strip()
    call = evaluate.spawn(samples, version)
    try:
        results, memory = call.get(timeout=600)
    except BaseException:
        call.cancel()
        raise
    output = ROOT / "reports" / ("known-case-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    for case, result in zip(manifest["cases"], results):
        name = result["id"]
        (output / (name + ".pdf")).write_bytes(result.pop("pdf"))
        (output / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        actual = result["analysis"]["findings"]["impression"]
        comparisons.append({**case, "actual": actual, "matches": actual == case["expected"],
                            "raw_response": result["analysis"]["raw_response"],
                            "attribution": result["analysis"]["attribution"]})
    summary = {**manifest, "cases": comparisons, "peak_gpu_gib": memory}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(output))
    print(json.dumps(summary, indent=2))
