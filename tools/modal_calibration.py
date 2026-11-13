"""Score a radiologist-labelled pneumothorax set with the app's own code path.

    modal run -m tools.modal_calibration

Reference standard: Google's radiologist-adjudicated labels for 4,376 NIH
ChestX-ray14 images (Majkowska et al., Radiology 2020), as bundled with
torchxrayvision. Images come from a pinned Hugging Face mirror of NIH
ChestX-ray14 and are cached in a Modal Volume. Only scores and labels come back;
images are never written to the repository.

Writes results/<name>/{scores.csv,meta.json} to the Modal Volume; fetch them into reports/
for tools/calibration_analysis.py.
"""
import csv
import json
import subprocess
from datetime import datetime, timezone

import modal

from tools.modal_report_check import ROOT, hf_secret

app = modal.App("medgemma-calibration")
NIH_MIRROR = "iraqigold/nih-chest-xray-14-flat"
NIH_REVISION = "98fdac982847ed878cb68065d9e9f66998a79689"
XRV_VERSION = "1.5.7"
# Weights trained without NIH data are independent of this set; the NIH-trained ones
# ('all') may have seen these images and are reported for reference only.
XRV_WEIGHTS = ["densenet121-res224-chex", "densenet121-res224-mimic_ch",
               "densenet121-res224-mimic_nb", "densenet121-res224-pc", "densenet121-res224-all"]
YES_NO = "Is there a pneumothorax on this chest X-ray? Answer Yes or No."

volume = modal.Volume.from_name("nih-cxr14-google-subset", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.lock"),
                                   extra_options="--require-hashes --only-binary=:all:")
    .pip_install("torchvision==0.23.0", "scikit-image==0.25.2", "pandas==2.3.3", "imageio==2.37.0",
                 "tqdm==4.67.1")
    .pip_install(f"torchxrayvision=={XRV_VERSION}", extra_options="--no-deps")
    .run_commands("python -c \"import torchxrayvision as xrv; "
                  + "; ".join(f"xrv.models.get_model('{w}')" for w in XRV_WEIGHTS) + "\"")
    .add_local_dir(ROOT / "src", "/app/src")
)


@app.function(image=image, volumes={"/data": volume}, secrets=[hf_secret()], cpu=4, timeout=3600)
def prepare() -> list[dict]:
    """Return the label rows and make sure every labelled image is in the volume."""
    import gzip
    import io
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import requests
    import torchxrayvision
    from huggingface_hub import HfApi

    label_path = Path(torchxrayvision.__file__).parent / "data/google2019_nih-chest-xray-labels.csv.gz"
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(label_path.read_bytes()).decode())))
    target = Path("/data/images")
    target.mkdir(parents=True, exist_ok=True)
    missing = {r["Image Index"] for r in rows if not (target / r["Image Index"]).exists()}
    if missing:
        api = HfApi()
        paths = {}
        for folder in (f"images_{i:03d}/images" for i in range(1, 13)):
            for entry in api.list_repo_tree(NIH_MIRROR, path_in_repo=folder, repo_type="dataset",
                                            revision=NIH_REVISION):
                name = entry.path.rsplit("/", 1)[1]
                if name in missing:
                    paths[name] = entry.path
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

        def download(name: str) -> None:
            url = f"https://huggingface.co/datasets/{NIH_MIRROR}/resolve/{NIH_REVISION}/{paths[name]}"
            for attempt in range(8):
                response = session.get(url, timeout=120)
                if response.status_code == 200 and response.content[:8] == b"\x89PNG\r\n\x1a\n":
                    (target / name).write_bytes(response.content)
                    return
                time.sleep(min(60, 2 ** attempt))
            raise RuntimeError(f"Could not download {name}: HTTP {response.status_code}")

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(download, sorted(paths)))
        volume.commit()
        print(f"Downloaded {len(paths)} images", flush=True)
    return rows


@app.function(image=image, volumes={"/data": volume}, secrets=[hf_secret()], gpu="A10",
              memory=16384, cpu=4, timeout=3600, max_containers=4, retries=0)
def score(names: list[str], quantize: bool = False, float32: bool = False) -> tuple[str, list[dict]]:
    import os
    import sys
    from pathlib import Path

    os.environ["MEDGEMMA_QUANTIZE"] = "1" if quantize else "0"
    os.chdir("/app")
    sys.path.insert(0, "/app")
    import numpy as np
    import torch
    import torchvision
    import torchxrayvision as xrv

    from src.core.attribution import compute_pneumothorax_score
    from src.core.medical_analyzer import MedicalImageAnalyzer
    from src.services.image_input import decode_image

    analyzer = MedicalImageAnalyzer()
    analyzer._load_model()
    if float32:
        # Stands in for the app's CPU path, which runs MedGemma in float32.
        analyzer.model.to(torch.float32)
        analyzer.runtime["precision"] = MedicalImageAnalyzer._describe_runtime("cpu", False)["precision"]
    model, processor = analyzer.model, analyzer.processor
    # torchvision is installed here (for the classifiers); the app must still resize with PIL.
    processor_class = type(processor.image_processor).__name__
    if "Pil" not in processor_class:
        raise RuntimeError(f"Expected the PIL image processor, got {processor_class}")
    tok = processor.tokenizer
    yes_id = tok("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("No", add_special_tokens=False)["input_ids"][0]

    # The classifiers do not depend on MedGemma's precision; score them once, in the bf16 run.
    classifiers = {} if quantize or float32 else {
        w: xrv.models.get_model(w).to("cuda").eval() for w in XRV_WEIGHTS}
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])

    results = []
    for name in names:
        decoded, _ = decode_image((Path("/data/images") / name).read_bytes())
        row = {"image": name,
               "medgemma_app": compute_pneumothorax_score(model, processor,
                                                          MedicalImageAnalyzer._messages(decoded))}
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": decoded},
                                          {"type": "text", "text": YES_NO}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
        row["medgemma_yes_no"] = float(logits[yes_id] - logits[no_id])
        results.append(row)
        if not classifiers:
            continue

        gray = xrv.datasets.normalize(np.asarray(decoded.convert("L"), dtype=np.float32), 255)
        tensor = torch.from_numpy(transform(gray[None, ...])).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            for weight, classifier in classifiers.items():
                index = classifier.pathologies.index("Pneumothorax")
                row["xrv_" + weight.split("-")[-1]] = float(classifier(tensor)[0, index])
    print(f"Scored {len(results)} images", flush=True)
    return analyzer.runtime["precision"], results


@app.function(image=image, volumes={"/data": volume}, cpu=1, timeout=7200, retries=0)
def calibrate(name: str, meta: dict, limit: int = 0, quantize: bool = False, float32: bool = False,
              negatives: int = 0) -> str:
    """Score the reference set and write /data/results/<name>/{scores.csv,meta.json}.

    Runs remotely so a dropped local connection (with `modal run --detach`) loses nothing.
    """
    import random
    from pathlib import Path

    rows = prepare.remote()
    if negatives:
        negative_rows = [r for r in rows if r["Pneumothorax"] != "YES"]
        keep = {r["Image Index"] for r in random.Random(0).sample(negative_rows, negatives)}
        rows = [r for r in rows if r["Pneumothorax"] == "YES" or r["Image Index"] in keep]
    if limit:
        rows = rows[:limit]
    names = [r["Image Index"] for r in rows]
    chunks = [names[i:i + 250] for i in range(0, len(names), 250)]
    precisions, scored = set(), {}
    for precision, chunk in score.map(chunks, kwargs={"quantize": quantize, "float32": float32}):
        precisions.add(precision)
        scored.update({r["image"]: r for r in chunk})
    (precision,) = precisions

    out_dir = Path("/data/results") / name
    out_dir.mkdir(parents=True, exist_ok=False)
    keep = ["Image Index", "Patient ID", "View Position", "Set Id"]
    score_keys = [k for k in next(iter(scored.values())) if k != "image"]
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "patient", "view", "split", "label"] + score_keys)
        for r in rows:
            s = scored[r["Image Index"]]
            writer.writerow([r[k] for k in keep] + [int(r["Pneumothorax"] == "YES")]
                            + [round(s[k], 4) for k in score_keys])
    meta = {**meta, "precision": precision, "n_images": len(rows),
            "n_positive": sum(r["Pneumothorax"] == "YES" for r in rows)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    volume.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(limit: int = 0, quantize: bool = False, float32: bool = False, negatives: int = 0):
    """--negatives N keeps every positive plus N negatives (a fixed sample), for precision checks.

    Start with `modal run --detach -m tools.modal_calibration ...`; results land in the
    Volume and are fetched with
    `modal volume get nih-cxr14-google-subset results/<name> reports/`.
    """
    from src.core.prompt import prompt_fingerprint

    def git(*args) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    version = git("rev-parse", "--short=12", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    tag = "nf4" if quantize else "fp32" if float32 else "bf16"
    name = f"calibration-{tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    meta = {"app_version": version, "model_id": "google/medgemma-1.5-4b-it",
            "prompt_fingerprint": prompt_fingerprint(),
            "labels": "Majkowska et al., Radiology 2020 (radiologist-adjudicated), via torchxrayvision "
                      + XRV_VERSION,
            "images": f"NIH ChestX-ray14, hf://datasets/{NIH_MIRROR}@{NIH_REVISION}"}
    print(f"Results will be written to volume path results/{name}", flush=True)
    print(calibrate.remote(name, meta, limit=limit, quantize=quantize, float32=float32,
                           negatives=negatives))
