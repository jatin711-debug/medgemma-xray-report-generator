"""Compare prompt formats and MedGemma versions on a small labelled public set.

    modal run -m tools.modal_prompt_probe

Only opaque IDs and image bytes go to the GPU; labels stay local. The set is
tiny and hand-picked, so this shows failure modes, not accuracy.
"""
import json
import subprocess
from datetime import datetime, timezone

import modal

from tools.modal_report_check import ROOT, hf_secret, image

app = modal.App("medgemma-prompt-probe")
MODELS = ["google/medgemma-1.5-4b-it", "google/medgemma-4b-it"]

TASK = ("Review this chest X-ray carefully for signs of pneumothorax, including loss of lung markings, "
        "pleural line visibility, and increased radiolucency.\n\n"
        "Respond strictly in the following format:\n\n"
        "**Finding**: [Pneumothorax detected | No pneumothorax detected | Indeterminate]\n"
        "**Brief Reasoning**: [One-sentence justification]")
SYSTEM = ("You are a highly experienced radiologist. You analyze chest X-rays for critical findings, "
          "especially pneumothorax.")
REASON = ("Examine each hemithorax of this chest X-ray in turn. For each side, state whether you see a "
          "visceral pleural line, absent lung markings peripheral to it, a deep sulcus, or mediastinal "
          "shift. Then give your conclusion on the last line exactly as 'Final answer: Yes' if a "
          "pneumothorax is present or 'Final answer: No' if not.")


def variants(img):
    """name -> (messages, forced prefix, positive answer, negative answer, max_new_tokens)."""
    sys_msg = {"role": "system", "content": [{"type": "text", "text": SYSTEM}]}
    return {
        "app_prompt": ([sys_msg, {"role": "user", "content": [{"type": "text", "text": TASK},
                                                             {"type": "image", "image": img}]}],
                       "**Finding**: ", "Pneumothorax detected", "No pneumothorax detected", 80),
        "app_prompt_image_first": ([sys_msg, {"role": "user", "content": [{"type": "image", "image": img},
                                                                         {"type": "text", "text": TASK}]}],
                                   "**Finding**: ", "Pneumothorax detected", "No pneumothorax detected", 80),
        "yes_no": ([{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Is there a pneumothorax on this chest X-ray? Answer Yes or No."}]}],
                   "", "Yes", "No", 8),
        "reason_then_answer": ([{"role": "user", "content": [{"type": "image", "image": img},
                                                            {"type": "text", "text": REASON}]}],
                               None, None, None, 400),
        "free_report": ([{"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                         {"role": "user", "content": [{"type": "text", "text": "Describe this X-ray"},
                                                      {"type": "image", "image": img}]}],
                        None, None, None, 300),
    }


@app.function(image=image, gpu="A10", memory=16384, cpu=2, secrets=[hf_secret()],
              max_containers=2, timeout=1800, startup_timeout=900, retries=0)
def probe(model_id: str, samples: list[tuple[str, bytes]]):
    import io

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.requires_grad_(False)
    tok = processor.tokenizer

    def encode(messages):
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                             return_dict=True, return_tensors="pt").to(model.device)

    def decision_score(messages, prefix, pos, neg):
        p = tok(prefix + pos, add_special_tokens=False)["input_ids"]
        n = tok(prefix + neg, add_special_tokens=False)["input_ids"]
        k = 0
        while p[k] == n[k]:
            k += 1
        batch = dict(encode(messages))
        if k:
            extra = torch.tensor([p[:k]], device=model.device)
            length = batch["input_ids"].shape[1]
            for key, value in list(batch.items()):
                if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == length:
                    fill = extra if key == "input_ids" else (
                        torch.ones_like(extra) if key == "attention_mask" else torch.zeros_like(extra))
                    batch[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
        with torch.inference_mode():
            logits = model(**batch, logits_to_keep=1).logits[0, -1].float()
        return round(float(logits[p[k]] - logits[n[k]]), 3)

    def generate(messages, max_new_tokens):
        inputs = encode(messages)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    results = []
    for sample_id, data in samples:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        row = {"id": sample_id}
        for name, (messages, prefix, pos, neg, max_new) in variants(img).items():
            entry = {"text": generate(messages, max_new)}
            if prefix is not None:
                entry["score"] = decision_score(messages, prefix, pos, neg)
            row[name] = entry
        results.append(row)
        print(model_id, sample_id, {k: v.get("score") for k, v in row.items() if k != "id"}, flush=True)
    return {"model_id": model_id, "model_revision": getattr(model.config, "_commit_hash", None),
            "results": results}


def cases():
    """Local labels; nothing here except IDs and bytes is sent to the GPU."""
    known = ROOT / "reports/known-case"
    manifest = json.loads((known / "manifest.json").read_text(encoding="utf-8"))
    rows = [{"id": c["id"], "path": known / c["file"], "expected": c["expected"].startswith("Pneumothorax"),
             "kind": "subtle positive (journal figure, arrows)" if c["expected"].startswith("Pneumothorax")
             else "negative (same patient, resolved)", "source": manifest["source"]} for c in manifest["cases"]]
    commons = known / "commons"
    sources = json.loads((commons / "sources.json").read_text(encoding="utf-8"))
    labels = {
        "Pneu.jpg": (True, "large positive, right"),
        "09-01-Pneumothorax.png": (True, "large positive, right"),
        "05-Spontanpneumothorax.jpg": (True, "moderate positive, left"),
        "Expiration-left-side-pneumo.jpg": (True, "moderate positive, left (arrow, expiration film)"),
        "Chest Xray PA 3-8-2010.png": (False, "normal PA"),
        "Weichteilfalte in der Bettlunge 80W - CR ap - 001.jpg": (False, "negative: skin-fold mimic, supine AP"),
        "Weichteilfalte in der Bettlunge 85M - CR ap - 001.jpg": (False, "negative: skin-fold mimic, supine AP"),
    }
    for index, (name, (expected, kind)) in enumerate(labels.items(), start=4):
        rows.append({"id": f"case-{index:02d}", "path": commons / name, "expected": expected,
                     "kind": kind, "source": sources[name]["url"], "license": sources[name]["license"]})
    return rows


def final_answer(text: str):
    import re
    match = re.findall(r"final answer\W*\s*(yes|no)", text, flags=re.I)
    return None if not match else match[-1].lower() == "yes"


@app.local_entrypoint()
def main():
    rows = cases()
    samples = [(r["id"], r["path"].read_bytes()) for r in rows]
    calls = [probe.spawn(model_id, samples) for model_id in MODELS]
    outputs = []
    try:
        for call in calls:
            outputs.append(call.get(timeout=2400))
    except BaseException:
        for call in calls:
            call.cancel()
        raise
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports" / f"prompt-probe-{stamp}"
    out_dir.mkdir(parents=True)
    by_id = {r["id"]: r for r in rows}
    summary = {"app_version": subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                                      cwd=ROOT, text=True).strip(),
               "cases": [{k: (str(v) if k == "path" else v) for k, v in r.items()} for r in rows],
               "models": []}
    for output in outputs:
        table = {}
        for res in output["results"]:
            case = by_id[res["id"]]
            table[res["id"]] = {"expected": case["expected"], "kind": case["kind"],
                                **{name: {**entry, "predicted": (entry["score"] > 0 if "score" in entry
                                                                 else final_answer(entry["text"]))}
                                   for name, entry in res.items() if name != "id"}}
        summary["models"].append({"model_id": output["model_id"], "revision": output["model_revision"],
                                  "results": table})
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_dir)
    for model in summary["models"]:
        print("\n==", model["model_id"])
        print(f"{'case':8} {'truth':5} {'app':>7} {'img1st':>7} {'yes/no':>7} {'reason':>6}  kind")
        for cid, r in model["results"].items():
            fmt = lambda v: f"{v['score']:+7.2f}" if "score" in v else ""
            ra = r["reason_then_answer"]["predicted"]
            print(f"{cid:8} {'POS' if r['expected'] else 'neg':5} {fmt(r['app_prompt'])} "
                  f"{fmt(r['app_prompt_image_first'])} {fmt(r['yes_no'])} "
                  f"{'-' if ra is None else ('yes' if ra else 'no'):>6}  {r['kind']}")
