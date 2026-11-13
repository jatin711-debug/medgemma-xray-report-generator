"""Summarise calibration scores and choose decision thresholds.

    python -m tools.calibration_analysis reports/calibration-<UTC>/scores.csv [--write]

Reports AUC with a patient-level bootstrap interval for every score column,
then picks two thresholds for one score:

* review threshold: the highest score that still flags at least
  TARGET_SENSITIVITY of positives (a must-not-miss finding), and
* detected threshold: the lowest score whose specificity is at least
  TARGET_SPECIFICITY.

Operating characteristics are estimated with patient-grouped cross-validation
(thresholds chosen on the other folds), so they are not fitted to the images
they are measured on. --write stores the thresholds fitted on all images in
src/core/calibration.json.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET_SENSITIVITY = 0.90
TARGET_SPECIFICITY = 0.95
FOLDS = 5
BOOTSTRAP = 2000
SEED = 20261113


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = labels.sum()
    negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def review_threshold(scores, labels, target=TARGET_SENSITIVITY) -> float:
    """Highest threshold t with sensitivity(score >= t) >= target."""
    positives = np.sort(scores[labels == 1])[::-1]
    return float(positives[int(np.ceil(target * len(positives))) - 1])


def detected_threshold(scores, labels, target=TARGET_SPECIFICITY) -> float:
    """Lowest threshold t with specificity(score >= t) >= target (strictly above that negative)."""
    negatives = np.sort(scores[labels == 0])
    allowed_false_positives = int(np.floor((1 - target) * len(negatives)))
    return float(np.nextafter(negatives[len(negatives) - allowed_false_positives - 1], np.inf))


def rates(scores, labels, threshold) -> tuple[float, float]:
    flagged = scores >= threshold
    return float(flagged[labels == 1].mean()), float((~flagged[labels == 0]).mean())


def patient_bootstrap(patients, statistic, rng) -> tuple[float, float]:
    unique = np.unique(patients)
    index_of = {p: np.flatnonzero(patients == p) for p in unique}
    values = []
    for _ in range(BOOTSTRAP):
        sample = np.concatenate([index_of[p] for p in rng.choice(unique, len(unique))])
        value = statistic(sample)
        if value is not None:
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def cross_validated(scores, labels, patients, rng) -> dict:
    """Held-out sensitivity/specificity for both thresholds, folds grouped by patient."""
    unique = np.unique(patients)
    fold_of_patient = dict(zip(rng.permutation(unique), np.arange(len(unique)) % FOLDS))
    folds = np.array([fold_of_patient[p] for p in patients])
    flagged_review = np.zeros(len(scores), bool)
    flagged_detected = np.zeros(len(scores), bool)
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        flagged_review[test] = scores[test] >= review_threshold(scores[train], labels[train])
        flagged_detected[test] = scores[test] >= detected_threshold(scores[train], labels[train])

    def summary(flagged):
        def sens(idx):
            pos = labels[idx] == 1
            return float(flagged[idx][pos].mean()) if pos.any() else None

        def spec(idx):
            neg = labels[idx] == 0
            return float((~flagged[idx][neg]).mean()) if neg.any() else None

        everything = np.arange(len(scores))
        return {"sensitivity": [sens(everything), *patient_bootstrap(patients, sens, rng)],
                "specificity": [spec(everything), *patient_bootstrap(patients, spec, rng)],
                "flagged_share": float(flagged.mean())}

    # As in the app, the suspected band also requires the review threshold.
    in_band = {"detected": flagged_detected & flagged_review,
               "review": flagged_review & ~flagged_detected,
               "negative": ~flagged_review}

    def band_summary(mask):
        def positive_share(idx):
            selected = mask[idx]
            return float(labels[idx][selected].mean()) if selected.any() else None

        everything = np.arange(len(scores))
        return {"positive_share": [positive_share(everything),
                                   *patient_bootstrap(patients, positive_share, rng)],
                "image_share": float(mask.mean())}

    return {"review": summary(flagged_review), "detected": summary(flagged_detected),
            "bands": {name: band_summary(mask) for name, mask in in_band.items()}}


def rounded(value):
    """Round every float in a nested structure to 4 places for storage."""
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return round(value, 4) if isinstance(value, float) else value


def load(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = [k for k in rows[0] if k not in {"image", "patient", "view", "split", "label"}]
    return rows, columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--score", default="medgemma_app")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows, columns = load(args.scores)
    labels = np.array([int(r["label"]) for r in rows])
    patients = np.array([r["patient"] for r in rows])
    views = np.array([r["view"] for r in rows])
    rng = np.random.default_rng(SEED)
    print(f"{len(rows)} images, {labels.sum()} positive, {len(np.unique(patients))} patients\n")
    print(f"{'score':30} {'AUC':>6}  95% CI         AUC-PA  AUC-AP")
    for column in columns:
        s = np.array([float(r[column]) for r in rows])
        lo, hi = patient_bootstrap(
            patients, lambda i: auc(s[i], labels[i]) if 0 < labels[i].sum() < len(i) else None, rng)
        by_view = [auc(s[views == v], labels[views == v]) for v in ("PA", "AP")]
        print(f"{column:30} {auc(s, labels):6.3f}  {lo:.3f}–{hi:.3f}    {by_view[0]:.3f}   {by_view[1]:.3f}")

    scores = np.array([float(r[args.score]) for r in rows])
    zero = rates(scores, labels, 0.0 if args.score.startswith("medgemma") else 0.5)
    print(f"\n{args.score} at the current cutoff: sensitivity {zero[0]:.3f}, specificity {zero[1]:.3f}")
    validated = cross_validated(scores, labels, patients, rng)
    review, detected = review_threshold(scores, labels), detected_threshold(scores, labels)
    detected = max(detected, review)
    print(f"Thresholds on all data: review >= {review:.3f}, detected >= {detected:.3f}")
    for band in ("review", "detected"):
        result = validated[band]
        sens, spec = result["sensitivity"], result["specificity"]
        print(f"  {band:9} held-out sensitivity {sens[0]:.3f} ({sens[1]:.3f}–{sens[2]:.3f}), "
              f"specificity {spec[0]:.3f} ({spec[1]:.3f}–{spec[2]:.3f}), "
              f"flags {result['flagged_share']:.1%} of images")
    for band, result in validated["bands"].items():
        share = result["positive_share"]
        print(f"  {band:9} band holds {result['image_share']:.1%} of images; "
              f"{share[0]:.1%} ({share[1]:.1%}–{share[2]:.1%}) of them are positive")

    if args.write:
        from src.core.prompt import prompt_fingerprint

        meta = json.loads((args.scores.parent / "meta.json").read_text(encoding="utf-8"))
        if meta.get("prompt_fingerprint") != prompt_fingerprint():
            raise SystemExit("These scores were produced with a different prompt; re-run the scoring.")
        s_auc = auc(scores, labels)
        calibration = {
            "score": "logit('Pneumothorax detected') - logit('No pneumothorax detected')",
            "model_id": meta["model_id"], "precision": meta["precision"],
            "prompt_fingerprint": meta.get("prompt_fingerprint"),
            "review_threshold": round(review, 4), "detected_threshold": round(detected, 4),
            "targets": {"review_sensitivity": TARGET_SENSITIVITY,
                        "detected_specificity": TARGET_SPECIFICITY},
            "reference": {"labels": meta["labels"], "images": meta["images"],
                          "n_images": len(rows), "n_positive": int(labels.sum()),
                          "n_patients": int(len(np.unique(patients)))},
            "auc": [round(s_auc, 4), *[round(v, 4) for v in patient_bootstrap(
                patients, lambda i: auc(scores[i], labels[i]) if 0 < labels[i].sum() < len(i) else None,
                rng)]],
            "prevalence": round(float(labels.mean()), 4),
            "cross_validated": rounded(validated),
            "created": datetime.now(timezone.utc).date().isoformat(),
            "app_version": meta["app_version"],
        }
        target = ROOT / "src/core/calibration.json"
        key = ("model_id", "precision", "prompt_fingerprint")
        entries = json.loads(target.read_text(encoding="utf-8"))["entries"] if target.exists() else []
        entries = [e for e in entries if any(e[k] != calibration[k] for k in key)] + [calibration]
        target.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {target} ({calibration['precision']})")


if __name__ == "__main__":
    main()
