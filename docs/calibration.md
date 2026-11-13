# How the outcome is decided, and how well it works

The app asks MedGemma 1.5 4B whether a chest X-ray shows a pneumothorax. This page explains how the reported outcome is derived from the model, why it no longer follows the model's written answer, and what the evaluation found. None of this makes the app a validated diagnostic tool.

## The problem with the original approach

The app used to report whatever the model wrote on its `**Finding**:` line. Because generation is greedy, that is equivalent to cutting the decision score at zero, where the score is

```
score = logit("Pneumothorax detected") − logit("No pneumothorax detected")
```

taken at the first answer token where the two answers differ. Zero has no special meaning for this score, and its position shifts with numeric precision. On the reference set described below, the cutoff at zero gave:

| Precision | Pneumothoraces caught | Other images cleared |
| --- | --- | --- |
| bfloat16 (`MEDGEMMA_QUANTIZE=0` on GPU) | 79.4% | 82.7% |
| 4-bit NF4 (the default on GPU) | 97.9% | 38.5% |
| float32 (CPU) | 78.6% | 83.0% |

So the default 4-bit setup flagged most normal images, and full precision missed about one pneumothorax in five. Separately, the one-sentence "reasoning" is written *after* the answer and is fitted to it (for example "no visible pleural line" on images where a pleural line is visible), so it is not evidence for the result.

## What the app does now

1. It still generates the answer and reasoning, and records them, but they no longer set the outcome. The report prints them under *Method → Generated text (not used for the outcome)* and flags a contradiction when the text disagrees with the outcome.
2. It computes the decision score in a separate forward pass (`compute_pneumothorax_score` in `src/core/attribution.py`).
3. It places the score in one of three bands, using thresholds from [`src/core/calibration.json`](../src/core/calibration.json):

| Outcome | Rule | Meaning |
| --- | --- | --- |
| **Pneumothorax suspected** | score ≥ suspected threshold | Set for 95% specificity |
| **Possible pneumothorax: review required** | review ≤ score < suspected | Pneumothorax not ruled out |
| **No pneumothorax flagged** | score < review threshold | Set so that 90% of pneumothoraces score above it |
| **Indeterminate** | no score, or no matching calibration | The score cannot be interpreted |

Each calibration entry is fitted for one **model ID**, one **numeric precision** and one **prompt** (identified by a SHA-256 fingerprint of the prompt text in `src/core/prompt.py`). If any of them changes, no entry matches and every outcome is *Indeterminate* until the calibration is re-run. A unit test fails if the shipped calibration does not match the current prompt.

Image preprocessing is pinned to the PIL backend (`IMAGE_PROCESSOR_BACKEND` in `src/core/medical_analyzer.py`). Transformers otherwise switches to a torchvision resizer whenever torchvision is installed, which moved scores by up to 1.4 in testing, enough to change bands.

## Reference data

- **Labels:** Google's radiologist-adjudicated labels for 4,376 NIH ChestX-ray14 images (Majkowska et al., *Radiology* 2020), as bundled with [torchxrayvision](https://github.com/mlmed/torchxrayvision) 1.5.7. Each label was adjudicated by a panel of board-certified radiologists. 238 images (5.4%) have a pneumothorax. 1,695 patients; 3,244 AP and 1,132 PA views.
- **Images:** the original 1024 × 1024 NIH PNGs, from a pinned Hugging Face mirror (`iraqigold/nih-chest-xray-14-flat` at commit `98fdac98`). They are not stored in this repository.
- **Validation:** thresholds were chosen with patient-grouped 5-fold cross-validation, so every reported sensitivity, specificity and band rate comes from images that were held out when their thresholds were fitted. 95% intervals are patient-level bootstrap intervals (2,000 resamples). The shipped thresholds are then fitted on all images.

## Results

AUC (95% CI) of the decision score: **0.885 (0.861–0.908)** in bfloat16, **0.889 (0.862–0.912)** in 4-bit and **0.885 (0.859–0.907)** in float32. Discrimination is better on PA views (0.92) than on AP views (0.85–0.86), which are often portable bedside films.

| Precision | Review threshold | Caught at review threshold | Cleared | Suspected threshold | Caught | Cleared |
| --- | --- | --- | --- | --- | --- | --- |
| bfloat16 | −3.5 | 89.5% (85.4–93.4) | 64.4% (61.1–67.4) | +6.75 | 46.6% (39.7–53.6) | 95.2% (93.4–96.5) |
| 4-bit NF4 | +4.625 | 89.9% (85.6–93.7) | 69.8% (66.7–72.8) | +9.5 | 44.5% (37.1–51.3) | 95.0% (92.6–96.6) |
| float32 | −3.145 | 89.1% (84.6–93.0) | 67.5% (64.4–70.7) | +6.565 | 46.6% (39.1–53.9) | 94.8% (92.8–96.3) |

How often images in each band had a pneumothorax:

| Band | bfloat16: share of images | Had a pneumothorax | 4-bit: share | Had one | float32: share | Had one |
| --- | --- | --- | --- | --- | --- | --- |
| Suspected | 7.1% | 35.7% (27.7–44.2) | 7.1% | 33.9% (24.9–44.2) | 7.4% | 34.2% (26.4–42.7) |
| Review | 31.4% | 7.4% (5.7–9.3) | 26.3% | 9.4% (7.4–11.6) | 28.1% | 8.2% (6.4–10.3) |
| No flag | 61.5% | 0.9% (0.6–1.3) | 66.5% | 0.8% (0.5–1.2) | 64.4% | 0.9% (0.6–1.3) |

The report prints the rate for the image's own band. These rates depend on how common pneumothorax is: they hold for a population like this one (5.4%), not for a screening or emergency setting with a different prevalence. The top band is called *suspected* rather than *detected* because about two-thirds of images in it did not have a pneumothorax.

### A dedicated classifier did not help

The [torchxrayvision](https://github.com/mlmed/torchxrayvision) DenseNet classifiers were scored on the same images as candidate second detectors:

| Model | AUC (95% CI) |
| --- | --- |
| MedGemma decision score (app prompt) | 0.885 (0.861–0.908) |
| MedGemma, plain "Yes or No?" prompt | 0.870 (0.836–0.900) |
| torchxrayvision `all` (trained partly on NIH, so may have seen these images) | 0.770 (0.739–0.802) |
| torchxrayvision `pc` | 0.761 (0.722–0.795) |
| torchxrayvision `mimic_nb` | 0.716 (0.676–0.753) |
| torchxrayvision `chex` | 0.698 (0.653–0.739) |
| torchxrayvision `mimic_ch` | 0.672 (0.631–0.713) |

All were clearly worse than MedGemma's own score, so none was added.

### Numeric precision changes individual results

4-bit quantization shifts the score by about +5.4 on average (correlation 0.86 with bfloat16), which is why each precision has its own thresholds. Even float32 versus bfloat16 moves individual scores by 0.5 on average and up to 7.3: aggregate performance is the same, but about 7% of images land in a different band. Float32, the precision used on CPU, therefore has its own fitted entry too. The model's answer for a borderline image is not stable across hardware set-ups.

## Known limitations

- **One data set.** All images come from one US hospital system (NIH Clinical Center), mostly inpatient AP films. Performance elsewhere is unknown.
- **Possible training overlap.** We could not verify that these images were excluded from MedGemma's training data, which would make the results optimistic.
- **Treated pneumothoraces.** Many positive NIH images are follow-up films with chest tubes in place. A model can score these highly by recognising the tube rather than the pneumothorax, so sensitivity for untreated pneumothorax may be lower.
- **Misses remain.** About 1 in 10 pneumothoraces fall below the review threshold. In a hand-checked set of public case images, a large right-sided pneumothorax scored −5.9 and was still not flagged.
- **Labels cover presence only**, not size, so performance on small pneumothoraces is not known separately.

## Re-running the calibration

Needed whenever the model, prompt or preprocessing changes. It requires a [Modal](https://modal.com) account, a Hugging Face login with MedGemma access, and about 40 minutes on 4 A10 GPUs per precision.

```shell
# Score all 4,376 images (first run also downloads them into a Modal Volume).
modal run --detach -m tools.modal_calibration                # bfloat16
modal run --detach -m tools.modal_calibration --quantize     # 4-bit NF4
modal run --detach -m tools.modal_calibration --float32      # float32 (the CPU path)

# Fetch a finished run, then review it and write its entry to src/core/calibration.json.
modal volume get nih-cxr14-google-subset results/<run-name> reports/
python -m tools.calibration_analysis reports/<run-name>/scores.csv
python -m tools.calibration_analysis reports/<run-name>/scores.csv --write
```

`--write` refuses scores produced with a different prompt, and replaces any existing entry for the same model, precision and prompt.
