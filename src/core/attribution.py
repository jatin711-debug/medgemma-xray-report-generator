"""Gradient-based attribution of MedGemma's pneumothorax decision to image regions.

The prompt allows two definite answers. After the answer prefix, the model's
choice between them comes down to one next-token decision, so the target score
is the logit difference between the first tokens where the two answers differ.
Relevance is gradient x activation on the projected image tokens the language
model receives (a 16 x 16 grid for MedGemma). It is signed: positive tokens
raised "Pneumothorax detected" and negative tokens raised "No pneumothorax
detected" at that decision.

This explains one uncalibrated model score. It is not a segmentation, a lesion
localisation, or a validated explanation of the model's reasoning.
"""
import torch

from src.core.heatmap import AttributionMap, relevance_grid

ANSWER_PREFIX = "**Finding**: "
POSITIVE_ANSWER = "Pneumothorax detected"
NEGATIVE_ANSWER = "No pneumothorax detected"
METHOD = "Gradient × activation on image tokens"
TARGET = "logit('Pneumothorax detected') - logit('No pneumothorax detected')"


class AttributionUnavailable(RuntimeError):
    """The model or tokenizer does not support this attribution method."""


def find_image_projector(model) -> torch.nn.Module:
    """Locate the module whose output is the image tokens fed to the language model."""
    for path in ("model.multi_modal_projector", "multi_modal_projector"):
        module = model
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, torch.nn.Module):
            return module
    for name, module in model.named_modules():
        if name.endswith("multi_modal_projector"):
            return module
    raise AttributionUnavailable("The model has no multi-modal projector")


def decision_tokens(tokenizer) -> tuple[list[int], int, int]:
    """Return the shared answer tokens and the first diverging token of each answer."""
    positive = list(tokenizer(ANSWER_PREFIX + POSITIVE_ANSWER, add_special_tokens=False)["input_ids"])
    negative = list(tokenizer(ANSWER_PREFIX + NEGATIVE_ANSWER, add_special_tokens=False)["input_ids"])
    shared = 0
    while shared < min(len(positive), len(negative)) and positive[shared] == negative[shared]:
        shared += 1
    if shared == min(len(positive), len(negative)):
        raise AttributionUnavailable("The two answers do not diverge at a single token")
    return positive[:shared], positive[shared], negative[shared]


def _append_tokens(batch: dict, tokens: list[int]) -> dict:
    """Append forced answer tokens to every per-token input tensor."""
    if not tokens:
        return batch
    input_ids = batch["input_ids"]
    length = input_ids.shape[1]
    extra = torch.tensor([tokens], dtype=input_ids.dtype, device=input_ids.device)
    extended = dict(batch)
    for key, value in batch.items():
        if not (torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == length):
            continue
        if key == "input_ids":
            addition = extra
        elif key == "attention_mask":
            addition = torch.ones(extra.shape, dtype=value.dtype, device=value.device)
        else:  # e.g. token_type_ids: forced answer tokens are text, not image.
            addition = torch.zeros(extra.shape, dtype=value.dtype, device=value.device)
        extended[key] = torch.cat([value, addition], dim=1)
    return extended


def _decision_batch(model, processor, messages) -> tuple[dict, int, int]:
    """Encode the prompt, force the shared answer prefix, and return the two answer token IDs."""
    forced, positive_id, negative_id = decision_tokens(processor.tokenizer)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    batch = {key: value.to(model.device) if torch.is_tensor(value) else value
             for key, value in dict(inputs).items()}
    return _append_tokens(batch, forced), positive_id, negative_id


def compute_pneumothorax_score(model, processor, messages) -> float:
    """Return logit('Pneumothorax detected') - logit('No pneumothorax detected').

    One forward pass without gradients. This is the score the decision bands
    are calibrated on, so calibration and the app must both use this function.
    """
    batch, positive_id, negative_id = _decision_batch(model, processor, messages)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    return float(logits[positive_id] - logits[negative_id])


def compute_pneumothorax_attribution(model, processor, messages) -> AttributionMap:
    """Run one gradient pass and return the image-token relevance map.

    Only the image tokens are made differentiable, so no gradients are
    computed for the vision encoder or any model weights.
    """
    batch, positive_id, negative_id = _decision_batch(model, processor, messages)

    captured: dict[str, torch.Tensor] = {}

    def capture_image_tokens(module, args, output):
        if not torch.is_tensor(output):
            return None
        features = output.detach().requires_grad_(True)
        captured["features"] = features
        return features

    handle = find_image_projector(model).register_forward_hook(capture_image_tokens)
    try:
        with torch.enable_grad():
            outputs = model(**batch, use_cache=False, logits_to_keep=1)
            features = captured.get("features")
            if features is None:
                raise AttributionUnavailable("Image tokens were not produced")
            logits = outputs.logits[0, -1].float()
            score = logits[positive_id] - logits[negative_id]
            (gradient,) = torch.autograd.grad(score, features)
        relevance = (gradient.float() * features.detach().float()).sum(dim=-1)
        grid = relevance_grid(relevance[0].cpu().numpy())
        return AttributionMap(grid=grid, target_score=float(score.detach()),
                              method=METHOD, target=TARGET)
    finally:
        handle.remove()
