"""Attribution checks against a tiny stand-in vision-language model (no download)."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.core.attribution import (AttributionUnavailable, compute_pneumothorax_attribution,
                                  decision_tokens, find_image_projector)

VOCAB = {"**Finding**:": 1, "Pneumothorax": 2, "detected": 3, "No": 4, "pneumothorax": 5}


class WordTokenizer:
    def __init__(self, vocab=VOCAB):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab[word] for word in text.split()]}


class FakeProcessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or WordTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        pixel_values = torch.zeros(1, 256, 4)
        pixel_values[0, 0] = 1.0  # Only the first image token carries signal.
        return {"input_ids": torch.tensor([[7, 8, 9]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "token_type_ids": torch.zeros(1, 3, dtype=torch.long), "pixel_values": pixel_values}


class FakeVisionLanguageModel(nn.Module):
    """Logits are a linear readout of the mean projected image token."""

    def __init__(self, favoured_token: int):
        super().__init__()
        self.model = nn.Module()
        self.model.multi_modal_projector = nn.Linear(4, 8, bias=False)
        self.head = nn.Linear(8, 10, bias=False)
        with torch.no_grad():
            self.model.multi_modal_projector.weight.fill_(1.0)
            self.head.weight.zero_()
            self.head.weight[favoured_token].fill_(1.0)
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, pixel_values, attention_mask=None, token_type_ids=None,
                use_cache=None, logits_to_keep=0):
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask,
                           "token_type_ids": token_type_ids, "use_cache": use_cache,
                           "logits_to_keep": logits_to_keep})
        features = self.model.multi_modal_projector(pixel_values)
        logits = self.head(features.mean(dim=1))[:, None, :]
        return SimpleNamespace(logits=logits)


class AttributionTests(unittest.TestCase):
    def make_model(self, favoured_token):
        model = FakeVisionLanguageModel(favoured_token)
        model.requires_grad_(False)  # Matches the analyzer's frozen inference model.
        return model

    def test_relevance_points_at_the_token_that_raised_the_pneumothorax_score(self):
        model = self.make_model(VOCAB["Pneumothorax"])
        result = compute_pneumothorax_attribution(model, FakeProcessor(), messages=[])
        self.assertEqual(result.grid.shape, (16, 16))
        self.assertEqual(result.grid[0, 0], 1.0)
        self.assertEqual(float(result.grid.sum()), 1.0)
        self.assertAlmostEqual(result.target_score, 8 * 4 / 256)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_evidence_against_pneumothorax_is_kept_as_negative_relevance(self):
        result = compute_pneumothorax_attribution(self.make_model(VOCAB["No"]), FakeProcessor(), [])
        self.assertEqual(result.grid[0, 0], -1.0)
        self.assertFalse(result.has_toward)
        self.assertTrue(result.has_against)
        self.assertLess(result.target_score, 0)

    def test_forced_answer_prefix_is_appended_to_every_token_input(self):
        model = self.make_model(VOCAB["Pneumothorax"])
        compute_pneumothorax_attribution(model, FakeProcessor(), [])
        call = model.calls[-1]
        self.assertEqual(call["input_ids"].tolist(), [[7, 8, 9, VOCAB["**Finding**:"]]])
        self.assertEqual(call["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(call["token_type_ids"].tolist(), [[0, 0, 0, 0]])
        self.assertEqual((call["use_cache"], call["logits_to_keep"]), (False, 1))

    def test_hook_is_removed_even_when_the_forward_pass_fails(self):
        model = self.make_model(VOCAB["Pneumothorax"])
        projector = find_image_projector(model)
        model.forward = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            compute_pneumothorax_attribution(model, FakeProcessor(), [])
        self.assertEqual(len(projector._forward_hooks), 0)

    def test_unsupported_models_and_tokenizers_are_reported(self):
        with self.assertRaises(AttributionUnavailable):
            find_image_projector(nn.Linear(2, 2))
        same = WordTokenizer({word: 1 for word in VOCAB})
        with self.assertRaises(AttributionUnavailable):
            decision_tokens(same)

    def test_decision_tokens_split_at_first_difference(self):
        self.assertEqual(decision_tokens(WordTokenizer()),
                         ([VOCAB["**Finding**:"]], VOCAB["Pneumothorax"], VOCAB["No"]))


if __name__ == "__main__":
    unittest.main()
