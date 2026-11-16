import importlib.util
from types import SimpleNamespace
import unittest

from src.generate import PROMPT

if importlib.util.find_spec("torch") is not None:
    import torch

    from src.hf_train import (
        backward_token_mean, collate_examples, encode_example, learning_rate_at_step, optimizer_batches,
    )
    from src.losses import LOSS_TYPES, causal_lm_loss
else:
    torch = None


class CharacterTokenizer:
    """An offline tokenizer fixture with a distinct ID for every character."""

    eos_token_id = 2
    eos_token = "<eos>"

    def encode(self, text, *, add_special_tokens):
        prefix = [1] if add_special_tokens else []
        suffix = [self.eos_token_id] if text.endswith(self.eos_token) else []
        literal = text.removesuffix(self.eos_token)
        return prefix + [ord(char) + 10 for char in literal] + suffix


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HFTrainTests(unittest.TestCase):
    def test_examples_learn_the_response_without_learning_the_prompt(self):
        tokenizer = CharacterTokenizer()
        row = {"problem": "What is half of 8?", "solution": r"Divide by 2: \boxed{4}."}
        prompt = PROMPT.format(problem=row["problem"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode("\n" + row["solution"], add_special_tokens=False) + [tokenizer.eos_token_id]

        ids, labels = encode_example(tokenizer, row, max_length=256)
        self.assertEqual(ids, prompt_ids + answer_ids)
        self.assertEqual(labels, [-100] * len(prompt_ids) + answer_ids)

        self.assertEqual(encode_example(tokenizer, row, max_length=len(ids)), (ids, labels))
        self.assertIsNone(encode_example(tokenizer, row, max_length=len(ids) - 1))
        with_eos = {**row, "solution": row["solution"] + tokenizer.eos_token}
        self.assertEqual(encode_example(tokenizer, with_eos, max_length=len(ids)), (ids, labels))

    def test_padding_masks_only_pads_even_when_eos_and_pad_use_the_same_id(self):
        eos = CharacterTokenizer.eos_token_id
        encoded = [([30, 40, eos], [-100, 40, eos]), ([30, 50, 60, eos], [-100, 50, 60, eos])]
        batch = collate_examples(encoded, pad_token_id=eos)

        self.assertEqual(batch["input_ids"].tolist(), [[30, 40, eos, eos], [30, 50, 60, eos]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 40, eos, -100], [-100, 50, 60, eos]])
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_optimizer_batches_keep_shuffle_order_and_remainder(self):
        examples = list("abcdefg")
        order = [6, 2, 3, 0, 5, 1, 4]
        self.assertEqual(
            list(optimizer_batches(examples, order, batch_size=2, accumulation_steps=2)),
            [[["g", "c"], ["d", "a"]], [["f", "b"], ["e"]]],
        )

    def test_uneven_microbatches_match_one_token_mean_backward(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Embedding(8, 3), torch.nn.Linear(3, 8)).double()

            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(logits=self.net(input_ids))

        examples = [
            ([1, 2], [-100, 2]),
            ([1, 3, 2], [-100, -100, 2]),
            ([1, 4, 5, 6, 2], [-100, 4, 5, 6, 2]),
        ]
        (micro_batches,) = optimizer_batches(examples, [0, 1, 2], batch_size=2, accumulation_steps=2)
        model = TinyModel()
        for loss_type in LOSS_TYPES:
            with self.subTest(loss_type=loss_type):
                model.zero_grad()
                batch = collate_examples(examples, pad_token_id=0)
                labels = batch.pop("labels")
                together = causal_lm_loss(model(**batch, use_cache=False).logits, labels, loss_type=loss_type)
                together.backward()
                gradients = [p.grad.clone() for p in model.parameters()]

                model.zero_grad()
                stats = backward_token_mean(model, micro_batches, 0, "cpu", loss_type, step=1)
                for p, expected_gradient in zip(model.parameters(), gradients):
                    torch.testing.assert_close(p.grad, expected_gradient)
                self.assertAlmostEqual(stats["loss"], together.item())
                self.assertEqual(stats["supervised_tokens"], 6)
                self.assertEqual(stats["effective_batch_size"], 3)
                self.assertEqual(stats["micro_batches"], 2)
                self.assertEqual(stats["max_sequence_width"], 5)

    def test_learning_rate_uses_optimizer_steps(self):
        self.assertEqual([learning_rate_at_step(1, s, 3, 0, "constant") for s in range(3)], [1, 1, 1])
        self.assertEqual(learning_rate_at_step(1, 0, 1, 0, "cosine"), 1)
        rates = [learning_rate_at_step(1, s, 6, 2, "cosine") for s in range(6)]
        self.assertEqual(rates[:3], [0.5, 1, 1])
        self.assertAlmostEqual(rates[4], 0.5)
        self.assertGreater(rates[5], 0)
        self.assertGreater(rates[3], rates[4])
        self.assertGreater(rates[4], rates[5])
