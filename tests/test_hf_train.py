import importlib.util
import unittest

from src.generate import PROMPT

if importlib.util.find_spec("torch") is not None:
    import torch

    from src.hf_train import collate_examples, encode_example
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
