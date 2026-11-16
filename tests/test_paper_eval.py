import tempfile
import unittest
from pathlib import Path

from src.grade import grade
from src.paper_eval import sample_rounds, score_rounds


class PaperEvalTests(unittest.TestCase):
    def test_rounds_preserve_units_and_use_mean_not_best_of(self):
        questions = [
            {"id": "a", "problem": "First problem", "answer": "9231"},
            {"id": "b", "problem": "Second problem", "answer": "7812"},
        ]
        seed_log = []

        def sample(prompts, seeds):
            self.assertIn("First problem", prompts[0])
            self.assertIn("Second problem", prompts[1])
            self.assertNotIn("9231", "".join(prompts))
            self.assertNotIn("7812", "".join(prompts))
            seed_log.extend(seeds)
            values = ("9231", "0") if len(seed_log) == 2 else ("0", "0")
            for value in values:
                yield {"completion": "\\boxed{" + value + "}"}

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample_rounds(directory, questions, 1, sample)
            with self.assertRaises(FileNotFoundError):
                score_rounds(directory, questions, 2, grade)
            self.assertFalse((directory / "graded.jsonl").exists())
            sample_rounds(directory, questions, 2, sample)
            result = score_rounds(directory, questions, 2, grade)

        self.assertEqual(seed_log, [42, 43, 44, 45])
        self.assertEqual(result["round_accuracy"], [0.5, 0.0])
        self.assertEqual(result["mean_accuracy"], 0.25)
        self.assertEqual(result["counts"], {"correct": 1, "incorrect": 3})
