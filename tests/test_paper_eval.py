import json
import tempfile
import unittest
from pathlib import Path

from src.generate import NEXT_QUESTION, read_questions
from src.grade import grade
from src.paper_eval import (
    MODEL_HASH, PAPER, PROMPT, PROTOCOL, SAMPLING, SEED,
    _limits, check_smoke, sample_rounds, score_rounds,
)


class PaperEvalTests(unittest.TestCase):
    def _write_smoke(self, directory, answers):
        questions = [
            {"id": "a", "problem": "One plus two?", "answer": "3"},
            {"id": "b", "problem": "Two plus three?", "answer": "5"},
            {"id": "c", "problem": "Three plus four?", "answer": "7"},
        ]
        path = directory / "questions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8")
        questions, meta = read_questions(path)
        manifest = {
            "protocol": PROTOCOL, "paper": PAPER, "suite": "smoke", "rounds": 2,
            "model_sha256": MODEL_HASH, "prompt_template": PROMPT, "sampling": SAMPLING,
            "seed_start": SEED, "questions": meta, **_limits("smoke"),
        }
        (directory / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        results = iter(answers)
        sample_rounds(directory, questions, 2, lambda prompts, seeds: (
            {"completion": next(results)} for _ in prompts
        ))
        return questions, meta

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
            self.assertTrue(all(s.startswith("Question: ") and s.endswith("\nSolution:") for s in prompts))
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

    def test_smoke_accepts_coverage_across_both_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            questions, meta = self._write_smoke(directory, [
                r"\boxed{3}", r"\boxed{0}", r"\boxed{7}",
                r"\boxed{0}", r"\boxed{5}", r"\boxed{0}",
            ])
            result = check_smoke(directory, questions, meta, grade)
        self.assertEqual(result, {"smoke": "passed", "covered_questions": 3, "questions": 3, "rounds": 2})

    def test_smoke_does_not_trust_metrics_or_allow_uncovered_questions(self):
        for answers, missing in (
            (["3", "5", "7", "3", "5", "7"], "a, b, c"),
            ([r"\boxed{3}", "5", r"\boxed{7}"] * 2, "b"),
        ):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                questions, meta = self._write_smoke(directory, answers)
                (directory / "metrics.json").write_text('{"mean_accuracy": 1, "smoke": "passed"}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"smoke failed:.* for {missing}$"):
                    check_smoke(directory, questions, meta, grade)

    def test_smoke_checks_manifest_and_attempt_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            questions, meta = self._write_smoke(directory, [r"\boxed{3}", r"\boxed{5}", r"\boxed{7}"] * 2)
            with self.assertRaisesRegex(ValueError, "questions do not match"):
                check_smoke(directory, questions, {**meta, "sha256": "invalid"}, grade)
            manifest_path = directory / "run.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for override in (
                {"suite": "math500"}, {"prompt_template": "other"},
                {"protocol": "idft-table8-base-reference-v1"}, {"max_new_tokens_cap": 4096},
            ):
                with self.subTest(override=override):
                    manifest_path.write_text(json.dumps({**manifest, **override}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "input does not match"):
                        check_smoke(directory, questions, meta, grade)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            path = directory / "round-02.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["id"] = "another-question"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched attempts"):
                check_smoke(directory, questions, meta, grade)

    def test_budget_profiles_and_stop_before_next_problem(self):
        self.assertEqual(_limits("smoke")["max_new_tokens_cap"], 384)
        self.assertEqual(_limits("math500")["max_new_tokens_cap"], 4096)
        self.assertIn(NEXT_QUESTION, SAMPLING["stop"])
