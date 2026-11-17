import json
import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generate import NEXT_QUESTION, read_questions
from src.grade import grade
from src.paper_eval import (
    MODEL_HASH, PAPER, PROMPT, PROTOCOL, SAMPLING, SEED,
    _adapter_metadata, _generate, _limits, check_smoke, sample_rounds, score_rounds,
)


class PaperEvalTests(unittest.TestCase):
    def test_vllm_adapter_generation_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            adapter = root / "adapter"
            adapter.mkdir()
            config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
                      "bias": "none", "target_modules": ["q_proj", "v_proj"]}
            (adapter / "adapter_config.json").write_text(json.dumps(config))
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"fixture-adapter")
            calls = []

            class FakeRequest:
                def __init__(self, name, number, path):
                    self.name, self.number, self.path = name, number, path

            class FakeLLM:
                def __init__(self, **kwargs):
                    calls.append(("init", kwargs))
                def get_tokenizer(self):
                    return types.SimpleNamespace(encode=lambda *a, **kw: [1, 2])
                def generate(self, inputs, params, **kwargs):
                    calls.append(("generate", kwargs))
                    candidate = types.SimpleNamespace(text=r"\boxed{3}", token_ids=[1],
                                                      finish_reason="stop", stop_reason=None)
                    return [types.SimpleNamespace(outputs=[candidate]) for _ in inputs]

            modules = {
                "vllm": types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs),
                "vllm.lora": types.ModuleType("vllm.lora"),
                "vllm.lora.request": types.SimpleNamespace(LoRARequest=FakeRequest),
            }
            questions = [{"id": "a", "problem": "1+2?", "answer": "3"}]
            meta = {"rows": 1, "sha256": "fixture"}
            args = argparse.Namespace(model=model, adapter=adapter, suite="smoke", output_dir=root / "eval")
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}), \
                    patch("src.paper_eval._model_sha", return_value=MODEL_HASH), \
                    patch("src.paper_eval.version", return_value="fixture"):
                _generate(args, questions, meta, 2)
                manifest = json.loads((args.output_dir / "run.json").read_text())
                self.assertEqual(manifest["adapter"], _adapter_metadata(adapter))
                self.assertTrue(calls[0][1]["enable_lora"])
                self.assertEqual(calls[0][1]["max_lora_rank"], 8)
                for _, kwargs in calls[1:]:
                    self.assertEqual(kwargs["lora_request"].path, str(adapter.resolve()))
                self.assertEqual(check_smoke(args.output_dir, questions, meta, grade)["smoke"], "passed")
                count = len(calls)
                _generate(args, questions, meta, 2)
                self.assertEqual(len(calls), count)
                weights.write_bytes(b"different-adapter")
                with self.assertRaisesRegex(ValueError, "run.json differs"):
                    _generate(args, questions, meta, 2)

                args.adapter = None
                args.output_dir = root / "base-eval"
                calls.clear()
                _generate(args, questions, meta, 2)
                self.assertNotIn("enable_lora", calls[0][1])
                self.assertNotIn("adapter", json.loads((args.output_dir / "run.json").read_text()))
                for _, kwargs in calls[1:]:
                    self.assertNotIn("lora_request", kwargs)

    def test_adapter_config_validation_rejects_wrong_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
            with self.assertRaisesRegex(ValueError, "not the expected"):
                _adapter_metadata(adapter)

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
