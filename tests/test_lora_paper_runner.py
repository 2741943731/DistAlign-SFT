"""Offline integration checks for binding formal runs to their Base and smoke."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.paper_eval import PAPER, PROMPT, PROTOCOL, SAMPLING, SEED, _limits


class LoRAPaperRunnerTests(unittest.TestCase):
    def test_checks_adapter_smoke_and_base_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[1]
            for folder in ("scripts", "src", "data", "examples", "bin", "model", "adapter", "smoke", "base"):
                (root / folder).mkdir()
            for name in ("scripts/run_lora_paper_eval.sh", "src/__init__.py", "src/generate.py",
                         "src/grade.py", "data/__init__.py", "data/validate.py", "examples/evaluation.jsonl"):
                shutil.copyfile(source / name, root / name)

            def digest(data):
                return hashlib.sha256(data).hexdigest()

            model = root / "model"
            (model / "model.safetensors").write_bytes(b"fixture model")
            (model / "config.json").write_text('{}', encoding="utf-8")
            model_hash = digest((model / "model.safetensors").read_bytes())
            config_hash = digest((model / "config.json").read_bytes())
            math_file = root / "math.jsonl"
            math_file.write_text("".join(json.dumps({"id": str(i), "problem": str(i), "answer": "1"}) + "\n"
                                         for i in range(500)), encoding="utf-8")
            math_hash = digest(math_file.read_bytes())
            module = (source / "src/paper_eval.py").read_text(encoding="utf-8")
            from src.paper_eval import MATH500_HASH, MODEL_HASH
            module = module.replace(MATH500_HASH, math_hash).replace(MODEL_HASH, model_hash)
            (root / "src/paper_eval.py").write_text(module, encoding="utf-8")
            (root / "src/paper_grade.py").write_text(
                'from .grade import grade as compare\nSCORER="fixture-scorer"\n'
                'def check_dependencies(): pass\n', encoding="utf-8")

            adapter = root / "adapter"
            cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
                   "bias": "none", "target_modules": ["q_proj", "v_proj"]}
            (adapter / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            identity = {"path": str(adapter.resolve()), "weights_sha256": digest(b"adapter"),
                        "config_sha256": digest((adapter / "adapter_config.json").read_bytes()), "rank": 8, "alpha": 16}
            questions = [json.loads(line) for line in (root / "examples/evaluation.jsonl").read_text().splitlines()]
            smoke_hash = digest((root / "examples/evaluation.jsonl").read_bytes())
            common = {"protocol": PROTOCOL, "paper": PAPER, "model_sha256": model_hash,
                      "model_config_files": {"config.json": config_hash}, "prompt_template": PROMPT,
                      "sampling": SAMPLING, "seed_start": SEED, "seed_rule": "fixture seed rule",
                      "vllm": "test", "torch": "test", "transformers": "test", "gpu": "GPU-test",
                      "dtype": "bfloat16", "generation_config": "vllm", "gpu_memory_utilization": .8}
            smoke = {**common, "suite": "smoke", "rounds": 2, "questions": {"sha256": smoke_hash, "rows": 3},
                     "adapter": identity, **_limits("smoke")}
            base = {**common, "suite": "math500", "rounds": 16, "questions": {"sha256": math_hash, "rows": 500},
                    **_limits("math500")}
            (root / "smoke/run.json").write_text(json.dumps(smoke))
            (root / "base/run.json").write_text(json.dumps(base))
            (root / "base/metrics.json").write_text(json.dumps({"scorer": "fixture-scorer", "questions": 500,
                                                                 "rounds": 16, "attempts": 8000}))
            for round_index in range(2):
                rows = [{"id": q["id"], "round": round_index + 1, "seed": SEED + round_index*3 + i,
                         "completion": r"\boxed{" + q["answer"] + "}"} for i, q in enumerate(questions)]
                (root / f"smoke/round-{round_index + 1:02d}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))

            def executable(path, body):
                path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
                path.chmod(0o755)

            executable(root / "bin/ninja", "exit 0")
            executable(root / "bin/nvidia-smi", 'echo GPU-test')
            executable(root / "bin/vllm-python", '''echo '{"vllm":"test","torch":"test","transformers":"test"}' ''')
            env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                   "DISTALIGN_ADAPTER": str(adapter), "DISTALIGN_SMOKE_DIR": str(root / "smoke"),
                   "DISTALIGN_BASE_MANIFEST": str(root / "base/run.json"), "DISTALIGN_MODEL": str(model),
                   "DISTALIGN_QUESTIONS": str(math_file), "DISTALIGN_SCORE_PY": sys.executable,
                   "DISTALIGN_VLLM_PY": str(root / "bin/vllm-python")}
            cmd = ["bash", str(root / "scripts/run_lora_paper_eval.sh"), "check"]
            def check(expected, success):
                result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                self.assertIn(expected, result.stdout + result.stderr)

            check('"smoke": "passed"', True)
            (adapter / "adapter_model.safetensors").write_bytes(b"different")
            check("adapter 身份", False)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            altered = {**base, "vllm": "different"}
            (root / "base/run.json").write_text(json.dumps(altered))
            check("推理环境的版本不一致", False)
            (root / "base/run.json").write_text(json.dumps(base))
            smoke.pop("adapter")
            (root / "smoke/run.json").write_text(json.dumps(smoke))
            check("adapter 身份", False)
            smoke["adapter"] = identity
            (root / "smoke/run.json").write_text(json.dumps(smoke))
            for round_index in range(2):
                path = root / f"smoke/round-{round_index + 1:02d}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["completion"] = "The answer is 408."
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            check("smoke failed", False)


if __name__ == "__main__":
    unittest.main()
