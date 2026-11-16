"""Re-run the published parts of the IDFT Base evaluation protocol."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

from data.validate import Jsonl
from .generate import read_questions


PAPER = "https://arxiv.org/html/2602.12222v2#S4.SS1"
# Reference only: Qwen2.5-Math evaluation/utils.py at a45202b, qwen25-math-cot.
PROMPT = (
    "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MATH500_HASH = "0baf4950c69ab971f1b45437b965927e5a2a8fda0d2c884c408c9b9c087fca84"
MODEL_HASH = "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"
CONTEXT = 4096
SEED = 42
SAMPLING = {
    "n": 1, "temperature": 0.3, "top_p": 1.0, "top_k": -1,
    "stop": ["</s>", "<|im_end|>", "<|endoftext|>"], "stop_token_ids": [151645, 151643],
}


def _questions(path, suite):
    rows, meta = read_questions(path)
    if suite == "math500":
        if len(rows) != 500 or meta["sha256"] != MATH500_HASH:
            raise ValueError("math500 requires the pinned 500-question file")
    elif len(rows) > 8:
        raise ValueError("smoke accepts at most eight development questions")
    return rows, meta, 16 if suite == "math500" else 2


def _write_jsonl(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".partial-", delete=False,
        ) as out:
            temporary = Path(out.name)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if path.exists():
            raise ValueError(f"will not overwrite {path}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _round_path(directory, index):
    return directory / f"round-{index + 1:02d}.jsonl"


def _seeds(size, index):
    return list(range(SEED + size * index, SEED + size * (index + 1)))


def _read_round(directory, questions, index):
    path = _round_path(directory, index)
    rows = [row for row, _ in Jsonl(path)]
    expected = [(q["id"], seed) for q, seed in zip(questions, _seeds(len(questions), index))]
    if [(r.get("id"), r.get("seed")) for r in rows] != expected or any(
        r.get("round") != index + 1 or not isinstance(r.get("completion"), str) for r in rows
    ):
        raise ValueError(f"{path}: missing, misordered, or mismatched attempts")
    return rows


def sample_rounds(directory, questions, rounds, sample):
    """The sampler receives prompts and seeds only; gold is attached at scoring."""
    prompts = [PROMPT.format(problem=row["problem"]) for row in questions]
    for index in range(rounds):
        output = _round_path(directory, index)
        if output.exists():
            _read_round(directory, questions, index)
            continue
        seeds = _seeds(len(questions), index)
        items = list(sample(prompts, seeds))
        if len(items) != len(questions) or any(not isinstance(r.get("completion"), str) for r in items):
            raise ValueError("sampler did not return one completion per prompt")
        _write_jsonl(output, (
            {**item, "id": row["id"], "round": index + 1, "seed": seed}
            for item, row, seed in zip(items, questions, seeds)
        ))
        print(f"Saved round {index + 1}/{rounds}: {output}", flush=True)


def score_rounds(directory, questions, rounds, compare):
    # Validate the full matrix before producing any metric.
    attempts = [_read_round(directory, questions, index) for index in range(rounds)]
    gold = {row["id"]: row["answer"] for row in questions}
    graded, rates, counts = [], [], Counter()
    for rows in attempts:
        current = []
        for row in rows:
            answer = gold[row["id"]]
            current.append({**row, "answer": answer, **compare(row["completion"], answer)})
        rates.append(sum(row["correct"] for row in current) / len(current))
        counts.update(row["status"] for row in current)
        graded.extend(current)
    _write_jsonl(directory / "graded.jsonl", graded)
    return {
        "questions": len(questions), "rounds": rounds, "attempts": len(graded),
        "mean_accuracy": counts["correct"] / len(graded), "round_accuracy": rates,
        "counts": dict(counts), "at_generation_limit": sum(r.get("finish_reason") == "length" for r in graded),
        "metric": "mean single-attempt accuracy",
    }


def _model_sha(model):
    with (model / "model.safetensors").open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != MODEL_HASH:
        raise ValueError("model weights do not match the pinned Qwen2.5-Math-1.5B Base")
    return digest


def _model_config_sha(model):
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")
    return {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in names if (model / name).is_file()
    }


def _generate(args, questions, meta, rounds):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("set CUDA_VISIBLE_DEVICES to a single GPU")
    model = args.model.resolve()
    weights = _model_sha(model)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ValueError(f"vLLM is not usable in this Python environment: {exc}") from exc
    manifest = {
        "protocol": "idft-table8-base-reference-v1", "paper": PAPER, "suite": args.suite,
        "rounds": rounds, "questions": meta, "model": str(model), "model_sha256": weights,
        "model_config_files": _model_config_sha(model),
        "vllm": version("vllm"), "torch": version("torch"), "transformers": version("transformers"), "gpu": gpu,
        "prompt_template": PROMPT, "sampling": SAMPLING, "seed_start": SEED,
        "seed_rule": "seed_start + zero_based_round * questions + zero_based_question",
        "max_model_len": CONTEXT, "max_tokens_rule": "max_model_len - prompt_token_count",
        "dtype": "bfloat16", "generation_config": "vllm", "gpu_memory_utilization": 0.8,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = args.output_dir / "run.json"
    if run.exists():
        if json.loads(run.read_text(encoding="utf-8")) != manifest:
            raise ValueError("run.json differs; use a new output directory")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("output directory is not empty and has no run.json")
        with run.open("x", encoding="utf-8") as out:
            out.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if all(_round_path(args.output_dir, i).exists() for i in range(rounds)):
        for index in range(rounds):
            _read_round(args.output_dir, questions, index)
        return
    engine = LLM(
        model=str(model), dtype="bfloat16", max_model_len=CONTEXT,
        tensor_parallel_size=1, generation_config="vllm", gpu_memory_utilization=0.8,
    )
    tokenizer = engine.get_tokenizer()

    def sample(prompts, seeds):
        inputs = [
            {"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}
            for prompt in prompts
        ]
        caps = [CONTEXT - len(item["prompt_token_ids"]) for item in inputs]
        if min(caps) < 1:
            raise ValueError("a prompt fills the model context; no silent truncation")
        params = [SamplingParams(**SAMPLING, max_tokens=cap, seed=seed) for cap, seed in zip(caps, seeds)]
        outputs = engine.generate(inputs, params)
        for output, cap in zip(outputs, caps, strict=True):
            if len(output.outputs) != 1:
                raise ValueError("expected one vLLM sample per prompt")
            candidate = output.outputs[0]
            yield {
                "completion": candidate.text, "generated_tokens": len(candidate.token_ids),
                "finish_reason": candidate.finish_reason, "max_new_tokens": cap,
            }

    sample_rounds(args.output_dir, questions, rounds, sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "score"))
    parser.add_argument("--suite", choices=("smoke", "math500"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        questions, meta, rounds = _questions(args.questions, args.suite)
        if args.action == "generate":
            if args.model is None:
                raise ValueError("--model is required for generation")
            _generate(args, questions, meta, rounds)
        else:
            from .paper_grade import SCORER, check_dependencies, compare

            check_dependencies()
            run = json.loads((args.output_dir / "run.json").read_text(encoding="utf-8"))
            if (run.get("protocol"), run.get("suite"), run.get("rounds"), run["questions"]["sha256"]) != (
                "idft-table8-base-reference-v1", args.suite, rounds, meta["sha256"],
            ):
                raise ValueError("scoring input does not match the generation manifest")
            summary = score_rounds(args.output_dir, questions, rounds, compare)
            summary["scorer"] = SCORER
            with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as out:
                out.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(summary))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
