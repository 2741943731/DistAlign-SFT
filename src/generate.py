"""Generate saved completions from a Base model or a local LoRA adapter."""

import argparse
import json
from pathlib import Path
import re

from data.validate import Jsonl, text_field


QWEN_BASE = "Qwen/Qwen2.5-Math-1.5B"
QWEN_REVISION = "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
PROMPT = (
    "Question: {problem}\n"
    "Show your reasoning and put the final answer in \\boxed{{}}.\n"
    "Solution:"
)
NEXT_QUESTION = "\n\nQuestion:"


def read_questions(path: Path) -> tuple[list[dict], dict]:
    reader = Jsonl(path)
    questions = []
    seen_ids = set()
    for row, where in reader:
        example = {key: text_field(row, key, where) for key in ("id", "problem", "answer")}
        if example["id"] in seen_ids:
            raise ValueError(f"{where}: duplicate id {example['id']!r}")
        seen_ids.add(example["id"])
        questions.append(example)
    if not questions:
        raise ValueError("at least one question is required")
    return questions, reader.provenance()


def generate_rows(questions, generate_one):
    for row in questions:
        # The standard answer is attached only after generation.
        prompt = PROMPT.format(problem=row["problem"])
        raw_completion, token_count, hit_token_limit = generate_one(prompt)
        completion = raw_completion.split(NEXT_QUESTION, 1)[0].strip()
        yield {
            **row, "completion": completion, "raw_completion": raw_completion,
            "generated_tokens": token_count, "hit_token_limit": hit_token_limit,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="JSONL with id, problem and answer")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for predictions and run metadata")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--model", default=QWEN_BASE, help="A Hub model ID or a local checkpoint directory")
    parser.add_argument("--revision", help="Full commit SHA for a Hub model; Qwen Base has a fixed default")
    parser.add_argument("--adapter", type=Path, help="Local LoRA adapter to load on top of --model")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if min(args.limit, args.max_new_tokens) < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.adapter is not None:
        adapter_files = ("adapter_config.json", "adapter_model.safetensors")
        if not all((args.adapter / filename).is_file() for filename in adapter_files):
            parser.error(f"expected adapter_config.json and adapter_model.safetensors in {args.adapter}")
    local_model = Path(args.model).is_dir()
    if local_model and args.revision:
        parser.error("--revision applies only to Hub models")
    revision = args.revision or (QWEN_REVISION if args.model == QWEN_BASE else None)
    if not local_model and not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        parser.error("a Hub model requires a full 40-character commit SHA via --revision")
    model_name = str(Path(args.model).resolve()) if local_model else args.model
    try:
        questions, source = read_questions(args.questions)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        parser.error(f"generation requires PyTorch and Transformers: {exc}")
    if args.adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            parser.error(f"--adapter requires compatible PEFT: {exc}")

    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    device = args.device if args.device != "auto" else next(name for name, ok in available.items() if ok)
    if not available[device]:
        parser.error(f"device is unavailable: {device}")

    load_options = {"revision": revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device == "mps" else "auto", use_safetensors=True, **load_options,
    ).to(device)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter.resolve(), is_trainable=False, local_files_only=True)
    model.eval()

    def generate_one(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decoder-only models return prompt tokens followed by new tokens.
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        count = new_tokens.shape[0]
        return tokenizer.decode(new_tokens, skip_special_tokens=True), count, count == args.max_new_tokens

    rows = list(generate_rows(questions[:args.limit], generate_one))
    run = {
        "model": model_name, "revision": revision, "device": device, "dtype": str(next(model.parameters()).dtype),
        "adapter": str(args.adapter.resolve()) if args.adapter is not None else None,
        "torch": torch.__version__, "transformers": transformers.__version__, "questions": source,
        "limit": args.limit, "generated_rows": len(rows), "prompt_template": PROMPT,
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1, "trim_at": NEXT_QUESTION,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
