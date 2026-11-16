"""Run a single-device LoRA pilot from a local Hugging Face model."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from data.validate import Jsonl, text_field
from .generate import PROMPT
from .losses import IDFT_CONFIG, LOSS_TYPES, causal_lm_loss


IGNORE_INDEX = -100
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS = ("q_proj", "v_proj")


def encode_example(tokenizer, row, max_length):
    """Match the inference prompt and supervise only the solution and EOS."""
    prompt_ids = tokenizer.encode(PROMPT.format(problem=row["problem"]), add_special_tokens=True)
    answer_ids = tokenizer.encode("\n" + row["solution"], add_special_tokens=False)
    if answer_ids[-1:] != [tokenizer.eos_token_id]:
        answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return prompt_ids + answer_ids, [IGNORE_INDEX] * len(prompt_ids) + answer_ids


def collate_examples(examples, pad_token_id):
    width = max(len(ids) for ids, _ in examples)
    return {
        "input_ids": torch.tensor([ids + [pad_token_id] * (width - len(ids)) for ids, _ in examples]),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in examples]),
        "labels": torch.tensor([labels + [IGNORE_INDEX] * (width - len(labels)) for _, labels in examples]),
    }


def read_training_data(path, tokenizer, max_length, limit):
    reader = Jsonl(path)
    examples = []
    over_length = above_limit = 0
    for row, where in reader:
        validated = {key: text_field(row, key, where) for key in ("problem", "solution")}
        encoded = encode_example(tokenizer, validated, max_length)
        if encoded is None:
            over_length += 1
        elif limit is not None and len(examples) >= limit:
            above_limit += 1
        else:
            examples.append(encoded)
    if not examples:
        raise ValueError("no training examples fit --max-length")
    return examples, {**reader.provenance(), "selected": len(examples), "over_length": over_length, "above_limit": above_limit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen model and tokenizer directory")
    parser.add_argument("--train-data", type=Path, required=True, help="JSONL with problem and solution")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for adapter and logs")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="sft")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, help="Select the first N examples that fit")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024, help="Prompt and solution combined, including EOS")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size) < 1 or args.max_length < 2:
        parser.error("--epochs and --batch-size must be positive, --max-length must be at least 2")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if not args.model.is_dir():
        parser.error(f"local model directory does not exist: {args.model}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current Python environment")

    try:
        import peft
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        parser.error(f"LoRA training requires compatible Transformers and PEFT: {exc}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        parser.error("the tokenizer must define an EOS token")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    try:
        examples, train_data = read_training_data(args.train_data, tokenizer, args.max_length, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, dtype=dtype,
    ).to(device)
    context_length = getattr(model.config, "max_position_embeddings", None)
    if context_length is not None and args.max_length > context_length:
        parser.error(f"--max-length exceeds the model context limit ({context_length})")
    torch.manual_seed(args.seed)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=list(LORA_TARGETS), lora_dropout=0.0, bias="none",
    ))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA did not expose any trainable parameters")
    initial = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)

    step = supervised_tokens = 0
    args.output_dir.mkdir(parents=True)
    print(json.dumps({"loss_type": args.loss_type, "train_data": train_data}), flush=True)
    start = time.perf_counter()
    with (args.output_dir / "steps.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(examples), generator=generator).tolist()
            for offset in range(0, len(order), args.batch_size):
                chunk = [examples[i] for i in order[offset:offset + args.batch_size]]
                batch = collate_examples(chunk, pad_token_id)
                labels = batch.pop("labels")
                tokens = labels[:, 1:].ne(IGNORE_INDEX).sum().item()
                inputs = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs, use_cache=False).logits
                loss = causal_lm_loss(logits, labels.to(device), loss_type=args.loss_type)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"nonfinite loss at step {step + 1}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True).item()
                optimizer.step()
                del logits, loss
                step += 1
                supervised_tokens += tokens
                record = json.dumps({
                    "step": step, "epoch": epoch, "loss": loss_value,
                    "supervised_tokens": tokens, "grad_norm": grad_norm,
                })
                log.write(record + "\n")
                print(record, flush=True)

    elapsed = time.perf_counter() - start
    if not any(not torch.equal(p, prev) for p, prev in zip(trainable, initial)):
        raise RuntimeError("Training did not update any LoRA parameters")
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    run = {
        "model": str(model_path), "train_data": train_data, "loss_type": args.loss_type,
        "device": args.device, "dtype": str(dtype), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "seed": args.seed, "max_length": args.max_length, "limit": args.limit, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "prompt_template": PROMPT,
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "target_modules": LORA_TARGETS},
        "trainable_parameters": sum(p.numel() for p in trainable), "parameters_updated": True,
        "steps": step, "supervised_tokens": supervised_tokens, "training_time_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
    }
    if args.loss_type == "idft":
        run["idft"] = {**IDFT_CONFIG, "warmup_ratio": 0.0, "transition_ratio": 0.0}
    (args.output_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter after {step} steps to {adapter_dir}")


if __name__ == "__main__":
    main()
