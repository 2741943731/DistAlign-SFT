"""Select the longest existing examples that fit the training encoder."""

import argparse
import json
from pathlib import Path

from data.validate import Jsonl, text_field
from .hf_train import encode_example


def select_longest(path, tokenizer, max_length, count):
    reader = Jsonl(path)
    eligible = []
    for row, where in reader:
        fields = {key: text_field(row, key, where) for key in ("problem", "solution")}
        encoded = encode_example(tokenizer, fields, max_length)
        if encoded is not None:
            eligible.append((len(encoded[0]), row))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} existing rows fit --max-length; need {count}")
    # Python's stable sort preserves the original order in ties.
    selected = sorted(eligible, key=lambda item: -item[0])[:count]
    return selected, reader.provenance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen tokenizer directory")
    parser.add_argument("--train-data", type=Path, required=True, help="Existing training JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL in training format")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.max_length < 2 or args.count < 1:
        parser.error("--max-length must be at least 2 and --count must be positive")
    if not args.model.is_dir():
        parser.error(f"local tokenizer directory does not exist: {args.model}")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
        if tokenizer.eos_token_id is None:
            parser.error("the tokenizer must define an EOS token")
        selected, provenance = select_longest(args.train_data, tokenizer, args.max_length, args.count)
        with args.output.open("x", encoding="utf-8") as stream:
            for _, row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "source": provenance, "probe_data": str(args.output),
        "selected": len(selected), "encoded_sequence_widths": [width for width, _ in selected],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
