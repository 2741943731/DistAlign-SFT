"""Run a tiny CPU train step through the shared SFT/DFT objective."""

import argparse
import json
import math
from pathlib import Path
import tomllib


BOS, EOS, PAD = 256, 257, 258
IGNORE_INDEX = -100


def encode(row, max_length):
    problem, solution = row["problem"], row["solution"]
    if not all(isinstance(value, str) and value.strip() for value in (problem, solution)):
        raise ValueError("problem and solution must be nonempty strings")
    prompt = [BOS, *f"Question: {problem}\nSolution: ".encode("utf-8")]
    answer = [*solution.encode("utf-8"), EOS]
    if len(prompt) + len(answer) > max_length:
        return None
    return prompt + answer, [IGNORE_INDEX] * len(prompt) + answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML smoke configuration")
    parser.add_argument("--train-data", type=Path, required=True, help="JSONL with problem and solution")
    args = parser.parse_args()

    import torch
    from torch import nn

    from .losses import causal_lm_loss

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    loss_type = config["loss_type"]
    steps, batch_size, max_length = (config[key] for key in ("steps", "batch_size", "max_length"))
    if min(steps, batch_size, max_length) < 1:
        parser.error("steps, batch_size and max_length must be positive")
    if loss_type not in {"sft", "dft"}:
        parser.error("loss_type must be sft or dft")

    examples = []
    over_length = 0
    with args.train_data.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                example = encode(json.loads(line), max_length)
            except (ValueError, KeyError, TypeError) as exc:
                parser.error(f"{args.train_data}:{line_no}: {exc}")
            if example is None:
                over_length += 1
            else:
                examples.append(example)
    if not examples:
        parser.error("no examples fit max_length")

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(PAD + 1, 32, padding_idx=PAD)
            self.rnn = nn.GRU(32, 32, batch_first=True)
            self.output = nn.Linear(32, PAD + 1)

        def forward(self, tokens):
            hidden, _ = self.rnn(self.embedding(tokens))
            return self.output(hidden)

    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    print(json.dumps({"loss_type": loss_type, "examples": len(examples), "over_length": over_length}))
    for step in range(1, steps + 1):
        indices = torch.randint(len(examples), (batch_size,), generator=generator).tolist()
        batch = [examples[index] for index in indices]
        width = max(len(tokens) for tokens, _ in batch)
        tokens = torch.tensor([ids + [PAD] * (width - len(ids)) for ids, _ in batch])
        labels = torch.tensor([ys + [IGNORE_INDEX] * (width - len(ys)) for _, ys in batch])
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model(tokens), labels, loss_type=loss_type)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"nonfinite {loss_type} loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({
            "step": step,
            "loss": loss.item(),
            "supervised_tokens": labels[:, 1:].ne(IGNORE_INDEX).sum().item(),
            "grad_norm": grad_norm.item(),
        }))


if __name__ == "__main__":
    main()
