#!/usr/bin/env bash
set -euo pipefail

DISTALIGN_PY=${DISTALIGN_VLLM_PY:-/mnt/d/lfy/ustc/shopping-grpo-longhorizon/.venv/bin/python}
DISTALIGN_MODEL=${DISTALIGN_MODEL:-/mnt/d/lfy/ustc/models/Qwen2.5-Math-1.5B}
DISTALIGN_ROOT=${DISTALIGN_ROOT:-/mnt/d/lfy/ustc/DistAlign-SFT}
DISTALIGN_GPU=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)
test -n "$DISTALIGN_GPU"
export CUDA_VISIBLE_DEVICES="$DISTALIGN_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi -i 6 --query-gpu=index,name,driver_version,memory.free --format=csv
test -s "$DISTALIGN_MODEL/model.safetensors"
DISTALIGN_OUT=$(mktemp -d "$DISTALIGN_ROOT/runs/idft-prompt-probe.XXXXXX")
printf '结果目录：%s\n' "$DISTALIGN_OUT"

"$DISTALIGN_PY" -B - "$DISTALIGN_MODEL" "$DISTALIGN_ROOT/examples/evaluation.jsonl" "$DISTALIGN_OUT" <<'PY'
import json
from pathlib import Path
import sys

from vllm import LLM, SamplingParams, __version__

model, questions_file, output_dir = sys.argv[1:]
with Path(questions_file).open(encoding="utf-8") as f:
    questions = list(map(json.loads, f))
if [row["id"] for row in questions] != ["pilot-multiply", "pilot-fractions", "pilot-equation"]:
    raise SystemExit("Use the same three hand-written smoke questions as the v1 paper runner.")

prompts = {
    "chatml-v1": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "plaintext": (
        "Question: {problem}\nShow your reasoning and put the final answer in \\boxed{{}}.\nSolution:"
    ),
}
sampling = dict(
    n=1, temperature=0.3, top_p=1.0, top_k=-1, max_tokens=384,
    stop=["</s>", "<|im_end|>", "<|endoftext|>"], stop_token_ids=[151645, 151643],
)
engine = LLM(
    model=model, dtype="bfloat16", max_model_len=4096, tensor_parallel_size=1,
    generation_config="vllm", gpu_memory_utilization=0.8,
)
tokenizer = engine.get_tokenizer()
print(f"vLLM {__version__}; same temperature, seeds, cap and stops for both prompts", flush=True)

with (Path(output_dir) / "comparison.jsonl").open("x", encoding="utf-8") as out:
    for style, template in prompts.items():
        inputs = [
            {"prompt_token_ids": tokenizer.encode(template.format(problem=q["problem"]), add_special_tokens=False)}
            for q in questions
        ]
        seeds = list(range(42, 42 + len(questions)))
        params = [SamplingParams(**sampling, seed=seed) for seed in seeds]
        outputs = engine.generate(inputs, params, use_tqdm=False)
        for q, item, seed, result in zip(questions, inputs, seeds, outputs, strict=True):
            candidate, = result.outputs
            token_ids = list(candidate.token_ids)
            raw_first = tokenizer.decode(token_ids[:30], skip_special_tokens=False)
            row = {
                "style": style, "id": q["id"], "seed": seed, "tokens": len(token_ids),
                "finish": candidate.finish_reason, "stop_reason": candidate.stop_reason,
                "prompt_token_ids": item["prompt_token_ids"], "output_token_ids": token_ids,
                "raw_first": raw_first, "completion": candidate.text,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            shown = {k: row[k] for k in ("style", "id", "tokens", "finish")}
            shown["raw_first"] = raw_first[:140]
            shown["preview"] = candidate.text if len(candidate.text) <= 350 else candidate.text[:140] + " … " + candidate.text[-150:]
            print(json.dumps(shown, ensure_ascii=False), flush=True)
PY
