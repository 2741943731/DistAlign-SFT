#!/usr/bin/env bash
set -euo pipefail

# Run the frozen Base-v2 MATH-500 protocol with one separately validated LoRA.
cd -- "$(dirname -- "$0")/.."
DISTALIGN_ROOT=$PWD
DISTALIGN_MODE=${1:-check}
DISTALIGN_VLLM_PY=${DISTALIGN_VLLM_PY:-/mnt/d/lfy/ustc/shopping-grpo-longhorizon/.venv/bin/python}
DISTALIGN_SCORE_PY=${DISTALIGN_SCORE_PY:-$DISTALIGN_ROOT/.score-venv/bin/python}
DISTALIGN_MODEL=${DISTALIGN_MODEL:-/mnt/d/lfy/ustc/models/Qwen2.5-Math-1.5B}
DISTALIGN_QUESTIONS=${DISTALIGN_QUESTIONS:-$DISTALIGN_ROOT/data/processed/final/math500-questions.jsonl}
DISTALIGN_SMOKE_QUESTIONS=${DISTALIGN_SMOKE_QUESTIONS:-$DISTALIGN_ROOT/examples/evaluation.jsonl}
DISTALIGN_ADAPTER=${DISTALIGN_ADAPTER:-}
DISTALIGN_SMOKE_DIR=${DISTALIGN_SMOKE_DIR:-}
DISTALIGN_OUTPUT_DIR=${DISTALIGN_OUTPUT_DIR:-}
DISTALIGN_BASE_MANIFEST=${DISTALIGN_BASE_MANIFEST:-}

case "$DISTALIGN_MODE" in
  check|math500) ;;
  *) printf 'Usage: bash %s {check|math500}\n' "$0" >&2; exit 2 ;;
esac

if [[ -z "$DISTALIGN_ADAPTER" || -z "$DISTALIGN_SMOKE_DIR" || -z "$DISTALIGN_BASE_MANIFEST" ]]; then
  printf '请设置 DISTALIGN_ADAPTER、DISTALIGN_SMOKE_DIR 和 DISTALIGN_BASE_MANIFEST。\n' >&2
  exit 1
fi
if [[ "$DISTALIGN_MODE" == math500 && -z "$DISTALIGN_OUTPUT_DIR" ]]; then
  printf '请为 MATH-500 评测设置 DISTALIGN_OUTPUT_DIR。\n' >&2
  exit 1
fi
if [[ -n "${DISTALIGN_NINJA_DIR:-}" ]]; then
  export PATH="$DISTALIGN_NINJA_DIR:$PATH"
fi
if [[ -n "${DISTALIGN_NINJA_BIN:-}" ]]; then
  test -x "$DISTALIGN_NINJA_BIN"
  export PATH="$(dirname -- "$DISTALIGN_NINJA_BIN"):$PATH"
fi
if ! command -v ninja >/dev/null 2>&1; then
  printf '找不到 ninja；可用 DISTALIGN_NINJA_DIR 指定包含 ninja 的目录。\n' >&2
  exit 1
fi
test -x "$DISTALIGN_VLLM_PY"
test -x "$DISTALIGN_SCORE_PY"
test -s "$DISTALIGN_MODEL/model.safetensors"
test -s "$DISTALIGN_QUESTIONS"
test -s "$DISTALIGN_SMOKE_QUESTIONS"

DISTALIGN_GPU=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)
test -n "$DISTALIGN_GPU"
export CUDA_VISIBLE_DEVICES="$DISTALIGN_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi -i 6 --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv

# Compare the actual local bytes, model configuration, engine versions, and
# GPU with the smoke manifest, then regrade all smoke completions.  The smoke
# record must be from this adapter, not from Base or another adapter.
"$DISTALIGN_SCORE_PY" -B - "$DISTALIGN_ADAPTER" "$DISTALIGN_SMOKE_DIR" \
  "$DISTALIGN_SMOKE_QUESTIONS" "$DISTALIGN_MODEL" "$DISTALIGN_VLLM_PY" "$DISTALIGN_OUTPUT_DIR" \
  "$DISTALIGN_BASE_MANIFEST" "$DISTALIGN_QUESTIONS" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

from src.paper_eval import _adapter_metadata, _model_config_sha, _model_sha, _questions, _validate_manifest, check_smoke
from src.paper_grade import SCORER, check_dependencies, compare

adapter_path, smoke_path, questions_path, model_path, engine_python, output_path, base_path, math_path = sys.argv[1:]
smoke = Path(smoke_path).resolve()
if output_path and Path(output_path).resolve() == smoke:
    raise SystemExit("正式评测目录不能与 smoke 目录相同。")
questions, meta, _ = _questions(Path(questions_path), "smoke")
if len(questions) != 3 or meta["sha256"] != "b8fcea91f37f6c0d5281daef254f3912b1440ddf947b65d09264822e1119469e":
    raise SystemExit("smoke 题目不是固定的三道手写题。")
run = json.loads((smoke / "run.json").read_text(encoding="utf-8"))
if run.get("adapter") != _adapter_metadata(Path(adapter_path)):
    raise SystemExit("smoke 的 adapter 身份与待测 adapter 不一致。")
if run.get("model_config_files") != _model_config_sha(Path(model_path).resolve()):
    raise SystemExit("smoke 的模型配置或 tokenizer 与待测模型不一致。")
base_file = Path(base_path).resolve()
if base_file.name != "run.json":
    raise SystemExit("DISTALIGN_BASE_MANIFEST 必须指向 Base 评测目录中的 run.json。")
_, math_meta, math_rounds = _questions(Path(math_path), "math500")
_validate_manifest(base_file.parent, "math500", math_meta, math_rounds)
base = json.loads(base_file.read_text(encoding="utf-8"))
if "adapter" in base:
    raise SystemExit("对照记录不是 Base 模型。")
if _model_sha(Path(model_path).resolve()) != base.get("model_sha256"):
    raise SystemExit("当前 Base 模型权重与原 Base 评测不一致。")
if base.get("model_config_files") != run.get("model_config_files"):
    raise SystemExit("Base 评测与 smoke 的模型配置或 tokenizer 不一致。")
base_metrics = json.loads((base_file.parent / "metrics.json").read_text(encoding="utf-8"))
if (base_metrics.get("scorer") != SCORER or base_metrics.get("questions") != 500
        or base_metrics.get("rounds") != 16 or base_metrics.get("attempts") != 8000):
    raise SystemExit("Base 评分器或评测次数与固定协议不一致。")
versions = json.loads(subprocess.check_output([
    engine_python, "-B", "-c",
    "from importlib.metadata import version; import json; print(json.dumps({x:version(x) for x in ('vllm','torch','transformers')}))",
], text=True))
if any(run.get(name) != value or base.get(name) != value for name, value in versions.items()):
    raise SystemExit("Base、smoke 与当前推理环境的版本不一致。")
import os
if any(record.get("gpu") != os.environ["CUDA_VISIBLE_DEVICES"] for record in (run, base)):
    raise SystemExit("Base、smoke 与当前运行的 GPU 不一致。")
for name in ("dtype", "generation_config", "gpu_memory_utilization", "seed_rule"):
    if base.get(name) != run.get(name):
        raise SystemExit(f"Base 与 smoke 的 {name} 不一致。")
check_dependencies()
print(json.dumps(check_smoke(smoke, questions, meta, compare), ensure_ascii=False))
print("已核对 Base 与 smoke 的 adapter、模型配置、评测协议和推理环境。")
PY

if [[ "$DISTALIGN_MODE" == check ]]; then
  exit 0
fi

DISTALIGN_ARGS=(--suite math500 --questions "$DISTALIGN_QUESTIONS" --output-dir "$DISTALIGN_OUTPUT_DIR")
"$DISTALIGN_VLLM_PY" -B -m src.paper_eval generate --model "$DISTALIGN_MODEL" \
  --adapter "$DISTALIGN_ADAPTER" "${DISTALIGN_ARGS[@]}"
if [[ ! -s "$DISTALIGN_OUTPUT_DIR/metrics.json" ]]; then
  if [[ -e "$DISTALIGN_OUTPUT_DIR/graded.jsonl" ]]; then
    printf '评分不完整：graded.jsonl 已存在但 metrics.json 缺失；旧文件已保留，请在副本中重新评分。\n' >&2
    exit 1
  fi
  "$DISTALIGN_SCORE_PY" -B -m src.paper_eval score "${DISTALIGN_ARGS[@]}"
fi
cat "$DISTALIGN_OUTPUT_DIR/metrics.json"
printf '结果目录：%s\n' "$DISTALIGN_OUTPUT_DIR"
