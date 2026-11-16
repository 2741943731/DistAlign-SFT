#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
DISTALIGN_ROOT=$PWD
DISTALIGN_VLLM_PY=${DISTALIGN_VLLM_PY:-/mnt/d/lfy/ustc/shopping-grpo-longhorizon/.venv/bin/python}
DISTALIGN_SCORE_PY=${DISTALIGN_SCORE_PY:-$DISTALIGN_ROOT/.score-venv/bin/python}
DISTALIGN_MODEL=${DISTALIGN_MODEL:-/mnt/d/lfy/ustc/models/Qwen2.5-Math-1.5B}
DISTALIGN_MODE=${1:-check}

if [[ "$DISTALIGN_MODE" == setup-score ]]; then
  /mnt/d/lfy/anaconda3/envs/nanovllm/bin/python -m venv "$DISTALIGN_ROOT/.score-venv"
  "$DISTALIGN_ROOT/.score-venv/bin/python" -m pip install -r "$DISTALIGN_ROOT/requirements-paper-score.txt"
  exit
fi

case "$DISTALIGN_MODE" in
  check|smoke|math500) ;;
  *) printf 'Usage: bash %s {check|setup-score|smoke|math500}\n' "$0" >&2; exit 2 ;;
esac

DISTALIGN_GPU=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)
test -n "$DISTALIGN_GPU"
export CUDA_VISIBLE_DEVICES="$DISTALIGN_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi -i 6 --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
test -s "$DISTALIGN_MODEL/model.safetensors"
"$DISTALIGN_VLLM_PY" -B -c 'import torch; from vllm import LLM, SamplingParams, __version__; print("vLLM", __version__, "torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "需要一张支持 BF16 的 CUDA 卡"'
if [[ ! -x "$DISTALIGN_SCORE_PY" ]]; then
  printf '请先运行 setup-score，创建独立的评分环境。\n' >&2
  exit 1
fi
"$DISTALIGN_SCORE_PY" -B -c 'from src.paper_grade import check_dependencies; check_dependencies(); print("Scoring dependencies OK")'
if [[ "$DISTALIGN_MODE" == check ]]; then
  exit
fi

DISTALIGN_SMOKE_QUESTIONS="$DISTALIGN_ROOT/examples/evaluation.jsonl"
DISTALIGN_SMOKE_OUTPUT="$DISTALIGN_ROOT/results/smoke"
distalign_check_smoke() {
  "$DISTALIGN_SCORE_PY" -B -m src.paper_eval check-smoke --suite smoke \
    --questions "$DISTALIGN_SMOKE_QUESTIONS" --output-dir "$DISTALIGN_SMOKE_OUTPUT"
}

if [[ "$DISTALIGN_MODE" == smoke ]]; then
  DISTALIGN_QUESTIONS="$DISTALIGN_SMOKE_QUESTIONS"
else
  distalign_check_smoke || {
    printf '手写题试跑未通过，MATH-500 未启动。\n' >&2
    exit 1
  }
  DISTALIGN_QUESTIONS="$DISTALIGN_ROOT/data/processed/final/math500-questions.jsonl"
fi
DISTALIGN_OUTPUT="$DISTALIGN_ROOT/results/$DISTALIGN_MODE"
DISTALIGN_ARGS=(--suite "$DISTALIGN_MODE" --questions "$DISTALIGN_QUESTIONS" --output-dir "$DISTALIGN_OUTPUT")
"$DISTALIGN_VLLM_PY" -B -m src.paper_eval generate --model "$DISTALIGN_MODEL" "${DISTALIGN_ARGS[@]}"
if [[ ! -s "$DISTALIGN_OUTPUT/metrics.json" ]]; then
  "$DISTALIGN_SCORE_PY" -B -m src.paper_eval score "${DISTALIGN_ARGS[@]}"
fi
cat "$DISTALIGN_OUTPUT/metrics.json"
printf '结果目录：%s\n' "$DISTALIGN_OUTPUT"
if [[ "$DISTALIGN_MODE" == smoke ]]; then
  distalign_check_smoke
fi
