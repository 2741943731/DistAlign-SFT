#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
DISTALIGN_ROOT=$PWD
DISTALIGN_TRAIN_PY=${DISTALIGN_TRAIN_PY:-/mnt/d/lfy/anaconda3/envs/nanovllm/bin/python}
DISTALIGN_MODEL=${DISTALIGN_MODEL:-/mnt/d/lfy/ustc/models/Qwen2.5-Math-1.5B}
DISTALIGN_PROBE_TRAIN=${DISTALIGN_PROBE_TRAIN:-/mnt/d/lfy/ustc/DistAlign-SFT/data/processed/train.jsonl}
DISTALIGN_DATA_VENV="$DISTALIGN_ROOT/.data-venv"
DISTALIGN_MODE=${1:-probe}

check_data_python() {
  "$1" -B -c 'from importlib.metadata import PackageNotFoundError, version
import sys
try:
    found = version("datasets")
except PackageNotFoundError:
    sys.exit("缺少 datasets==4.4.1")
if found != "4.4.1":
    sys.exit(f"需要 datasets==4.4.1；当前是 {found}")
print("datasets", found, sys.executable)'
}

case "$DISTALIGN_MODE" in
  probe)
    test -x "$DISTALIGN_TRAIN_PY"
    test -s "$DISTALIGN_MODEL/model.safetensors"
    test -s "$DISTALIGN_PROBE_TRAIN"
    DISTALIGN_GPU=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)
    test -n "$DISTALIGN_GPU"
    export CUDA_VISIBLE_DEVICES="$DISTALIGN_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    nvidia-smi -i 6 --query-gpu=index,name,memory.total,memory.free --format=csv
    mkdir -p "$DISTALIGN_ROOT/results"
    DISTALIGN_PROBE=$(mktemp -d "$DISTALIGN_ROOT/results/sft-probe.XXXXXX")
    printf '试跑目录：%s\n' "$DISTALIGN_PROBE"
    "$DISTALIGN_TRAIN_PY" -B -m src.numina_sft_probe --model "$DISTALIGN_MODEL" \
      --train-data "$DISTALIGN_PROBE_TRAIN" --output "$DISTALIGN_PROBE/train-longest-8.jsonl" \
      --count 8 --max-length 2048
    "$DISTALIGN_TRAIN_PY" -B -m src.hf_train --model "$DISTALIGN_MODEL" \
      --train-data "$DISTALIGN_PROBE/train-longest-8.jsonl" --output-dir "$DISTALIGN_PROBE/train" \
      --device cuda --loss-type sft --batch-size 1 --gradient-accumulation-steps 4 \
      --limit 8 --max-length 2048 --epochs 1 --learning-rate 5e-5 --adam-beta2 .95 --weight-decay .01
    "$DISTALIGN_TRAIN_PY" -B - "$DISTALIGN_PROBE/train/run.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
run = json.loads(path.read_text(encoding="utf-8"))
if (run["optimizer_steps"], run["micro_batches"], run["train_data"]["selected"]) != (2, 8, 8):
    raise SystemExit("试跑预期为 8 个样本、8 个微批次和 2 次优化器更新。")
print(json.dumps({
    "run": str(path), "optimizer_steps": run["optimizer_steps"], "micro_batches": run["micro_batches"],
    "max_observed_sequence_width": run["max_observed_sequence_width"],
    "training_time_seconds": round(run["training_time_seconds"], 2),
    "peak_cuda_memory_gib": round(run["peak_cuda_memory_bytes"] / 2**30, 3),
    "peak_cuda_memory_reserved_gib": round(run["peak_cuda_memory_reserved_bytes"] / 2**30, 3),
}, ensure_ascii=False))
PY
    ;;
  setup-data)
    if [[ -n "${DISTALIGN_DATA_PY:-}" ]]; then
      check_data_python "$DISTALIGN_DATA_PY"
      exit
    fi
    DISTALIGN_DATA_PY="$DISTALIGN_DATA_VENV/bin/python"
    if [[ -x "$DISTALIGN_DATA_PY" ]] && check_data_python "$DISTALIGN_DATA_PY"; then
      exit
    fi
    "$DISTALIGN_TRAIN_PY" -B -m venv "$DISTALIGN_DATA_VENV"
    "$DISTALIGN_DATA_PY" -B -m pip install --disable-pip-version-check 'datasets==4.4.1'
    check_data_python "$DISTALIGN_DATA_PY"
    ;;
  prepare)
    DISTALIGN_DATA_PY=${DISTALIGN_DATA_PY:-$DISTALIGN_DATA_VENV/bin/python}
    if [[ ! -x "$DISTALIGN_DATA_PY" ]]; then
      printf '找不到数据环境。请先运行 setup-data，或用 DISTALIGN_DATA_PY 指定其 Python。\n' >&2
      exit 1
    fi
    check_data_python "$DISTALIGN_DATA_PY"
    DISTALIGN_HOLDOUT="$DISTALIGN_ROOT/data/processed/final/math500-holdout-problems.jsonl"
    DISTALIGN_DATA_OUT="$DISTALIGN_ROOT/results/data"
    test -s "$DISTALIGN_HOLDOUT"
    if [[ -e "$DISTALIGN_DATA_OUT" || -L "$DISTALIGN_DATA_OUT" ]]; then
      printf '数据输出已存在：%s\n' "$DISTALIGN_DATA_OUT" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$DISTALIGN_DATA_PY" -B -m data.numina_pilot \
      --held-out "$DISTALIGN_HOLDOUT" --output-dir "$DISTALIGN_DATA_OUT" \
      --candidate-limit "${DISTALIGN_CANDIDATE_LIMIT:-16384}"
    ;;
  *)
    printf 'Usage: bash %s {probe|setup-data|prepare}\n' "$0" >&2
    exit 2
    ;;
esac
