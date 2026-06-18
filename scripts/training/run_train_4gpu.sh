#!/usr/bin/env bash
# run_train_4gpu.sh — Launch DDP training across 4 GPUs (3rd.md Stage 1).
#
# Usage (from project root):
#   bash scripts/training/run_train_4gpu.sh                       # full1024 preset
#   bash scripts/training/run_train_4gpu.sh proto512              # 512² prototype
#   PRESET=full1024 DATASET=data/training_dataset bash scripts/training/run_train_4gpu.sh
#
# Uses the project venv. Logs to results/training_runs/<preset>/train.log.

set -euo pipefail

PRESET="${1:-${PRESET:-full1024}}"
DATASET="${DATASET:-data/training_dataset}"
GPUS="${GPUS:-4}"
PY="${PY:-venv/bin/python}"

OUT="results/training_runs/${PRESET}"
mkdir -p "$OUT"

echo "========================================================"
echo "Training — deterministic G-buffer renderer (pix2pixHD)"
echo "  Preset   : $PRESET"
echo "  Dataset  : $DATASET"
echo "  GPUs     : $GPUS"
echo "  Output   : $OUT"
echo "========================================================"

"$PY" -m torch.distributed.run \
    --standalone --nproc_per_node="$GPUS" \
    scripts/training/train.py \
    --preset "$PRESET" \
    --data-root "$DATASET" \
    --output-dir "$OUT" \
    2>&1 | tee "$OUT/train.log"

echo "Done. Checkpoints in $OUT/checkpoints, samples in $OUT/samples."
