#!/usr/bin/env bash
# run_train_4gpu.sh — Launch DDP training across 4 GPUs (3rd.md Stage 1).
#
# Usage (from project root):
#   bash scripts/training/run_train_4gpu.sh                       # full1024 preset
#   bash scripts/training/run_train_4gpu.sh proto512              # 512² prototype
#   PRESET=full1024 DATASET=data/training_dataset bash scripts/training/run_train_4gpu.sh
#
# Any extra args after the preset are forwarded verbatim to train.py, e.g.
#   bash scripts/training/run_train_4gpu.sh rgb_1024 --epochs 60
#   bash scripts/training/run_train_4gpu.sh full1024 --resume results/.../latest.pt
#
# Uses the project venv. Appends to results/training_runs/<preset>/train.log
# (append, not overwrite, so a --resume does not erase the earlier loss history).

set -euo pipefail

PRESET="${1:-${PRESET:-full1024}}"
if [[ $# -gt 0 ]]; then shift; fi   # the rest is passed straight to train.py
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
    "$@" \
    2>&1 | tee -a "$OUT/train.log"

echo "Done. Checkpoints in $OUT/checkpoints, samples in $OUT/samples."
