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
# Output goes to results/training_runs/<preset> — NOT the preset's own output_dir, which
# this launcher overrides. Set OUT to put the run somewhere else; checkpoints, samples and
# train.log then all land there together:
#   OUT=results/training_runs/rgb_baseline_v2 bash scripts/training/run_train_4gpu.sh rgb_1024
# (Do not pass --output-dir as an extra arg instead: train.py would honour it, but train.log
# would still be written to the default folder.)
#
# Uses the project venv. Appends to $OUT/train.log
# (append, not overwrite, so a --resume does not erase the earlier loss history).

set -euo pipefail

PRESET="${1:-${PRESET:-full1024}}"
if [[ $# -gt 0 ]]; then shift; fi   # the rest is passed straight to train.py
DATASET="${DATASET:-data/training_dataset}"
GPUS="${GPUS:-4}"
PY="${PY:-venv/bin/python}"

OUT="${OUT:-results/training_runs/${PRESET}}"
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
