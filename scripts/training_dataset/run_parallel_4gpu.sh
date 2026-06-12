#!/usr/bin/env bash
# run_parallel_4gpu.sh — Launch dataset generation across 4 GPUs in parallel.
#
# Splits all 1228 subjects evenly across 4 workers, each pinned to one GPU
# via CUDA_VISIBLE_DEVICES. Safe to re-run — completed subjects are skipped.
# Logs per worker to logs/gpu{0-3}.log.
#
# Usage (run from project root):
#   bash scripts/training_dataset/run_parallel_4gpu.sh
#   bash scripts/training_dataset/run_parallel_4gpu.sh --dataset /path/to/Totalsegmentator_dataset_v201

set -euo pipefail

DATASET="${DATASET:-/home/vulovic/jasmina/Totalsegmentator_dataset_v201}"
SPP=384
SIZE=1024
DEVICE="GPU"
GPU_COUNT=4
TOTAL_SUBJECTS=1228

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --spp)     SPP="$2";     shift 2 ;;
        --size)    SIZE="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p logs

PER_GPU=$(( (TOTAL_SUBJECTS + GPU_COUNT - 1) / GPU_COUNT ))

echo "========================================================"
echo "Parallel dataset generation — 4x NVIDIA L4"
echo "  Total subjects : $TOTAL_SUBJECTS"
echo "  Per GPU        : ~$PER_GPU subjects"
echo "  SPP / Size     : $SPP / ${SIZE}px"
echo "  Dataset        : $DATASET"
echo "  Logs           : logs/gpu{0-3}.log"
echo "========================================================"
echo ""

for gpu_id in $(seq 0 $(( GPU_COUNT - 1 ))); do
    START=$(( gpu_id * PER_GPU ))
    LOG="logs/gpu${gpu_id}.log"
    echo "GPU $gpu_id  subjects ${START}–$(( START + PER_GPU - 1 ))  →  $LOG"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    bash scripts/training_dataset/run_dataset_generation.sh \
        --start  "$START" \
        --count  "$PER_GPU" \
        --dataset "$DATASET" \
        --spp    "$SPP" \
        --size   "$SIZE" \
        --device "$DEVICE" \
        > "$LOG" 2>&1 &
done

echo ""
echo "All 4 workers launched."
echo ""
echo "Monitor:"
echo "  tail -f logs/gpu0.log"
echo "  watch -n 30 \"find data/training_dataset -maxdepth 1 -mindepth 1 -type d | wc -l\""
echo ""
echo "Waiting for all workers to finish..."
wait
echo "Done."
