#!/usr/bin/env bash
# run_dataset_generation.sh — Batch dataset generation across TotalSegmentator subjects
#
# Iterates over subject directories that have pre-extracted meshes and runs
# generate_training_dataset.py for each. Skips subjects that are already complete
# (all 15 sample dirs present) so it's safe to re-run after interruption.
#
# Requirements:
#   - Blender accessible as `blender` in PATH (or set BLENDER_PATH below)
#   - Meshes already extracted to data/renders/totalseg/{subject}/meshes/
#     (run your mesh extraction pipeline first)
#
# Usage:
#   bash scripts/run_dataset_generation.sh
#   bash scripts/run_dataset_generation.sh --subjects "s0050 s0001 s0100"
#   bash scripts/run_dataset_generation.sh --spp 128 --size 512 --device GPU
#   bash scripts/run_dataset_generation.sh --start 0 --count 50

set -euo pipefail

# ── Configuration — edit these if needed ──────────────────────────────────────
BLENDER_PATH="${BLENDER_PATH:-blender}"
SCRIPT="scripts/training_dataset/generate_training_dataset.py"
DATASET="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201"
MESH_BASE="data/meshes"
OUTPUT_DIR="data/training_dataset"
SPP=384
SIZE=1024
DEVICE="CPU"
SUBJECTS=""
START=0
COUNT=9999  # process up to this many subjects

# ── Parse CLI overrides ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --spp)      SPP="$2";      shift 2 ;;
        --size)     SIZE="$2";     shift 2 ;;
        --device)   DEVICE="$2";   shift 2 ;;
        --start)    START="$2";    shift 2 ;;
        --count)    COUNT="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Collect subjects ───────────────────────────────────────────────────────────
if [[ -n "$SUBJECTS" ]]; then
    subject_list=($SUBJECTS)
else
    # Auto-discover: subjects that have meshes extracted under data/meshes/{subject}/
    subject_list=()
    for mesh_dir in "$MESH_BASE"/*/; do
        subj=$(basename "$mesh_dir")
        # Only include if at least one OBJ exists
        if compgen -G "$mesh_dir/*.obj" > /dev/null 2>&1; then
            subject_list+=("$subj")
        fi
    done
    # Sort for deterministic ordering
    IFS=$'\n' subject_list=($(sort <<<"${subject_list[*]}")); unset IFS
fi

total=${#subject_list[@]}
echo "========================================================"
echo "Dataset generation batch run"
echo "  Subjects found:  $total"
echo "  SPP: $SPP  |  Size: ${SIZE}px  |  Device: $DEVICE"
echo "  Output: $OUTPUT_DIR"
echo "========================================================"

processed=0
skipped=0
failed=0
idx=0

for subject in "${subject_list[@]}"; do
    # Respect --start and --count
    if (( idx < START )); then
        (( idx++ )) || true
        continue
    fi
    if (( processed >= COUNT )); then
        break
    fi
    (( idx++ )) || true

    # Check if already complete (all 15 sample dirs present)
    existing=$(find "$OUTPUT_DIR/$subject" -maxdepth 1 -type d -name "v*" 2>/dev/null | wc -l | tr -d ' ')
    if (( existing >= 20 )); then
        echo "[$idx/$total] $subject — already complete ($existing/20 views), skipping"
        (( skipped++ )) || true
        continue
    fi

    echo ""
    echo "[$idx/$total] Processing $subject (${existing}/20 views done)..."
    echo "  Started: $(date '+%H:%M:%S')"

    if "$BLENDER_PATH" --background --python "$SCRIPT" -- \
        --subject "$subject" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --spp "$SPP" \
        --size "$SIZE" \
        --device "$DEVICE" \
        2>&1 | grep -E '^\[|View \[|seg|rgb|depth|normals|ERROR|Done|warn'; then
        echo "  Finished: $(date '+%H:%M:%S')"
        (( processed++ )) || true
    else
        echo "  [ERROR] Failed for subject $subject — continuing to next"
        (( failed++ )) || true
    fi
done

echo ""
echo "========================================================"
echo "Batch complete"
echo "  Processed: $processed"
echo "  Skipped (already done): $skipped"
echo "  Failed: $failed"
echo ""
echo "Run split when you have enough subjects:"
echo "  python scripts/split_training_dataset.py"
echo "========================================================"
