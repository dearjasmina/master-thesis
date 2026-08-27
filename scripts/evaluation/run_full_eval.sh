#!/usr/bin/env bash
# run_full_eval.sh — complete evaluation of one trained checkpoint.
#
# Run this BEFORE deleting or regenerating the dataset. Every metric here needs the
# ground-truth images; once the dataset is gone none of it can be recomputed.
#
# Usage:
#   bash scripts/evaluation/run_full_eval.sh \
#        results/training_runs/full1024/checkpoints/latest.pt \
#        EVALUATION/v20_network \
#        [CYCLES_SEC_PER_FRAME]
#
# The third argument is the measured Cycles cost per rendered view. Without it the
# speed-up — the headline claim of the thesis — is not computed. Get it from your
# generation logs, e.g.:
#   grep -A2 "Processing" logs/gpu0.log | head -40
# or pass --cycles-log to eval_speed.py directly.

set -euo pipefail

CKPT="${1:?usage: run_full_eval.sh <checkpoint.pt> <outdir> [cycles_sec_per_frame]}"
OUT="${2:?usage: run_full_eval.sh <checkpoint.pt> <outdir> [cycles_sec_per_frame]}"
CYCLES="${3:-}"
PRESET="${PRESET:-full1024}"
SPLIT="${SPLIT:-test}"
PY="${PY:-venv/bin/python}"
EV="scripts/evaluation"

mkdir -p "$OUT"
echo "=============================================================="
echo " Full evaluation"
echo "   checkpoint : $CKPT"
echo "   preset     : $PRESET      split: $SPLIT"
echo "   output     : $OUT"
echo "=============================================================="

echo
echo "[1/5] baseline PSNR / SSIM / LPIPS  (existing script)"
"$PY" scripts/training/evaluate.py \
    --preset "$PRESET" --checkpoint "$CKPT" --split "$SPLIT" \
    --out "$OUT" --grids 16 --worst 12

echo
echo "[2/5] per-organ breakdown"
"$PY" "$EV/eval_per_tissue.py" \
    --preset "$PRESET" --checkpoint "$CKPT" --split "$SPLIT" \
    --out "$OUT" --max-samples 400

echo
echo "[3/5] distribution realism (FID / KID)"
"$PY" "$EV/eval_fid_kid.py" \
    --preset "$PRESET" --checkpoint "$CKPT" --split "$SPLIT" \
    --out "$OUT"

echo
echo "[4/5] multi-view consistency"
"$PY" "$EV/eval_multiview.py" \
    --preset "$PRESET" --checkpoint "$CKPT" --split "$SPLIT" \
    --out "$OUT" --subjects 15 --pairs-per-subject 6

echo
echo "[5/5] inference speed"
if [[ -n "$CYCLES" ]]; then
    "$PY" "$EV/eval_speed.py" \
        --preset "$PRESET" --checkpoint "$CKPT" --out "$OUT" \
        --cycles-sec-per-frame "$CYCLES"
else
    echo "  (no Cycles reference given — speed-up will NOT be computed)"
    "$PY" "$EV/eval_speed.py" \
        --preset "$PRESET" --checkpoint "$CKPT" --out "$OUT"
fi

echo
echo "[report]"
"$PY" "$EV/eval_report.py" --out "$OUT"

echo
echo "=============================================================="
echo " Done. Report: $OUT/report.md"
echo " Keep this whole folder — it cannot be regenerated once the"
echo " dataset is deleted."
echo "=============================================================="
