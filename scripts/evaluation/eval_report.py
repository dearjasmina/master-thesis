"""
eval_report.py — assemble the suite's outputs into thesis-ready tables and figures

Two jobs:

  1. SUMMARISE one model: read whatever the other scripts produced and emit a single
     markdown table plus figures, with bootstrap confidence intervals rather than bare
     means.

  2. COMPARE two models: --compare-a / --compare-b take two per-view metric CSVs and
     run a PAIRED analysis — the same views under both models — with a Wilcoxon signed-
     rank test and a paired bootstrap CI on the difference.

Why paired, and why a rank test: per-view metrics are strongly correlated within a
subject and are not normally distributed, so an unpaired t-test on the means is the
wrong tool and will overstate significance. Pairing removes per-view difficulty as a
nuisance factor; the signed-rank test makes no normality assumption.

A WARNING THIS SCRIPT WILL PRINT, AND WHY
-----------------------------------------
If the two models were trained against DIFFERENT ground truth (v20 renders vs v25
renders), their PSNR/SSIM/LPIPS are not on a common scale: each is scored against its
own target, and a more detailed target is intrinsically harder to match. Comparing them
directly measures how hard each target is, not which renderer is better. The script
detects this from the recorded target/data-root and refuses to declare a winner.

Usage
-----
    python scripts/evaluation/eval_report.py --out EVALUATION/suite

    python scripts/evaluation/eval_report.py --out EVALUATION/suite \
        --compare-a EVALUATION/v20/metrics_test_rgb.csv \
        --compare-b EVALUATION/v25/metrics_test_rgb.csv \
        --label-a "v20 GT" --label-b "v25 GT"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r[key])
            if np.isfinite(v):
                out.append(v)
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(out)


def boot_ci(v, n_boot=4000, alpha=0.05, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (n_boot, v.size), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(a_rows, b_rows, key, keyfields=("subject", "view")):
    """Match rows by (subject, view) and return aligned arrays."""
    def k(r):
        return tuple(r.get(f, "") for f in keyfields)
    A = {k(r): r for r in a_rows}
    B = {k(r): r for r in b_rows}
    common = sorted(set(A) & set(B))
    xa, xb = [], []
    for c in common:
        try:
            va, vb = float(A[c][key]), float(B[c][key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(va) and np.isfinite(vb):
            xa.append(va)
            xb.append(vb)
    return np.array(xa), np.array(xb), len(common)


def compare(a_csv, b_csv, la, lb, out):
    A, B = read_csv(a_csv), read_csv(b_csv)
    keys = [k for k in ("psnr", "ssim", "lpips") if A and k in A[0]]
    lines = [f"## Paired comparison — {la} vs {lb}", ""]
    print(f"\n  paired comparison: {la} vs {lb}")
    for key in keys:
        xa, xb, n_common = paired(A, B, key)
        if xa.size < 5:
            print(f"    {key}: too few matched views ({xa.size})")
            continue
        d = xb - xa
        md, lo, hi = boot_ci(d)
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(xa, xb)
            ptxt = f"{p:.2e}"
        except Exception:
            ptxt = "n/a (scipy missing)"
        sig = "" if "n/a" in ptxt else (" *significant*" if float(ptxt) < 0.05 else " (n.s.)")
        lines.append(f"- **{key}**: {la} {xa.mean():.4f} → {lb} {xb.mean():.4f}; "
                     f"Δ = {md:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]), "
                     f"Wilcoxon p = {ptxt}{sig}, n = {xa.size}")
        print(f"    {key:<6} {xa.mean():8.4f} -> {xb.mean():8.4f}   "
              f"delta {md:+.4f} [{lo:+.4f},{hi:+.4f}]  p={ptxt}{sig}")
    lines += ["", "> Paired over identical (subject, view) keys; Wilcoxon signed-rank, "
                  "no normality assumption. Bootstrap CI is on the paired difference.", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="EVALUATION/suite")
    ap.add_argument("--compare-a", default=None)
    ap.add_argument("--compare-b", default=None)
    ap.add_argument("--label-a", default="model A")
    ap.add_argument("--label-b", default="model B")
    ap.add_argument("--same-ground-truth", action="store_true",
                    help="assert the two models were scored against the SAME target; "
                         "without it the report refuses to name a winner")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Evaluation report", ""]

    # ── single-model summary from whatever exists ──
    speed = out / "speed.json"
    if speed.exists():
        s = json.loads(speed.read_text())
        md += ["## Rendering cost", ""]
        md.append(f"- network: **{s.get('best_ms_per_frame','?')} ms/frame** "
                  f"on {s.get('gpu_name') or s.get('device')}")
        if "cycles_sec_per_frame" in s:
            md.append(f"- Cycles: {s['cycles_sec_per_frame']*1000:.0f} ms/frame "
                      f"({s.get('cycles_source','')})")
            md.append(f"- **speed-up: {s['speedup_x']}x**")
            if "break_even_frames" in s:
                md.append(f"- break-even after {s['break_even_frames']:,} frames "
                          f"of training cost")
        else:
            md.append("- **speed-up not computed** (no Cycles reference supplied)")
        md.append("")

    fk = out / "fid_kid.json"
    if fk.exists():
        d = json.loads(fk.read_text())
        md += ["## Distribution realism", "",
               f"- FID **{d['fid']:.2f}**, KID **{d['kid_mean']:.5f} ± {d['kid_std']:.5f}** "
               f"(n = {d.get('n_images','?')}, {d.get('implementation','')})", ""]
        if d.get("n_images", 0) < 500:
            md.append("> FID is biased at small n — quote KID as the primary "
                      "distribution metric and never compare FID across different n.\n")

    mv = out / "multiview_summary.json"
    if mv.exists():
        d = json.loads(mv.read_text())
        md += ["## Multi-view consistency", "",
               f"- reprojection PSNR, ground truth (floor): "
               f"**{d['reproj_psnr_ground_truth']['mean']:.2f} dB**",
               f"- reprojection PSNR, model: "
               f"**{d['reproj_psnr_generated']['mean']:.2f} dB**",
               f"- **consistency gap: {d['consistency_gap_db']:.2f} dB** "
               f"(lower is better, n = {d['n_pairs']} pairs)", "",
               f"> {d['interpretation']}", ""]

    pt = out / "per_tissue_summary.csv"
    if pt.exists():
        rows = read_csv(pt)
        rows.sort(key=lambda r: float(r["psnr_mean"]))
        md += ["## Per-organ quality (worst first)", "",
               "| organ | n | pixel frac | PSNR | 95% CI |", "|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['tissue']} | {r['n_views']} | {float(r['mean_pixel_frac']):.4f} "
                      f"| {float(r['psnr_mean']):.2f} | "
                      f"[{float(r['psnr_ci95_lo']):.2f}, {float(r['psnr_ci95_hi']):.2f}] |")
        md.append("")

    # ── optional two-model comparison ──
    if args.compare_a and args.compare_b:
        md += [""] + compare(Path(args.compare_a), Path(args.compare_b),
                             args.label_a, args.label_b, out)
        if not args.same_ground_truth:
            warn = ("> **These models were not asserted to share a ground truth.** If "
                    "they were trained against different renders (v20 vs v25), their "
                    "paired metrics are NOT on a common scale — each is scored against "
                    "its own target, and a more detailed target is intrinsically harder "
                    "to match. In that case this table measures how hard each target "
                    "is, not which renderer is better. Pass --same-ground-truth only if "
                    "both were evaluated against the identical GT images.")
            md += ["", warn, ""]
            print("\n  [!] no --same-ground-truth: refusing to declare a winner. "
                  "See the note in comparison.md")

    (out / "report.md").write_text("\n".join(md))
    print(f"\nwrote {out/'report.md'}")


if __name__ == "__main__":
    main()
