"""
eval_per_tissue.py — which ORGANS the network gets right

A single test-set PSNR hides everything useful. The interesting questions for the
thesis are per-organ: does it reproduce liver but fail on lung? Are the organs with
strong procedural texture (bowel vasa recta, lung septa) systematically worse than the
smooth ones? Does error track organ SIZE, i.e. is it just a pixel-count effect?

That last one matters: a whole-image metric is dominated by whichever organ occupies
the most pixels, so an apparently good model can be failing completely on every small
structure. Only a per-tissue breakdown separates those.

Uses the IndexOB pass (tissue pass_index) from render.exr together with tissue_ids.json
to label each pixel, then computes masked PSNR/L1/LPIPS per organ. LPIPS is patch-based
and cannot be masked meaningfully, so it is computed on the organ's bounding-box crop
and flagged as approximate rather than silently reported as exact.

Usage
-----
    python scripts/evaluation/eval_per_tissue.py \
        --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
        --split test --max-samples 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_common import (add_common_args, build_cfg, pick_device, load_generator,
                         open_dataset, to01, view_dir_of, sample_indices, ensure_out,
                         bootstrap_ci)

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
from exr import read_render_exr        # noqa: E402


def load_tissue_ids(data_root):
    """tissue_ids.json maps organ name -> pass_index, written by the generator."""
    for c in (Path(data_root) / "tissue_ids.json",
              Path(data_root).parent / "tissue_ids.json"):
        if c.exists():
            d = json.loads(c.read_text())
            ids = d.get("tissues", d)
            return {int(v): k for k, v in ids.items()}, c
    return None, None


def load_indexob(view_dir):
    d = Path(view_dir) / "segid.exr"
    if d.exists():
        got = read_render_exr(str(d), want=("image",))
        a = got.get("image")
        if a is not None:
            a = np.asarray(a)
            return np.rint((a[..., 0] if a.ndim == 3 else a)).astype(np.int32)
    r = Path(view_dir) / "render.exr"
    if r.exists():
        got = read_render_exr(str(r), want=("indexob",))
        a = got.get("indexob")
        if a is not None:
            a = np.asarray(a)
            return np.rint((a[..., 0] if a.ndim == 3 else a)).astype(np.int32)
    return None


def masked_psnr(a, b, m):
    if m.sum() < 64:
        return float("nan")
    mse = ((a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2).mean()
    return float("inf") if mse <= 0 else float(10 * np.log10(1.0 / mse))


def masked_l1(a, b, m):
    if m.sum() < 64:
        return float("nan")
    return float(np.abs(a[m].astype(np.float64) - b[m].astype(np.float64)).mean())


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--lpips", action="store_true",
                    help="also compute bounding-box LPIPS per organ (slower, approximate)")
    args = ap.parse_args()

    out = ensure_out(args.out)
    device = pick_device(args.device)
    cfg = build_cfg(args)
    print(f"eval_per_tissue — device={device}")
    G = load_generator(cfg, args.checkpoint, device)
    ds = open_dataset(cfg, args.split)

    id2name, src = load_tissue_ids(cfg.data.root)
    if id2name is None:
        print("  [warn] tissue_ids.json not found — organs will be reported by numeric id")
        id2name = {}
    else:
        print(f"  tissue ids: {len(id2name)} organs from {src}")

    lp = None
    if args.lpips:
        from losses import build_lpips
        lp = build_lpips(device)

    idxs = sample_indices(len(ds), args.max_samples)
    rows = []
    with torch.no_grad():
        for n, i in enumerate(idxs):
            vd = view_dir_of(ds, i)
            idx_map = load_indexob(vd)
            if idx_map is None:
                continue
            b = ds[i]
            fake = to01(G(b["input"].unsqueeze(0).to(device))[0])
            real = to01(b["target"])
            if idx_map.shape != fake.shape[:2]:
                continue
            subject = Path(vd).parent.name
            for tid in np.unique(idx_map):
                if tid <= 0:                       # 0 is background
                    continue
                m = idx_map == tid
                frac = float(m.mean())
                if m.sum() < 256:
                    continue
                r = {"subject": subject, "view": Path(vd).name,
                     "tissue_id": int(tid),
                     "tissue": id2name.get(int(tid), f"id_{int(tid)}"),
                     "pixel_frac": round(frac, 5),
                     "psnr": masked_psnr(real, fake, m),
                     "l1": masked_l1(real, fake, m)}
                if lp is not None:
                    ys, xs = np.where(m)
                    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    if (y1 - y0) > 15 and (x1 - x0) > 15:
                        fa = torch.from_numpy(fake[y0:y1, x0:x1]).permute(2, 0, 1)[None]
                        ra = torch.from_numpy(real[y0:y1, x0:x1]).permute(2, 0, 1)[None]
                        r["lpips_bbox"] = float(lp(fa.to(device) * 2 - 1,
                                                   ra.to(device) * 2 - 1).mean())
                rows.append(r)
            if n % 25 == 0:
                print(f"    {n}/{len(idxs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("\nno per-tissue rows — is the IndexOB/segid pass present?")

    f = out / "per_tissue_views.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[r["tissue"]].append(r)
    summary = []
    for t, rs in agg.items():
        m, lo, hi = bootstrap_ci([x["psnr"] for x in rs])
        summary.append({
            "tissue": t, "n_views": len(rs),
            "mean_pixel_frac": float(np.mean([x["pixel_frac"] for x in rs])),
            "psnr_mean": m, "psnr_ci95_lo": lo, "psnr_ci95_hi": hi,
            "l1_mean": float(np.nanmean([x["l1"] for x in rs])),
        })
    summary.sort(key=lambda d: d["psnr_mean"])

    g = out / "per_tissue_summary.csv"
    with open(g, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"\n\n  {'tissue':<30}{'n':>6}{'px frac':>9}{'PSNR':>8}  95% CI")
    for s in summary:
        print(f"  {s['tissue']:<30}{s['n_views']:>6}{s['mean_pixel_frac']:>9.4f}"
              f"{s['psnr_mean']:>8.2f}  [{s['psnr_ci95_lo']:.2f}, {s['psnr_ci95_hi']:.2f}]")

    # Is the ranking just organ size? If so, per-organ differences say little.
    fr = np.array([s["mean_pixel_frac"] for s in summary])
    ps = np.array([s["psnr_mean"] for s in summary])
    ok = np.isfinite(fr) & np.isfinite(ps)
    if ok.sum() > 2:
        rho = float(np.corrcoef(fr[ok], ps[ok])[0, 1])
        print(f"\n  correlation(pixel fraction, PSNR) = {rho:+.2f}")
        print("  |rho| > 0.7 would mean the ranking mostly reflects organ SIZE rather "
              "than difficulty — interpret per-organ differences cautiously in that case.")
    print(f"\nwrote {f}\n      {g}")


if __name__ == "__main__":
    main()
