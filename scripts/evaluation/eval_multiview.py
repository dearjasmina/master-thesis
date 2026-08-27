"""
eval_multiview.py — cross-view consistency (implements the Stage-2 placeholder)

scripts/training/evaluate.py contains `cross_view_reprojection_gate`, which raises
NotImplementedError. This is that gate.

WHY IT MATTERS MORE THAN PSNR HERE
----------------------------------
The network is a RENDERER. It is applied per view, independently, with no knowledge
that two views show the same organ. Nothing in the training loss forbids it from
inventing different surface detail in each view — and a model that does so scores
perfectly well on PSNR/SSIM/LPIPS, because each frame is compared only against its own
ground truth. The failure only appears when you move the camera, and then it appears as
flicker, which is disqualifying for a renderer.

This is also the sharpest test of the concern raised when the v25 textures were
introduced: vessels and micro-relief are functions of OBJECT-SPACE position, which is
not recoverable from a screen-space G-buffer. If the network cannot place them
consistently, this metric is where it shows up.

METHOD
------
For each pair of views (i, j) of the same subject:
  1. unproject every pixel of view i using its metric depth and K
  2. transform to world using cam_to_world, then project into view j
  3. reject pixels landing outside the frame, behind the camera, or whose reprojected
     depth disagrees with view j's own depth map (occlusion test)
  4. on the surviving CO-VISIBLE pixels, compare warped_i against j

Reported for BOTH the generated images and the Cycles ground truth. The GT number is
the floor: it is not zero, because of interpolation error, depth quantisation and
genuinely view-dependent shading (specular highlights move). What matters is the GAP
between the model's inconsistency and the GT's — that isolates inconsistency the model
introduced from inconsistency inherent to the setup.

Blender's camera looks down -Z with +Y up; meta.json says to convert to OpenCV by
flipping rotation columns 1 and 2, which is what _cv() below does.

Usage
-----
    python scripts/evaluation/eval_multiview.py \
        --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
        --split test --subjects 12 --pairs-per-subject 6
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
                         open_dataset, to01, read_meta, view_dir_of, ensure_out,
                         bootstrap_ci)

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
from exr import read_render_exr        # noqa: E402


def _cv(c2w):
    """Blender cam-to-world -> OpenCV convention (+Z forward, +Y down)."""
    m = np.array(c2w, dtype=np.float64).reshape(4, 4).copy()
    m[:3, 1] *= -1.0
    m[:3, 2] *= -1.0
    return m


def load_depth(view_dir):
    """Metric depth for a view, from depth.exr if present else render.exr."""
    d = Path(view_dir) / "depth.exr"
    if d.exists():
        got = read_render_exr(str(d), want=("image",))
        a = got.get("image")
        if a is not None:
            return np.asarray(a)[..., 0].astype(np.float32)
    r = Path(view_dir) / "render.exr"
    if r.exists():
        got = read_render_exr(str(r), want=("depth",))
        a = got.get("depth")
        if a is not None:
            a = np.asarray(a)
            return (a[..., 0] if a.ndim == 3 else a).astype(np.float32)
    return None


def warp(img_i, depth_i, meta_i, depth_j, meta_j, depth_tol_rel=0.02):
    """Warp view i into view j. Returns (warped_image, covisible_mask)."""
    H, W = depth_i.shape
    Ki = np.array(meta_i["K"], dtype=np.float64).reshape(3, 3)
    Kj = np.array(meta_j["K"], dtype=np.float64).reshape(3, 3)
    Ci, Cj = _cv(meta_i["cam_to_world"]), _cv(meta_j["cam_to_world"])

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_i.astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-6)

    # unproject i -> camera coords -> world
    x = (uu - Ki[0, 2]) * z / Ki[0, 0]
    y = (vv - Ki[1, 2]) * z / Ki[1, 1]
    pc = np.stack([x, y, z], -1)
    pw = pc @ Ci[:3, :3].T + Ci[:3, 3]

    # world -> camera j
    Rj, tj = Cj[:3, :3], Cj[:3, 3]
    pj = (pw - tj) @ Rj                      # R^T (p - t), since R is orthonormal
    zj = pj[..., 2]
    valid &= zj > 1e-6                       # in front of camera j

    uj = Kj[0, 0] * pj[..., 0] / np.where(zj == 0, 1, zj) + Kj[0, 2]
    vj = Kj[1, 1] * pj[..., 1] / np.where(zj == 0, 1, zj) + Kj[1, 2]
    ui = np.round(uj).astype(np.int64)
    vi = np.round(vj).astype(np.int64)
    valid &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    # occlusion test: what view j actually sees at that pixel must be at the same depth
    uic = np.clip(ui, 0, W - 1)
    vic = np.clip(vi, 0, H - 1)
    dj_at = depth_j[vic, uic].astype(np.float64)
    ok_depth = np.abs(zj - dj_at) <= depth_tol_rel * np.maximum(dj_at, 1e-6)
    valid &= np.isfinite(dj_at) & (dj_at > 1e-6) & ok_depth

    out = np.zeros_like(img_i)
    out[vic[valid], uic[valid]] = img_i[vv[valid], uu[valid]]
    mask = np.zeros((H, W), bool)
    mask[vic[valid], uic[valid]] = True
    return out, mask


def masked_psnr(a, b, m):
    if m.sum() < 64:
        return float("nan")
    d = (a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2
    mse = d.mean()
    return float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--subjects", type=int, default=12, help="how many subjects to use")
    ap.add_argument("--pairs-per-subject", type=int, default=6)
    ap.add_argument("--depth-tol", type=float, default=0.02,
                    help="relative depth agreement for the occlusion test")
    ap.add_argument("--min-covisible", type=float, default=0.03,
                    help="skip pairs with less than this fraction co-visible")
    args = ap.parse_args()

    out = ensure_out(args.out)
    device = pick_device(args.device)
    cfg = build_cfg(args)
    print(f"eval_multiview — device={device}")
    G = load_generator(cfg, args.checkpoint, device)
    ds = open_dataset(cfg, args.split)

    # group sample indices by subject (the view dir's parent)
    by_subject = defaultdict(list)
    for i in range(len(ds)):
        try:
            vd = view_dir_of(ds, i)
        except RuntimeError as e:
            raise SystemExit(str(e))
        by_subject[Path(vd).parent.name].append((i, Path(vd)))
    subs = sorted(by_subject)[:args.subjects]
    print(f"  {len(subs)} subjects, up to {args.pairs_per_subject} pairs each")

    rows = []
    with torch.no_grad():
        for si, s in enumerate(subs):
            views = by_subject[s]
            if len(views) < 2:
                continue
            cache = {}
            for idx, vd in views:
                meta = read_meta(vd)
                dep = load_depth(vd)
                if meta is None or dep is None or "K" not in meta:
                    continue
                b = ds[idx]
                fake = to01(G(b["input"].unsqueeze(0).to(device))[0])
                real = to01(b["target"])
                cache[idx] = (vd, meta, dep, fake, real)

            keys = list(cache)
            pairs = [(keys[a], keys[b]) for a in range(len(keys))
                     for b in range(a + 1, len(keys))][:args.pairs_per_subject]
            for ia, ib in pairs:
                _, ma, da, fa, ra = cache[ia]
                _, mb, db, fb, rb = cache[ib]
                if da.shape != fa.shape[:2]:
                    continue
                wf, mk = warp(fa, da, ma, db, mb, args.depth_tol)
                wr, mk2 = warp(ra, da, ma, db, mb, args.depth_tol)
                m = mk & mk2
                cov = float(m.mean())
                if cov < args.min_covisible:
                    continue
                rows.append({
                    "subject": s,
                    "view_a": Path(cache[ia][0]).name,
                    "view_b": Path(cache[ib][0]).name,
                    "covisible_frac": round(cov, 4),
                    "psnr_gen": masked_psnr(wf, fb, m),
                    "psnr_gt": masked_psnr(wr, rb, m),
                })
            print(f"    [{si+1}/{len(subs)}] {s}: {len(rows)} pairs so far", end="\r")

    if not rows:
        raise SystemExit(
            "\nno usable view pairs. Most likely the depth pass is missing "
            "(need depth.exr or a Depth pass in render.exr) or meta.json lacks K.")

    f = out / "multiview_pairs.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    g = [r["psnr_gen"] for r in rows]
    t = [r["psnr_gt"] for r in rows]
    mg, lg, hg = bootstrap_ci(g)
    mt, lt, ht = bootstrap_ci(t)
    gap = mt - mg
    summary = {
        "n_pairs": len(rows),
        "mean_covisible_frac": float(np.mean([r["covisible_frac"] for r in rows])),
        "reproj_psnr_generated": {"mean": mg, "ci95": [lg, hg]},
        "reproj_psnr_ground_truth": {"mean": mt, "ci95": [lt, ht]},
        "consistency_gap_db": gap,
        "interpretation":
            "GT reprojection PSNR is the FLOOR (interpolation, depth quantisation and "
            "genuinely view-dependent shading). The gap is the inconsistency the model "
            "ADDS. A gap near 0 dB means the model is as multi-view consistent as the "
            "renderer it imitates; a large gap means it invents per-view detail, which "
            "would appear as flicker under camera motion.",
    }
    (out / "multiview_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n\n  pairs                 {len(rows)}")
    print(f"  co-visible fraction   {summary['mean_covisible_frac']:.3f}")
    print(f"  reproj PSNR (GT)      {mt:6.2f} dB   [{lt:.2f}, {ht:.2f}]   <- floor")
    print(f"  reproj PSNR (model)   {mg:6.2f} dB   [{lg:.2f}, {hg:.2f}]")
    print(f"  CONSISTENCY GAP       {gap:6.2f} dB   (lower is better)")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
