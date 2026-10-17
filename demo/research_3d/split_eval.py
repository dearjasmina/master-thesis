"""
split_eval.py — the comparator for a 3DGS run: same train/test split gsplat uses with
`--test_every 5` on the 20 sorted views (test = v01, v06, v11, v16; train = the other 16).
Mesh + ULR / diffuse bake from the 16 train renders, scored on the 4 test views exactly
like gsplat scores itself (full-frame PSNR on [0,1] RGB) plus the organ-only PSNR.

Optionally scores gsplat's own test renders:  --gs-renders DIR  containing <view>.png
(1024^2 renders at the test cameras, e.g. extracted from simple_trainer's
renders/val_step29999_*.png, whose left half is GT and right half the render).

    venv/bin/python demo/research_3d/split_eval.py [--gs-renders DIR]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import common as C
from fuse_eval import blend, lookups

TEST = [v for i, v in enumerate(C.VIEWS) if i % 5 == 0]
TRAIN = [v for v in C.VIEWS if v not in TEST]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-renders", default=None)
    ap.add_argument("--gs-sbs", default=None,
                    help="glob of gsplat's side-by-side eval images, e.g. "
                         "'results/s0006_default/renders/val_step29999_*.png' ([GT | render], "
                         "sorted order = TEST order); the right half is cut out into out/gs_renders/")
    a = ap.parse_args()
    if a.gs_sbs:
        import glob
        files = sorted(glob.glob(a.gs_sbs))
        assert len(files) == len(TEST), f"expected {len(TEST)} images, got {len(files)}"
        dst = C.RES / "gs_renders"; dst.mkdir(exist_ok=True)
        for f, v in zip(files, TEST):
            im = C.load_rgb(f)
            h, w = im.shape[:2]
            assert w == 2 * h, f"{f}: not a [GT | render] pair ({w}x{h})"
            gt_half = im[:, :h]
            ref = C.load_rgb(C.OUT_DIR / f"{v}.png")
            assert C.psnr(gt_half, ref) > 40, f"{f}: left half is not {v} -- split order differs"
            C.save_rgb(dst / f"{v}.png", im[:, h:])
        a.gs_renders = str(dst)
    vs = {}
    for name in C.VIEWS:
        v = C.View(name, need=("depth", "segid", "out") + (("normals",) if name in TEST else ()))
        v.out8 = (v.out * 255 + 0.5).astype(np.uint8); del v.out
        vs[name] = v
    rows = []
    for h in TEST:
        vh = vs[h]
        fg = vh.fg
        P = vh.points_mm()[fg]; pid = vh.pid[fg]; n = vh.N[fg]
        d = vh.C_mm[None, :] - P; d /= np.linalg.norm(d, axis=1, keepdims=True)
        col, vis, dirs = lookups(P, pid, vs, TRAIN, "out")
        res, cov = blend(col, vis, dirs, n, d)
        ref = vh.out8.astype(np.float32) / 255.0
        row = {"view": h, "coverage": round(float(cov.mean()), 4)}
        for meth in ("ulr", "diffuse"):
            c = res[meth].copy()
            for p in np.unique(pid):
                s = pid == p
                if (s & cov).any():
                    c[s & ~cov] = c[s & cov].mean(0)
            img = np.zeros_like(ref); img[fg] = c
            row[f"{meth}_psnr_full"] = round(C.psnr(img, ref), 2)
            row[f"{meth}_psnr_fg"] = round(C.psnr(img, ref, fg), 2)
        if a.gs_renders:
            g = C.load_rgb(f"{a.gs_renders}/{h}.png")
            row["gs_psnr_full"] = round(C.psnr(g, ref), 2)
            row["gs_psnr_fg"] = round(C.psnr(g, ref, fg), 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    mean = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != "view"}
    print("MEAN", json.dumps(mean))
    (C.RES / "split_eval.json").write_text(json.dumps({"test": TEST, "train": TRAIN, "mean": mean,
                                                        "per_view": rows}, indent=2))


if __name__ == "__main__":
    main()
