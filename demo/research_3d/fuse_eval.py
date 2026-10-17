"""
fuse_eval.py — leave-one-out test of "fuse the 20 neural renders into one 3-D model".

For every held-out view h (all 20): take h's own geometry (depth.exr -> world mm via
camera_model.unproject; that is exactly what the mesh rasterises to, see align.py), look
the surface points up in the OTHER 19 neural renders (project, depth test + organ-id test,
bilinear colour), and predict h's image three ways:

  diffuse  view-independent bake: cos^2-weighted mean over all visible views
           (= per-texel / per-point colour; what (a) point cloud and (b) baking converge to
           at full resolution)
  ulr      unstructured-lumigraph blend (Buehler 2001): the 3 visible views whose ray is
           angularly closest to h's ray, weights (1 - th/th4)/th * cos  (view-dependent
           projective texturing in the browser)
  nearest  single angularly closest visible view (shows seams)

Same procedure on the Cycles ground truth (rgb_preview.png) gives the FLOOR: how much of
the error is genuine view-dependent shading that no fused model can reproduce, vs.
inconsistency the network adds.

    venv/bin/python demo/research_3d/fuse_eval.py
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

import common as C
import camera_model as cm

TOL_MM = 3.0
import os
VIS_MODE = os.environ.get("VIS_MODE", "depth")      # "depth" (shader-realistic) | "strict"
FIG_VIEWS = ("v02_az+0_el+0", "v07_az+72_el+30", "v13_az+216_el-50")


def load_views():
    vs = {}
    for name in C.VIEWS:
        v = C.View(name, need=("depth", "segid", "normals", "out", "gt"))
        v.out8 = (v.out * 255 + 0.5).astype(np.uint8); del v.out
        v.gt8 = (v.gt * 255 + 0.5).astype(np.uint8); del v.gt
        vs[name] = v
    return vs


def lookups(P, pid, vs, src_names, which):
    """For surface points P (N,3 mm) with organ ids pid, sample every source view.
    Returns colours (S,N,3), visibility (S,N) and unit view dirs (S,N,3) point->camera."""
    S, N = len(src_names), len(P)
    col = np.zeros((S, N, 3), np.float32)
    vis = np.zeros((S, N), bool)
    dirs = np.zeros((S, N, 3), np.float32)
    for s, name in enumerate(src_names):
        v = vs[name]
        u, w, Z = cm.project(P.astype(np.float64), v.meta)
        ok = (Z > 0) & (u >= 0) & (u <= 1022.999) & (w >= 0) & (w <= 1022.999)
        ui = np.clip(np.rint(u).astype(np.int64), 0, 1023)
        wi = np.clip(np.rint(w).astype(np.int64), 0, 1023)
        ok &= np.abs(v.Z[wi, ui] - Z) * 1000.0 < TOL_MM
        if VIS_MODE == "strict":
            # all four bilinear taps on the same organ (no colour bleeding at organ edges)
            x0 = np.clip(np.floor(u).astype(np.int64), 0, 1022)
            y0 = np.clip(np.floor(w).astype(np.int64), 0, 1022)
            for dy in (0, 1):
                for dx in (0, 1):
                    ok &= v.pid[y0 + dy, x0 + dx] == pid
        # VIS_MODE == "depth": depth test only -- exactly what a shadow-map test in the
        # WebGL shader does (the browser has no segid maps)
        img = v.out8 if which == "out" else v.gt8
        col[s] = C.bilinear(img, u, w).astype(np.float32) / 255.0
        vis[s] = ok
        d = v.C_mm[None, :] - P
        dirs[s] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return col, vis, dirs


def blend(col, vis, dirs, n, d_h):
    """-> dict(method -> (N,3) colour, covered mask)."""
    cos_n = np.clip(np.einsum("snk,nk->sn", dirs, n), 0, 1)
    covered = vis.any(0)
    res = {}
    # diffuse
    w = np.where(vis, cos_n ** 2 + 1e-6, 0.0)
    res["diffuse"] = (w[..., None] * col).sum(0) / np.maximum(w.sum(0), 1e-12)[:, None]
    # angles to the target ray
    th = np.arccos(np.clip(np.einsum("snk,nk->sn", dirs, d_h), -1, 1))
    th = np.where(vis, th, np.inf)
    order = np.argsort(th, axis=0)
    th_sorted = np.take_along_axis(th, order, 0)
    k = 3
    th4 = np.where(np.isfinite(th_sorted[k]), th_sorted[k], np.pi)
    wu = np.zeros_like(th)
    for j in range(k):
        idx = order[j]
        t = th_sorted[j]
        wj = np.where(np.isfinite(t), np.clip(1 - t / th4, 0, 1) / (t + 1e-3), 0.0)
        wj = wj * np.maximum(np.take_along_axis(cos_n, idx[None], 0)[0], 0.05)
        np.put_along_axis(wu, idx[None], wj[None], 0)
    # degenerate (th == th4 for all) -> fall back to nearest
    zero = wu.sum(0) <= 0
    np.put_along_axis(wu, order[0][None], np.where(zero, 1.0, np.take_along_axis(wu, order[0][None], 0)[0])[None], 0)
    wu = np.where(vis, wu, 0.0)
    res["ulr"] = (wu[..., None] * col).sum(0) / np.maximum(wu.sum(0), 1e-12)[:, None]
    res["nearest"] = np.take_along_axis(col, order[0][None, :, None].repeat(3, 2), 0)[0]
    # nearest-view angle stats (how far the closest source ray is)
    res["_th_min_deg"] = np.degrees(th_sorted[0])
    return res, covered


def fill_holes(img, covered_img, fg, pid_img, pred_cols_by_pid):
    out = img.copy()
    holes = fg & ~covered_img
    for p, c in pred_cols_by_pid.items():
        out[holes & (pid_img == p)] = c
    return out


def main():
    t0 = time.time()
    vs = load_views()
    print(f"loaded {len(vs)} views in {time.time()-t0:.1f}s, RSS {C.mem_gb():.2f} GB")
    rows = []
    for h in C.VIEWS:
        t1 = time.time()
        vh = vs[h]
        fg = vh.fg
        P = vh.points_mm()[fg]
        pid = vh.pid[fg]
        n = vh.N[fg]
        d_h = vh.C_mm[None, :] - P
        d_h = d_h / np.linalg.norm(d_h, axis=1, keepdims=True)
        src = [s for s in C.VIEWS if s != h]
        out_h = vh.out8.astype(np.float32) / 255.0
        gt_h = vh.gt8.astype(np.float32) / 255.0
        row = {"view": h, "fg_px": int(fg.sum())}
        figs = {}
        for which, ref in (("out", out_h), ("gt", gt_h)):
            col, vis, dirs = lookups(P, pid, vs, src, which)
            res, covered = blend(col, vis, dirs, n, d_h)
            cov_img = np.zeros(fg.shape, bool); cov_img[fg] = covered
            row[f"{which}_coverage"] = round(float(covered.mean()), 4)
            row[f"{which}_n_visible_mean"] = round(float(vis.sum(0).mean()), 2)
            if which == "out":
                row["nearest_src_angle_deg_median"] = round(float(np.median(res["_th_min_deg"][covered])), 1)
            for meth in ("diffuse", "ulr", "nearest"):
                img = np.zeros_like(ref)
                img[fg] = res[meth]
                organ_mean = {int(p): res[meth][covered & (pid == p)].mean(0)
                              for p in np.unique(pid) if (covered & (pid == p)).any()}
                img = fill_holes(img, cov_img, fg, vh.pid, organ_mean)
                row[f"{which}_{meth}_psnr_fg_covered"] = round(C.psnr(img, ref, cov_img), 2)
                row[f"{which}_{meth}_psnr_full"] = round(C.psnr(img, ref), 2)
                figs[(which, meth)] = img
            del col, vis, dirs
        row["net_vs_gt_psnr_full"] = round(C.psnr(out_h, gt_h), 2)
        row["net_vs_gt_psnr_fg"] = round(C.psnr(out_h, gt_h, fg), 2)
        # fused-from-network, judged against Cycles (what the viewer shows vs the truth)
        row["out_ulr_vs_gt_psnr_full"] = round(C.psnr(figs[("out", "ulr")], gt_h), 2)
        row["out_diffuse_vs_gt_psnr_full"] = round(C.psnr(figs[("out", "diffuse")], gt_h), 2)
        row["seconds"] = round(time.time() - t1, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if h in FIG_VIEWS:
            tiles = [C.label(out_h, "held-out neural render"),
                     C.label(figs[("out", "ulr")], "ULR from other 19"),
                     C.label(figs[("out", "diffuse")], "diffuse bake from 19"),
                     C.label(figs[("out", "nearest")], "nearest view only"),
                     C.label(gt_h, "Cycles GT")]
            strip = np.concatenate([cv2.resize(t, (512, 512), interpolation=cv2.INTER_AREA) for t in tiles], 1)
            C.save_rgb(C.RES / f"loo_{h}.png", strip)
            # full-res crops around the organ centre for detail
            ys, xs = np.nonzero(fg)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            y0, x0 = np.clip(cy - 160, 0, 1024 - 320), np.clip(cx - 160, 0, 1024 - 320)
            crops = [t[y0:y0 + 320, x0:x0 + 320] for t in
                     (out_h, figs[("out", "ulr")], figs[("out", "diffuse")], figs[("out", "nearest")], gt_h)]
            C.save_rgb(C.RES / f"loo_{h}_crop.png", np.concatenate(crops, 1))
            err = np.abs(figs[("out", "ulr")] - out_h).mean(-1)
            C.save_rgb(C.RES / f"loo_{h}_ulr_err.png",
                       cv2.applyColorMap((np.clip(err * 4, 0, 1) * 255).astype(np.uint8),
                                         cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255)
    keys = [k for k in rows[0] if k not in ("view",)]
    summary = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
    (C.RES / f"fuse_eval_{VIS_MODE}.json").write_text(json.dumps({"summary_mean": summary, "per_view": rows}, indent=2))
    print("\nMEAN over 20 held-out views:")
    for k, v in summary.items():
        print(f"  {k:36s} {v}")
    print(f"total {time.time()-t0:.1f}s, peak RSS {C.mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
