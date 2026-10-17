"""
align.py — Q1: do the run's meshes, its G-buffers and camera_model agree?

For a few views: z-buffer the full-resolution organ meshes (meshes/s0006/*.obj, world mm)
through camera_model.project, and compare the rasterised organ-id map / depth with the
Blender segid.exr / depth.exr of the same view. Also: unproject depth.exr and measure the
distance to the nearest mesh vertex (the texture_learnability `geometry` check).

    venv/bin/python demo/research_3d/align.py
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

import common as C

VIEWS = ["v01_az+0_el-50", "v02_az+0_el+0", "v07_az+72_el+30", "v16_az+216_el+60"]


def boundary(mask):
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, k)) > 0


def chamfer_px(a, b):
    """Symmetric mean distance (px) between the silhouette boundaries of masks a and b."""
    ba, bb = boundary(a), boundary(b)
    da = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (da[ba].mean() + db[bb].mean())), float(max(np.percentile(da[ba], 99),
                                                                     np.percentile(db[bb], 99)))


def main():
    t0 = time.time()
    V, F, fpid, vpid, names = C.load_all_meshes()
    edge = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    print(f"meshes: {len(names)} organs, {len(V)} verts, {len(F)} tris, "
          f"median edge {np.median(edge):.2f} mm  ({time.time()-t0:.1f}s)")
    tree = cKDTree(V)
    rows = []
    for view in VIEWS:
        t1 = time.time()
        vw = C.View(view, need=("depth", "segid"))
        zb, fid, _ = C.rasterize(V, F, vw.meta)
        pid_r = np.where(fid >= 0, fpid[np.maximum(fid, 0)], 0)
        fg_r, fg_b = pid_r > 0, vw.fg
        iou_fg = (fg_r & fg_b).sum() / (fg_r | fg_b).sum()
        both = fg_r & fg_b
        label_agree = float((pid_r[both] == vw.pid[both]).mean())
        ious = {}
        for pid in np.unique(vw.pid[vw.fg]):
            a, b = pid_r == pid, vw.pid == pid
            if b.sum() > 200:
                ious[int(pid)] = float((a & b).sum() / (a | b).sum())
        areas = {p: int((vw.pid == p).sum()) for p in ious}
        iou_w = sum(ious[p] * areas[p] for p in ious) / sum(areas.values())
        same = both & (pid_r == vw.pid)
        dz_mm = np.abs(zb[same] - vw.Z[same]) * 1000.0
        ch_mean, ch_p99 = chamfer_px(fg_r, fg_b)
        # unprojected depth -> nearest mesh vertex
        P = vw.points_mm()[vw.fg]
        sel = np.random.default_rng(0).choice(len(P), size=min(len(P), 40000), replace=False)
        dvert = tree.query(P[sel])[0]
        # Same check with meta's K taken literally (no lens shift): what goes wrong without camera_model
        u_leg, v_leg = None, None
        row = dict(view=view, fg_px=int(fg_b.sum()), iou_fg=round(float(iou_fg), 4),
                   iou_organ_areaweighted=round(float(iou_w), 4),
                   iou_organ_min=round(min(ious.values()), 4),
                   label_agree=round(label_agree, 4),
                   depth_absdiff_mm_median=round(float(np.median(dz_mm)), 3),
                   depth_absdiff_mm_p90=round(float(np.percentile(dz_mm, 90)), 3),
                   silhouette_chamfer_px=round(ch_mean, 3), silhouette_p99_px=round(ch_p99, 2),
                   unproj_to_vertex_mm_median=round(float(np.median(dvert)), 3),
                   unproj_to_vertex_mm_p90=round(float(np.percentile(dvert, 90)), 3),
                   seconds=round(time.time() - t1, 2))
        rows.append(row)
        print(json.dumps(row))
        # overlay figure: blender segid boundary (green) vs raster boundary (magenta)
        if view == "v02_az+0_el+0":
            out = C.load_rgb(C.OUT_DIR / f"{view}.png")
            ov = out.copy()
            br = np.zeros(fg_b.shape, bool); bb = np.zeros(fg_b.shape, bool)
            for pid in ious:
                br |= boundary(pid_r == pid); bb |= boundary(vw.pid == pid)
            ov[bb] = (0.2, 1.0, 0.3)
            ov[br] = (1.0, 0.2, 1.0)
            C.save_rgb(C.RES / "align_v02_boundaries.png", ov)
            crop = ov[330:650, 380:700]
            C.save_rgb(C.RES / "align_v02_boundaries_crop.png",
                       cv2.resize(crop, (960, 960), interpolation=cv2.INTER_NEAREST))
        # quantisation of the depth buffer (for the record)
        if view == VIEWS[1]:
            z = np.unique(vw.Z[vw.fg])
            dzq = np.diff(z)
            print(f"  depth quantisation: median step {np.median(dzq[dzq>0])*1000:.3f} mm "
                  f"at Z~{np.median(z):.3f} m ({len(z)} distinct values)")
        del vw, zb, fid, pid_r
    (C.RES / "align.json").write_text(json.dumps(rows, indent=2))
    print(f"total {time.time()-t0:.1f}s, peak RSS {C.mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
