"""
bake_eval.py — what the concrete 3-D representations lose, leave-one-out.

fuse_eval.py measured per-PIXEL fusion (infinite texture resolution). Real representations
store colour at a finite density:

  (a) point cloud      unproject every fg pixel of the source views, colour = neural render,
                       voxel-downsample (s mm, mean colour); render = z-buffered square
                       splats (what WebGL gl.POINTS does).
  (b) per-vertex bake  colour every mesh vertex from the source views (depth test +
                       organ-id test, cos^2 weights); render = rasterise the mesh with
                       barycentric colour interpolation (what the mesh viewer would do).
                       Meshes: the web meshes (decimated, <= 15k tris/organ), the full OBJs
                       (1.5 mm edges) and the OBJs with one midpoint subdivision (0.75 mm).

Also: how much of the organ surface the 20 cameras see at all (area-weighted).

    venv/bin/python demo/research_3d/bake_eval.py
"""
from __future__ import annotations

import json
import struct
import time

import cv2
import numpy as np

import common as C
import camera_model as cm

TOL_MM = 3.0
HELD = ("v02_az+0_el+0", "v07_az+72_el+30", "v13_az+216_el-50", "v18_az+288_el+0")


def load_views():
    vs = {}
    for name in C.VIEWS:
        v = C.View(name, need=("depth", "segid", "out"))
        v.out8 = (v.out * 255 + 0.5).astype(np.uint8); del v.out
        vs[name] = v
    return vs


def load_web_meshes():
    man = json.loads((C.RUN / "web" / "manifest.json").read_text())
    center = np.asarray(man["center_mm"], np.float64)
    ids = C.tissue_ids()
    Vs, Fs, vp = [], [], []
    off = 0
    for o in man["organs"]:
        b = (C.RUN / "web" / "meshes" / f"{o['name']}.bin").read_bytes()
        nV, nI = struct.unpack("<II", b[:8])
        pos = np.frombuffer(b, "<f4", 3 * nV, 8).reshape(-1, 3).astype(np.float64) + center
        idx = np.frombuffer(b, "<u4", nI, 8 + 24 * nV).reshape(-1, 3).astype(np.int64)
        Vs.append(pos); Fs.append(idx + off); vp.append(np.full(nV, ids[o["name"]], np.int16))
        off += nV
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(vp)


def subdivide(V, F, vpid):
    """One midpoint (1->4) subdivision, no smoothing: vertices stay on the original surface."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    mid = len(V) + np.arange(len(ue))
    V2 = np.concatenate([V, 0.5 * (V[ue[:, 0]] + V[ue[:, 1]])])
    p2 = np.concatenate([vpid, vpid[ue[:, 0]]])
    nF = len(F)
    m01, m12, m20 = mid[inv[:nF]], mid[inv[nF:2 * nF]], mid[inv[2 * nF:]]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    F2 = np.concatenate([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1),
                         np.stack([m20, m12, c], 1), np.stack([m01, m12, m20], 1)])
    return V2, F2, p2


def vertex_contrib(V, Nrm, vpid, v):
    """Per-vertex (weight, weight*colour) from one view; weight 0 where not visible."""
    u, w, Z = cm.project(V, v.meta)
    ok = (Z > 0) & (u >= 0) & (u <= 1022.999) & (w >= 0) & (w <= 1022.999)
    ui = np.clip(np.rint(u).astype(np.int64), 0, 1023)
    wi = np.clip(np.rint(w).astype(np.int64), 0, 1023)
    ok &= np.abs(v.Z[wi, ui] - Z) * 1000.0 < TOL_MM
    ok &= v.pid[wi, ui] == vpid
    d = v.C_mm[None, :] - V
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cosn = np.clip((d * Nrm).sum(1), 0, 1)
    wt = np.where(ok, cosn ** 2 + 1e-6, 0.0).astype(np.float32)
    col = C.bilinear(v.out8, u, w).astype(np.float32) / 255.0
    return wt, wt[:, None] * col


def render_vertex_colours(V, F, vcol, m):
    zb, fid, bary = C.rasterize(V, F, m)
    img = np.zeros((1024, 1024, 3), np.float32)
    hit = fid >= 0
    tri = F[fid[hit]]
    b = bary[hit]
    img[hit] = (vcol[tri[:, 0]] * b[:, 0:1] + vcol[tri[:, 1]] * b[:, 1:2] + vcol[tri[:, 2]] * b[:, 2:3])
    return img, hit


def eval_bake(tag, V, F, vpid, vs, rows, figs):
    t0 = time.time()
    Nrm = C.vertex_normals(V, F)
    Wsum = np.zeros(len(V), np.float32)
    WCsum = np.zeros((len(V), 3), np.float32)
    nvis = np.zeros(len(V), np.int16)
    for name in C.VIEWS:
        wt, wc = vertex_contrib(V, Nrm, vpid, vs[name])
        Wsum += wt; WCsum += wc; nvis += (wt > 0)
    t_bake = time.time() - t0
    # surface seen by >= 1 camera (area-weighted over triangles with all 3 verts seen)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    seen_tri = (nvis[F] > 0).all(1)
    info = dict(mesh=tag, verts=int(len(V)), tris=int(len(F)),
                median_edge_mm=round(float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))), 2),
                surface_seen_frac=round(float(area[seen_tri].sum() / area.sum()), 4),
                bake_seconds_20views=round(t_bake, 2))
    ps_fg, ps_full = [], []
    for h in HELD:
        vh = vs[h]
        wt, wc = vertex_contrib(V, Nrm, vpid, vh)
        W = Wsum - wt; WC = WCsum - wc
        seen = W > 1e-5
        vcol = np.zeros((len(V), 3), np.float32)
        vcol[seen] = WC[seen] / W[seen, None]
        for p in np.unique(vpid):                       # never-seen verts -> organ mean
            s = (vpid == p)
            if (s & seen).any():
                vcol[s & ~seen] = vcol[s & seen].mean(0)
        img, hit = render_vertex_colours(V, F, vcol, vh.meta)
        ref = vh.out8.astype(np.float32) / 255.0
        # compare on pixels whose nearest vertex set was seen by another view (like fuse_eval)
        ps_fg.append(C.psnr(img, ref, vh.fg & hit))
        ps_full.append(C.psnr(img, ref))
        if h == HELD[0]:
            figs.setdefault(h, {})[tag] = img
    info["loo_psnr_fg"] = round(float(np.mean(ps_fg)), 2)
    info["loo_psnr_full"] = round(float(np.mean(ps_full)), 2)
    info["seconds"] = round(time.time() - t0, 1)
    rows.append(info)
    print(json.dumps(info), flush=True)
    return Wsum, WCsum


def splat_impl(u, v, z, col, r, W, H):
    zbuf = np.full((H, W), np.inf, np.float32)
    img = np.zeros((H, W, 3), np.float32)
    for i in range(u.shape[0]):
        if z[i] <= 0:
            continue
        rr = r[i]
        x0 = int(np.floor(u[i] - rr + 0.5)); x1 = int(np.floor(u[i] + rr + 0.5))
        y0 = int(np.floor(v[i] - rr + 0.5)); y1 = int(np.floor(v[i] + rr + 0.5))
        for y in range(max(y0, 0), min(y1, H - 1) + 1):
            for x in range(max(x0, 0), min(x1, W - 1) + 1):
                if z[i] < zbuf[y, x]:
                    zbuf[y, x] = z[i]
                    img[y, x, 0] = col[i, 0]; img[y, x, 1] = col[i, 1]; img[y, x, 2] = col[i, 2]
    return img, zbuf


splat = C.njit(cache=True)(splat_impl) if C.njit else splat_impl


def eval_points(vs, rows, figs, voxels=(0.5, 1.0, 2.0)):
    for s_mm in voxels:
        t0 = time.time()
        ps_fg, ps_full, npts = [], [], []
        for h in HELD:
            P_all, c_all = [], []
            for name in C.VIEWS:
                if name == h:
                    continue
                v = vs[name]
                fgm = v.fg & (cv2.erode(v.fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                P_all.append(v.points_mm()[fgm]); c_all.append(v.out8[fgm])
            P = np.concatenate(P_all); col = np.concatenate(c_all).astype(np.float32) / 255.0
            del P_all, c_all
            key = np.floor(P / s_mm).astype(np.int64)
            key = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
            uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            Pv = np.zeros((len(uk), 3)); Cv = np.zeros((len(uk), 3))
            np.add.at(Pv, inv, P); np.add.at(Cv, inv, col)
            Pv /= cnt[:, None]; Cv /= cnt[:, None]
            npts.append(len(uk))
            vh = vs[h]
            u, w, Z = cm.project(Pv, vh.meta)
            fx = float(vh.meta["K"][0][0])
            r = 0.5 * s_mm / 1000.0 * fx / np.maximum(Z, 1e-6) * 1.25   # square half-size, px
            img, _ = splat(u, w, Z, Cv.astype(np.float32), r, 1024, 1024)
            ref = vh.out8.astype(np.float32) / 255.0
            ps_fg.append(C.psnr(img, ref, vh.fg))
            ps_full.append(C.psnr(img, ref))
            if h == HELD[0]:
                figs.setdefault(h, {})[f"points_{s_mm}mm"] = img
            del P, col, key, inv
        info = dict(mesh=f"points voxel {s_mm} mm", points_from_19_views=int(np.mean(npts)),
                    bytes_at_9B_per_point=int(np.mean(npts) * 9),
                    loo_psnr_fg=round(float(np.mean(ps_fg)), 2),
                    loo_psnr_full=round(float(np.mean(ps_full)), 2),
                    seconds=round(time.time() - t0, 1))
        rows.append(info)
        print(json.dumps(info), flush=True)


def main():
    t0 = time.time()
    vs = load_views()
    print(f"views loaded {time.time()-t0:.1f}s RSS {C.mem_gb():.2f} GB")
    rows, figs = [], {}
    Vw, Fw, pw = load_web_meshes()
    eval_bake("web (decimated)", Vw, Fw, pw, vs, rows, figs)
    V, F, fp, vp, _ = C.load_all_meshes()
    eval_bake("obj full-res", V, F, vp, vs, rows, figs)
    V2, F2, p2 = subdivide(V, F, vp)
    del V, F
    eval_bake("obj subdiv x1", V2, F2, p2, vs, rows, figs)
    del V2, F2, p2
    print(f"RSS {C.mem_gb():.2f} GB")
    eval_points(vs, rows, figs)
    (C.RES / "bake_eval.json").write_text(json.dumps(rows, indent=2))
    # figure: held-out v02 crop, each representation
    h = HELD[0]
    ref = vs[h].out8.astype(np.float32) / 255.0
    ys, xs = np.nonzero(vs[h].fg)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = np.clip(cy - 160, 0, 704), np.clip(cx - 160, 0, 704)
    tiles = [C.label(ref[y0:y0+320, x0:x0+320], "neural (held out)", 0.55)]
    for k, im in figs[h].items():
        tiles.append(C.label(im[y0:y0+320, x0:x0+320], k, 0.55))
    C.save_rgb(C.RES / f"bake_{h}_crop.png", np.concatenate(tiles, 1))
    print(f"total {time.time()-t0:.1f}s, peak RSS {C.mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
