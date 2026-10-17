"""
novel.py — what the viewer would show: the organ meshes, rasterised from NEW cameras,
coloured from all 20 neural renders (ULR = view-dependent projective texturing, and the
view-independent diffuse bake). This is the CPU reference for the WebGL shader.

  out/novel_panel.png        novel cameras x {ULR, diffuse}
  out/turntable_ulr.mp4      36-frame orbit at el=+20 deg (512 px)
  out/turntable_sheet.png    12 frames of it
  out/captured_check.json    ULR from all 20 at a CAPTURED camera == that neural render?

    venv/bin/python demo/research_3d/novel.py
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

import common as C
import camera_model as cm
from fuse_eval import blend, lookups

NOVEL = [(36, 15), (108, 45), (180, 0), (252, -25), (324, 75), (0, 90)]


def load_views():
    vs = {}
    for name in C.VIEWS:
        v = C.View(name, need=("depth", "segid", "out"))
        v.out8 = (v.out * 255 + 0.5).astype(np.uint8); del v.out
        vs[name] = v
    return vs


def render(V, F, fpid, VN, vs, m, methods=("ulr", "diffuse")):
    zb, fid, bary = C.rasterize(V, F, m)
    hit = fid >= 0
    Z = np.where(hit, zb, 1.0).astype(np.float64)
    P = cm.unproject(Z, m)[hit].astype(np.float32)
    tri = F[fid[hit]]
    b = bary[hit]
    n = VN[tri[:, 0]] * b[:, 0:1] + VN[tri[:, 1]] * b[:, 1:2] + VN[tri[:, 2]] * b[:, 2:3]
    n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    pid = fpid[fid[hit]]
    C_mm = np.asarray(m["cam_to_world"], np.float64)[:3, 3] * 1000.0
    d_h = C_mm[None, :] - P
    d_h = (d_h / np.linalg.norm(d_h, axis=1, keepdims=True)).astype(np.float32)
    col, vis, dirs = lookups(P, pid, vs, C.VIEWS, "out")
    res, covered = blend(col, vis, dirs, n, d_h)
    out = {}
    for meth in methods:
        c = res[meth].copy()
        for p in np.unique(pid):                      # never seen by any camera -> organ mean
            s = pid == p
            if (s & covered).any():
                c[s & ~covered] = c[s & covered].mean(0)
        img = np.zeros((1024, 1024, 3), np.float32)
        img[hit] = c
        out[meth] = img
    return out, float(covered.mean()), float(np.median(res["_th_min_deg"][covered]))


def main():
    t0 = time.time()
    vs = load_views()
    V, F, fpid, vpid, _ = C.load_all_meshes()
    VN = C.vertex_normals(V, F).astype(np.float32)
    tmpl = vs["v02_az+0_el+0"].meta
    print(f"loaded {time.time()-t0:.1f}s RSS {C.mem_gb():.2f} GB")

    # 1) captured-camera check: ULR over all 20 views at view h must return h's render
    chk = {}
    for h in ("v02_az+0_el+0", "v11_az+144_el+30"):
        imgs, cov, _ = render(V, F, fpid, VN, vs, vs[h].meta, ("ulr",))
        ref = vs[h].out8.astype(np.float32) / 255.0
        chk[h] = dict(psnr_full=round(C.psnr(imgs["ulr"], ref), 2),
                      psnr_fg=round(C.psnr(imgs["ulr"], ref, vs[h].fg), 2), coverage=round(cov, 4))
        print(h, chk[h], flush=True)
    (C.RES / "captured_check.json").write_text(json.dumps(chk, indent=2))

    # 2) novel cameras
    rows, tiles = [], []
    for az, el in NOVEL:
        t1 = time.time()
        m = C.lookat_meta(tmpl, az, el)
        imgs, cov, thmin = render(V, F, fpid, VN, vs, m)
        rows.append(dict(az=az, el=el, coverage=round(cov, 4), nearest_src_angle_deg_median=round(thmin, 1),
                         seconds=round(time.time() - t1, 2)))
        print(json.dumps(rows[-1]), flush=True)
        pair = [C.label(cv2.resize(imgs[k], (512, 512), interpolation=cv2.INTER_AREA),
                        f"az {az} el {el} {k}", 0.6) for k in ("ulr", "diffuse")]
        tiles.append(np.concatenate(pair, 1))
    grid = np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)], 0)
    C.save_rgb(C.RES / "novel_panel.png", grid)
    (C.RES / "novel.json").write_text(json.dumps(rows, indent=2))

    # 3) turntable
    import imageio.v2 as imageio
    frames = []
    for i, az in enumerate(range(0, 360, 10)):
        imgs, _, _ = render(V, F, fpid, VN, vs, C.lookat_meta(tmpl, az, 20), ("ulr",))
        frames.append((np.clip(cv2.resize(imgs["ulr"], (512, 512), interpolation=cv2.INTER_AREA), 0, 1)
                       * 255).astype(np.uint8))
    imageio.mimwrite(C.RES / "turntable_ulr.mp4", frames, fps=12, quality=8, macro_block_size=16)
    sel = frames[::3][:12]
    sheet = np.concatenate([np.concatenate(sel[0:6], 1), np.concatenate(sel[6:12], 1)], 0)
    cv2.imwrite(str(C.RES / "turntable_sheet.png"),
                cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))[..., ::-1])
    print(f"total {time.time()-t0:.1f}s, peak RSS {C.mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
