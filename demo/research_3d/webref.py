"""
webref.py — CPU reference images for the WebGL recon viewer's self-test (PLAN_3D.md §4.5, acceptance R3).

Uses exactly what the browser will have: the decimated web meshes (web space = world mm -
center_mm), recon/bake.bin (from recon_proto.py) and recon/cameras.json. Renders a captured
camera and two novel cameras in 'ulr' and 'baked' mode and writes, under out/webref/:

  cameras.json   the 3 test cameras in the SAME schema as recon/cameras.json entries
  <name>_<mode>.png   1024^2 reference renders
  webref.json    PSNR of each vs the captured neural render (captured camera only)

    venv/bin/python demo/research_3d/recon_proto.py && venv/bin/python demo/research_3d/webref.py
"""
from __future__ import annotations

import json
import struct

import numpy as np

import common as C
import camera_model as cm
from bake_eval import load_web_meshes
from fuse_eval import blend, lookups

OUT = C.RES / "webref"


def organ_rgb():
    man = json.loads((C.RUN / "web" / "manifest.json").read_text())
    ids = C.tissue_ids()
    return {ids[o["name"]]: np.array([int(o["color"][i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
            for o in man["organs"]}


def web_camera(m, center, name):
    c2w = np.asarray(m["cam_to_world"], np.float64).copy()
    c2w[:3, 1] *= -1; c2w[:3, 2] *= -1
    R = c2w[:3, :3].T
    Cw = c2w[:3, 3] * 1000.0 - center
    cx, cy = cm.principal_point(1024, 1024)
    return dict(view=name, W=1024, H=1024, fx=m["K"][0][0], fy=m["K"][1][1], cx=cx, cy=cy,
                R=R.round(9).tolist(), t=(-R @ Cw).round(6).tolist(), center=Cw.round(4).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    vs = {}
    for name in C.VIEWS:
        v = C.View(name, need=("depth", "segid", "out"))
        v.out8 = (v.out * 255 + 0.5).astype(np.uint8); del v.out
        vs[name] = v
    man = json.loads((C.RUN / "web" / "manifest.json").read_text())
    center = np.asarray(man["center_mm"], np.float64)
    V, F, vpid = load_web_meshes()                       # world mm
    fpid = vpid[F[:, 0]]
    VN = C.vertex_normals(V, F).astype(np.float32)
    b = (C.RES / "recon" / "bake.bin").read_bytes()
    nV = struct.unpack_from("<I", b)[0]
    rgba = np.frombuffer(b, np.uint8, 4 * nV, 4).reshape(-1, 4)
    assert nV == len(V)
    vcol = rgba[:, :3].astype(np.float32) / 255.0      # unseen verts already hold organ x 0.6
    tmpl = vs["v02_az+0_el+0"].meta
    tests = [("v02_az+0_el+0", tmpl), ("novel_az36_el15", C.lookat_meta(tmpl, 36, 15)),
             ("novel_az180_el0", C.lookat_meta(tmpl, 180, 0))]
    cams, scores = [], {}
    for name, m in tests:
        cams.append(web_camera(m, center, name))
        zb, fid, bary = C.rasterize(V, F, m)
        hit = fid >= 0
        tri = F[fid[hit]]; w = bary[hit]
        # baked
        img_b = np.zeros((1024, 1024, 3), np.float32)
        img_b[hit] = vcol[tri[:, 0]] * w[:, 0:1] + vcol[tri[:, 1]] * w[:, 1:2] + vcol[tri[:, 2]] * w[:, 2:3]
        # ulr
        P = cm.unproject(np.where(hit, zb, 1.0).astype(np.float64), m)[hit].astype(np.float32)
        n = VN[tri[:, 0]] * w[:, 0:1] + VN[tri[:, 1]] * w[:, 1:2] + VN[tri[:, 2]] * w[:, 2:3]
        n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        Cmm = np.asarray(m["cam_to_world"], np.float64)[:3, 3] * 1000.0
        d = Cmm - P; d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)
        col, vis, dirs = lookups(P, fpid[fid[hit]], vs, C.VIEWS, "out")
        res, cov = blend(col, vis, dirs, n, d)
        c = res["ulr"].copy()
        bk = img_b[hit]
        c[~cov] = bk[~cov]                               # holes -> baked / organ colour
        img_u = np.zeros_like(img_b); img_u[hit] = c
        C.save_rgb(OUT / f"{name}_ulr.png", img_u)
        C.save_rgb(OUT / f"{name}_baked.png", img_b)
        if name in vs:
            ref = vs[name].out8.astype(np.float32) / 255.0
            fg = vs[name].fg
            scores[name] = {k: dict(psnr_full=round(C.psnr(im, ref), 2), psnr_fg=round(C.psnr(im, ref, fg), 2))
                            for k, im in (("ulr", img_u), ("baked", img_b))}
        print(name, scores.get(name, ""), f"coverage {cov.mean():.4f}", flush=True)
    (OUT / "cameras.json").write_text(json.dumps({"cameras": cams}, indent=1))
    (OUT / "webref.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
