"""
recon_proto.py — reference implementation of the proposed `recon` pipeline stage
(demo/PLAN_3D.md §4.2). Reads a finished run, writes the viewer assets, measures time/RSS.

Inputs  (all already in the run dir):  web/manifest.json + web/meshes/*.bin (decimated,
         web space = world mm - center_mm), gbuffers/<sid>/<view>/{depth.exr, meta.json},
         output/<view>.png, gt/<view>.png (optional)
Outputs (<out>/recon/):
  cameras.json   per view: OpenCV world->camera in WEB space (mm), fx fy cx cy W H, centre
  bake.bin       u32 nV_total, then per vertex (manifest organ order, concatenated)
                 u8 r,g,b,seen  (seen = 0: no camera saw this vertex; rgb then already holds
                 the fallback = organ hex colour x 0.6)
  recon.json     stats (below)

    venv/bin/python demo/research_3d/recon_proto.py [--run DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "training"))
import camera_model as cm  # noqa: E402

TOL_MM = 3.0


def bilinear(img, u, v):
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, W - 2)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, H - 2)
    fx = np.clip(u - x0, 0, 1)[:, None]; fy = np.clip(v - y0, 0, 1)[:, None]
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def read_web_meshes(run):
    man = json.loads((run / "web" / "manifest.json").read_text())
    center = np.asarray(man["center_mm"], np.float64)
    Vs, Ns, counts = [], [], []
    for o in man["organs"]:
        b = (run / "web" / "meshes" / f"{o['name']}.bin").read_bytes()
        nV, nI = struct.unpack("<II", b[:8])
        Vs.append(np.frombuffer(b, "<f4", 3 * nV, 8).reshape(-1, 3).astype(np.float64))
        Ns.append(np.frombuffer(b, "<f4", 3 * nV, 8 + 12 * nV).reshape(-1, 3).astype(np.float64))
        counts.append(nV)
    return man, center, np.concatenate(Vs), np.concatenate(Ns), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "demo/runs/20260922-165227_s0006"))
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out) / "recon"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sid = json.loads((run / "run.json").read_text())["subject"]
    gb = run / "gbuffers" / sid
    views = sorted(p.name for p in gb.iterdir() if p.is_dir() and p.name.startswith("v"))
    man, center, Vweb, Nweb, counts = read_web_meshes(run)
    Vw = Vweb + center                                   # world mm, for camera_model
    Nn = Nweb / np.maximum(np.linalg.norm(Nweb, axis=1, keepdims=True), 1e-9)
    nV = len(Vw)
    Wsum = np.zeros(nV); WC = np.zeros((nV, 3)); WC2 = np.zeros((nV, 3))
    Wg = np.zeros(nV); WCg = np.zeros((nV, 3)); WCg2 = np.zeros((nV, 3))
    cams = []
    has_gt = (run / "gt").is_dir()
    for i, v in enumerate(views):
        meta = json.loads((gb / v / "meta.json").read_text())
        Z = cm.depth_m(gb / v).astype(np.float32)
        img = cv2.imread(str(run / "output" / f"{v}.png"))[..., ::-1].astype(np.float32) / 255.0
        u, w, z = cm.project(Vw, meta)
        ok = (z > 0) & (u >= 0) & (u <= 1022.999) & (w >= 0) & (w <= 1022.999)
        ui = np.clip(np.rint(u).astype(np.int64), 0, 1023); wi = np.clip(np.rint(w).astype(np.int64), 0, 1023)
        ok &= np.abs(Z[wi, ui] - z) * 1000.0 < TOL_MM
        C_mm = np.asarray(meta["cam_to_world"], np.float64)[:3, 3] * 1000.0
        d = C_mm - Vw; d /= np.linalg.norm(d, axis=1, keepdims=True)
        wt = np.where(ok, np.clip((d * Nn).sum(1), 0, 1) ** 2 + 1e-6, 0.0)
        c = bilinear(img, u, w)
        Wsum += wt; WC += wt[:, None] * c; WC2 += wt[:, None] * c * c
        gpath = run / "gt" / f"{v}.png"
        if has_gt and gpath.exists():
            g = bilinear(cv2.imread(str(gpath))[..., ::-1].astype(np.float32) / 255.0, u, w)
            Wg += wt; WCg += wt[:, None] * g; WCg2 += wt[:, None] * g * g
        # OpenCV world->camera, translated into web space (positions there = world - center)
        c2w = np.asarray(meta["cam_to_world"], np.float64).copy()
        c2w[:3, 1] *= -1; c2w[:3, 2] *= -1
        R = c2w[:3, :3].T
        C_web = C_mm - center
        t = -R @ C_web
        cx, cy = cm.principal_point(1024, 1024)
        cams.append(dict(view=v, image=f"output/{v}.png", az=meta["azimuth_nominal"], el=meta["elevation_nominal"],
                         W=1024, H=1024, fx=meta["K"][0][0], fy=meta["K"][1][1], cx=cx, cy=cy,
                         R=R.round(9).tolist(), t=t.round(6).tolist(), center=C_web.round(4).tolist()))
    seen = Wsum > 1e-5
    col = np.zeros((nV, 3)); col[seen] = WC[seen] / Wsum[seen, None]
    # never-seen vertices carry the fallback colour (organ hex x 0.6) in rgb, so the shader
    # can interpolate vBake.rgb across mixed triangles without darkening toward black
    off = 0
    for o, n in zip(man["organs"], counts):
        hexc = o["color"]
        oc = np.array([int(hexc[k:k + 2], 16) for k in (1, 3, 5)]) / 255.0 * 0.6
        sl = slice(off, off + n)
        col[sl][~seen[sl]] = oc
        off += n
    rgba = np.zeros((nV, 4), np.uint8)
    rgba[:, :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8); rgba[:, 3] = seen * 255
    with open(out / "bake.bin", "wb") as f:
        f.write(struct.pack("<I", nV)); f.write(rgba.tobytes())
    (out / "cameras.json").write_text(json.dumps({"space": "web mm (world - center_mm), OpenCV x-right y-down z-forward",
                                                  "center_mm": center.tolist(), "tol_mm": TOL_MM,
                                                  "cameras": cams}, indent=1))
    # cross-view colour spread (weighted RMS over views, 8-bit units) -- network vs Cycles
    multi = seen & ((WC2.sum(1) > 0))
    var = np.clip(WC2[multi] / Wsum[multi, None] - col[multi] ** 2, 0, None).sum(1)
    spread = float(np.sqrt(var.mean()) * 255)
    stats = dict(views=len(views), vertices=int(nV), organs=len(counts),
                 vertices_seen_frac=round(float(seen.mean()), 4), spread_net_rms8=round(spread, 2))
    if Wg.any():
        sg = Wg > 1e-5
        cg = WCg[sg] / Wg[sg, None]
        varg = np.clip(WCg2[sg] / Wg[sg, None] - cg ** 2, 0, None).sum(1)
        stats["spread_cycles_rms8"] = round(float(np.sqrt(varg.mean()) * 255), 2)
    import resource
    stats["seconds"] = round(time.time() - t0, 2)
    stats["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    stats["bake_bin_bytes"] = (out / "bake.bin").stat().st_size
    stats["cameras_json_bytes"] = (out / "cameras.json").stat().st_size
    stats["web_meshes_bytes"] = sum(p.stat().st_size for p in (run / "web" / "meshes").glob("*.bin"))
    stats["output_png_bytes"] = sum(p.stat().st_size for p in (run / "output").glob("v*.png"))
    (out / "recon.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
