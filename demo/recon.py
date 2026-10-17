"""
recon.py — the `recon` pipeline stage (demo/PLAN_3D.md §4.2): builds the assets for the
in-browser "3-D view" (mesh + view-dependent projective texturing of the 20 neural renders).

This is a direct port of `demo/research_3d/recon_proto.py` — the maths is UNCHANGED (same
accumulation order, same weights, same rounding), so `bake.bin` reproduces the reference
byte-for-byte. Additions over the prototype: an `emit` callback for pipeline progress/stat/
artifact events, `organ_center_web`/`near_mm`/`far_mm` in cameras.json, an area-weighted
seen fraction from the web-mesh faces, `viewer.json`, and a `--check` CLI.

Inputs  (all already in the run dir): web/manifest.json + web/meshes/*.bin (decimated web
         meshes, positions centred on manifest center_mm), gbuffers/<sid>/<view>/{depth.exr,
         meta.json}, output/<view>.png, gt/<view>.png (optional)
Outputs (<run>/recon/): bake.bin, cameras.json, viewer.json, recon.json — see PLAN_3D.md §4.2.

    venv/bin/python demo/recon.py [<run_dir>]              # build
    venv/bin/python demo/recon.py <run_dir> --check        # verify cameras.json reprojection
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
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts" / "training"))
import camera_model as cm  # noqa: E402

TOL_MM = 3.0
NEAR_MM = 500.0
FAR_MM = 3000.0


def bilinear(img, u, v):
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, W - 2)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, H - 2)
    fx = np.clip(u - x0, 0, 1)[:, None]; fy = np.clip(v - y0, 0, 1)[:, None]
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def infill_unseen(col: np.ndarray, seen: np.ndarray, Faces: np.ndarray, max_passes: int = 50):
    """Geodesic colour infill: propagate colour to never-seen vertices from their
    (already-filled) neighbours over the mesh edge graph, BFS-style, via repeated sparse
    adjacency multiplies. `col[~seen]` must already hold a fallback colour (organ hex x
    0.6) on entry — a vertex whose whole connected component has no seen vertex at all
    keeps that fallback untouched. Never touches `col[seen]`. Organs never share an edge
    (each triangle is wholly inside one organ), so running this once over the global,
    concatenated mesh does not leak colour across organs.

    Returns (col, filled) — `filled` is `seen` plus every vertex this reached; the `a`
    (seen) channel in bake.bin is NOT this — it stays exactly what it was (whether a
    vertex was directly photographed by a camera), so downstream code never loses that
    distinction even though its rgb may now be inferred rather than flat."""
    import scipy.sparse as sp
    nV = len(col)
    e0 = np.concatenate([Faces[:, 0], Faces[:, 1], Faces[:, 2]])
    e1 = np.concatenate([Faces[:, 1], Faces[:, 2], Faces[:, 0]])
    src = np.concatenate([e0, e1]); dst = np.concatenate([e1, e0])
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(nV, nV)).tocsr()
    A.data[:] = 1.0  # unweighted: one edge counts once even if shared by two triangles
    filled = seen.copy()
    out = col.copy()
    for _ in range(max_passes):
        unfilled = ~filled
        if not unfilled.any():
            break
        numer = A @ (out * filled[:, None])
        denom = np.asarray(A @ filled.astype(np.float64))
        newly = unfilled & (denom > 1e-9)
        if not newly.any():
            break
        out[newly] = numer[newly] / denom[newly, None]
        filled[newly] = True
    return out, filled


def read_web_meshes(run: Path):
    """-> manifest, center_mm, V (N,3 web mm), N (N,3 normals), counts (per organ),
    F (n_tri, 3 global vertex indices) — same order recon_proto.py used (manifest order)."""
    man = json.loads((run / "web" / "manifest.json").read_text())
    center = np.asarray(man["center_mm"], np.float64)
    Vs, Ns, Fs, counts = [], [], [], []
    off = 0
    for o in man["organs"]:
        b = (run / "web" / "meshes" / f"{o['name']}.bin").read_bytes()
        nV, nI = struct.unpack("<II", b[:8])
        Vs.append(np.frombuffer(b, "<f4", 3 * nV, 8).reshape(-1, 3).astype(np.float64))
        Ns.append(np.frombuffer(b, "<f4", 3 * nV, 8 + 12 * nV).reshape(-1, 3).astype(np.float64))
        idx = np.frombuffer(b, "<u4", nI, 8 + 24 * nV).reshape(-1, 3).astype(np.int64)
        Fs.append(idx + off)
        counts.append(nV)
        off += nV
    return man, center, np.concatenate(Vs), np.concatenate(Ns), counts, np.concatenate(Fs)


def build(run_dir: Path, sid: str, emit=None) -> dict:
    """Read a finished run's web meshes + G-buffers + neural renders, bake per-vertex
    colours (view-dependent-projective-texturing input), write <run>/recon/*.

    `emit(kind, **kw)` is called with kind in {"progress", "stat", "artifact", "log"} —
    the pipeline stage wraps it around Run.progress/stat/artifact/log."""
    def _emit(kind, **kw):
        if emit is not None:
            emit(kind, **kw)

    t0 = time.time()
    run = Path(run_dir)
    web_dir = run / "web" / "meshes"
    if not (run / "web" / "manifest.json").exists() or not web_dir.is_dir():
        raise RuntimeError("web meshes missing — mesh stage did not run")
    out = run / "recon"
    out.mkdir(parents=True, exist_ok=True)

    gb = run / "gbuffers" / sid
    views = sorted(p.name for p in gb.iterdir() if p.is_dir() and p.name.startswith("v"))
    man, center, Vweb, Nweb, counts, Faces = read_web_meshes(run)
    Vw = Vweb + center                                   # world mm, for camera_model
    Nn = Nweb / np.maximum(np.linalg.norm(Nweb, axis=1, keepdims=True), 1e-9)
    nV = len(Vw)
    Wsum = np.zeros(nV); WC = np.zeros((nV, 3)); WC2 = np.zeros((nV, 3))
    Wg = np.zeros(nV); WCg = np.zeros((nV, 3)); WCg2 = np.zeros((nV, 3))
    cams = []
    has_gt = (run / "gt").is_dir()
    organ_center_m = None
    for i, v in enumerate(views):
        meta = json.loads((gb / v / "meta.json").read_text())
        if organ_center_m is None:
            organ_center_m = np.asarray(meta["organ_center"], np.float64)
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
        cams.append(dict(view=v, image=f"/runs/{run.name}/output/{v}.png",
                         az=meta["azimuth_nominal"], el=meta["elevation_nominal"],
                         W=1024, H=1024, fx=meta["K"][0][0], fy=meta["K"][1][1], cx=cx, cy=cy,
                         R=R.round(9).tolist(), t=t.round(6).tolist(), center=C_web.round(4).tolist()))
        _emit("progress", done=i + 1, total=len(views), label=v)

    seen = Wsum > 1e-5
    col = np.zeros((nV, 3)); col[seen] = WC[seen] / Wsum[seen, None]
    # never-seen vertices start at the flat fallback colour (organ hex x 0.6) ...
    off = 0
    for o, n in zip(man["organs"], counts):
        hexc = o["color"]
        oc = np.array([int(hexc[k:k + 2], 16) for k in (1, 3, 5)]) / 255.0 * 0.6
        sl = slice(off, off + n)
        col[sl][~seen[sl]] = oc
        off += n
    # ... then geodesic infill overwrites every one reachable from a seen vertex over the
    # mesh edges with the mean of its filled neighbours (flat colour only remains where a
    # whole connected patch of the surface was never seen by any of the 20 cameras). `a`
    # (seen) is computed from `seen` alone below and is unaffected by this — it still marks
    # "photographed by a camera" vs "inferred/fallback", never losing that distinction.
    col, _ = infill_unseen(col, seen, Faces)
    rgba = np.zeros((nV, 4), np.uint8)
    rgba[:, :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8); rgba[:, 3] = seen * 255
    with open(out / "bake.bin", "wb") as f:
        f.write(struct.pack("<I", nV)); f.write(rgba.tobytes())

    # area-weighted seen fraction: a triangle counts as seen when all 3 of its vertices are
    # (matches what the shader will show as "already textured" rather than fallback colour)
    tri_seen = seen[Faces].all(axis=1)
    p0, p1, p2 = Vw[Faces[:, 0]], Vw[Faces[:, 1]], Vw[Faces[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    seen_area_frac = float(tri_area[tri_seen].sum() / max(float(tri_area.sum()), 1e-9))

    organ_center_web = (organ_center_m * 1000.0 - center).round(4).tolist()
    cameras_doc = {
        "space": "web mm (world - center_mm), OpenCV x-right y-down z-forward",
        "center_mm": center.tolist(), "tol_mm": TOL_MM,
        "organ_center_web": organ_center_web, "near_mm": NEAR_MM, "far_mm": FAR_MM,
        "cameras": cams,
    }
    (out / "cameras.json").write_text(json.dumps(cameras_doc, indent=1))

    # cross-view colour spread (weighted RMS over views, 8-bit units) -- network vs Cycles
    multi = seen & (WC2.sum(1) > 0)
    var = np.clip(WC2[multi] / Wsum[multi, None] - col[multi] ** 2, 0, None).sum(1)
    spread = float(np.sqrt(var.mean()) * 255)
    stats = dict(views=len(views), vertices=int(nV), organs=len(counts),
                 vertices_seen_frac=round(float(seen.mean()), 4),
                 seen_frac_area_weighted=round(seen_area_frac, 4),
                 spread_net_rms8=round(spread, 2))
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
    stats["web_meshes_bytes"] = sum(p.stat().st_size for p in web_dir.glob("*.bin"))
    stats["output_png_bytes"] = sum(p.stat().st_size for p in (run / "output").glob("v*.png"))
    (out / "recon.json").write_text(json.dumps(stats, indent=2))

    viewer_doc = {
        "manifest": f"/runs/{run.name}/web/manifest.json",
        "cameras": f"/runs/{run.name}/recon/cameras.json",
        "bake": f"/runs/{run.name}/recon/bake.bin",
        "n_views": len(views),
        "stats": stats,
    }
    (out / "viewer.json").write_text(json.dumps(viewer_doc, indent=2))

    _emit("stat", key="views", label="views", value=len(views), unit="", fmt="int")
    _emit("stat", key="verts", label="vertices", value=int(nV), unit="", fmt="int")
    _emit("stat", key="seen_pct", label="surface seen", value=round(seen_area_frac * 100, 1), unit="%", fmt="float1")
    _emit("stat", key="spread_net", label="cross-view spread · network", value=stats["spread_net_rms8"], unit="RMS/255", fmt="float1")
    if "spread_cycles_rms8" in stats:
        _emit("stat", key="spread_cycles", label="cross-view spread · Cycles", value=stats["spread_cycles_rms8"], unit="RMS/255", fmt="float1")
    _emit("stat", key="build_s", label="build time", value=stats["seconds"], unit="s", fmt="float2")

    _emit("artifact", key="viewer", url="recon/viewer.json")
    _emit("artifact", key="bake", url="recon/bake.bin")
    _emit("artifact", key="cameras", url="recon/cameras.json", meta={"n": len(views)})
    _emit("log", line=f"bake: {nV} vertices, {seen_area_frac * 100:.1f} % seen by ≥ 1 camera")

    return stats


def check(run_dir: Path) -> float:
    """Re-project every web vertex with recon/cameras.json's (R, t, K) and compare with
    camera_model.project(world_mm, meta) + 0.5 (array coords -> COLMAP pixel-centre
    convention). Returns the max deviation in pixels over all cameras/vertices."""
    run = Path(run_dir)
    cam_doc = json.loads((run / "recon" / "cameras.json").read_text())
    center = np.asarray(cam_doc["center_mm"], np.float64)
    sid = json.loads((run / "run.json").read_text())["subject"]
    gb = run / "gbuffers" / sid
    _, _, Vweb, _, _, _ = read_web_meshes(run)
    Vw = Vweb + center
    max_dev = 0.0
    for c in cam_doc["cameras"]:
        meta = json.loads((gb / c["view"] / "meta.json").read_text())
        u_ref, v_ref, _ = cm.project(Vw, meta)
        u_ref = u_ref + 0.5; v_ref = v_ref + 0.5
        R = np.asarray(c["R"], np.float64); t = np.asarray(c["t"], np.float64)
        Xc = Vweb @ R.T + t
        u = c["fx"] * Xc[:, 0] / Xc[:, 2] + c["cx"]
        v = c["fy"] * Xc[:, 1] / Xc[:, 2] + c["cy"]
        dev = np.sqrt((u - u_ref) ** 2 + (v - v_ref) ** 2)
        dev = dev[np.isfinite(dev)]
        if len(dev):
            max_dev = max(max_dev, float(dev.max()))
    return max_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=str(REPO / "demo/runs/20260922-165227_s0006"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    run_dir = Path(a.run)
    sid = json.loads((run_dir / "run.json").read_text())["subject"]
    if a.check:
        dev = check(run_dir)
        print(json.dumps({"max_dev_px": dev}))
    else:
        stats = build(run_dir, sid)
        print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
