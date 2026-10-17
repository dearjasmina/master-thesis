"""
export_colmap.py — write a run's 20 neural renders + EXACT cameras as a COLMAP binary model,
for 3D Gaussian Splatting with gsplat (examples/simple_trainer.py) or Brush.

    <out>/images/<view>.png          the neural renders (1024^2, black background)
    <out>/sparse/0/cameras.bin       one PINHOLE camera per view: fx fy cx cy
                                     (cx, cy = camera_model.principal_point = the lens shift!)
    <out>/sparse/0/images.bin        world->camera (OpenCV: x right, y down, z forward), metres
    <out>/sparse/0/points3D.bin      initial points: unprojected depth.exr of all views,
                                     voxel-downsampled, coloured by the renders (empty tracks)

COLMAP pixel convention: the centre of the top-left pixel is (0.5, 0.5). camera_model's
array coordinates have that centre at 0, i.e. u_colmap = u_array + 0.5 = cx + fx*x/z, so
cx/cy are exported unchanged.

WARNING: the Inria gaussian-splatting repo (./gaussian-splatting) IGNORES cx, cy (symmetric
frustum). Training it on this export would misplace every pixel by (51.2, 20.5) px. Use gsplat
or Brush, which honour the principal point.

    venv/bin/python demo/research_3d/export_colmap.py [--run demo/runs/<id>] [--out DIR] [--voxel-mm 2]
    venv/bin/python demo/research_3d/export_colmap.py --verify DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "training"))
import camera_model as cm  # noqa: E402


def rotmat_to_qvec(R):
    """COLMAP's rotmat2qvec (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                  [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def w2c_opencv(meta):
    """Blender cam_to_world (-Z forward, +Y up) -> OpenCV world_to_camera (R, t), metres."""
    c2w = np.asarray(meta["cam_to_world"], np.float64).copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    return R, t


def export(run: Path, out: Path, voxel_mm: float):
    sid = json.loads((run / "run.json").read_text())["subject"]
    gb = run / "gbuffers" / sid
    views = sorted(p.name for p in gb.iterdir() if p.is_dir() and p.name.startswith("v"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    sp = out / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    cams, imgs, P_all, C_all = [], [], [], []
    import cv2
    for i, v in enumerate(views, 1):
        meta = json.loads((gb / v / "meta.json").read_text())
        W = H = int(meta["resolution"])
        fx = float(meta["K"][0][0]); fy = float(meta["K"][1][1])
        cx, cy = cm.principal_point(W, H)
        cams.append((i, W, H, fx, fy, cx, cy))
        R, t = w2c_opencv(meta)
        imgs.append((i, rotmat_to_qvec(R), t, i, f"{v}.png"))
        shutil.copy2(run / "output" / f"{v}.png", out / "images" / f"{v}.png")
        Z = cm.depth_m(gb / v)
        fg = cm.segid_pid(gb / v) > 0
        fg &= cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # no silhouette px
        P_all.append(cm.unproject(Z, meta)[fg] / 1000.0)                      # metres
        rgb = cv2.imread(str(run / "output" / f"{v}.png"))[..., ::-1]
        C_all.append(rgb[fg])
    P = np.concatenate(P_all); col = np.concatenate(C_all).astype(np.float64)
    key = np.floor(P * 1000.0 / voxel_mm).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Pv = np.zeros((len(cnt), 3)); Cv = np.zeros((len(cnt), 3))
    np.add.at(Pv, inv, P); np.add.at(Cv, inv, col)
    Pv /= cnt[:, None]; Cv = np.clip(Cv / cnt[:, None] + 0.5, 0, 255).astype(np.uint8)

    with open(sp / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, W, H, fx, fy, cx, cy in cams:
            f.write(struct.pack("<iiQQ", cid, 1, W, H))            # model 1 = PINHOLE
            f.write(struct.pack("<4d", fx, fy, cx, cy))
    with open(sp / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, q, t, cid, name in imgs:
            f.write(struct.pack("<i4d3di", iid, *q, *t, cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))                         # no 2-D points
    with open(sp / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(Pv)))
        for j in range(len(Pv)):
            f.write(struct.pack("<Q3d3BdQ", j + 1, *Pv[j], *Cv[j], 0.0, 0))
    info = dict(subject=sid, views=len(views), points=int(len(Pv)), voxel_mm=voxel_mm,
                principal_point=list(cm.principal_point(1024, 1024)), units="metres")
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


def verify(out: Path):
    """Independent check: read the binaries back, project the points with the plain
    COLMAP pinhole model and compare with the renders' non-black pixels."""
    import cv2
    sp = out / "sparse" / "0"
    b = (sp / "cameras.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]; o = 8; cams = {}
    for _ in range(n):
        cid, model, W, H = struct.unpack_from("<iiQQ", b, o); o += 24
        cams[cid] = (W, H, *struct.unpack_from("<4d", b, o)); o += 32
    b = (sp / "images.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]; o = 8; ims = []
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<i4d3di", b, o); o += 64
        e = b.index(b"\x00", o); name = b[o:e].decode(); o = e + 1
        n2 = struct.unpack_from("<Q", b, o)[0]; o += 8 + 24 * n2
        ims.append((name, qvec_to_rotmat((qw, qx, qy, qz)), np.array([tx, ty, tz]), cams[cid]))
    b = (sp / "points3D.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]; o = 8; P = np.zeros((n, 3))
    for j in range(n):
        P[j] = struct.unpack_from("<3d", b, o + 8); o += 8 + 24 + 3 + 8
        tl = struct.unpack_from("<Q", b, o)[0]; o += 8 + 8 * tl
    for name, R, t, (W, H, fx, fy, cx, cy) in ims[:20:6]:
        Xc = P @ R.T + t
        u = fx * Xc[:, 0] / Xc[:, 2] + cx; v = fy * Xc[:, 1] / Xc[:, 2] + cy     # COLMAP coords
        img = cv2.imread(str(out / "images" / name))
        fg = img.max(-1) > 8
        ui = np.clip(np.floor(u).astype(int), 0, W - 1); vi = np.clip(np.floor(v).astype(int), 0, H - 1)
        hit = fg[vi, ui].mean()
        # the same with the principal point at the image centre (what Inria 3DGS assumes)
        u2 = fx * Xc[:, 0] / Xc[:, 2] + W / 2; v2 = fy * Xc[:, 1] / Xc[:, 2] + H / 2
        hit2 = fg[np.clip(np.floor(v2).astype(int), 0, H - 1), np.clip(np.floor(u2).astype(int), 0, W - 1)].mean()
        # coverage the other way: fraction of fg pixels hit by >= 1 projected point
        m = np.zeros_like(fg); m[vi, ui] = True
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        print(f"{name}: points landing on non-black px {hit:.4f} (centred pp: {hit2:.4f}); "
              f"fg px within 1 px of a point {(m & fg).sum() / fg.sum():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "demo/runs/20260922-165227_s0006"))
    ap.add_argument("--out", default=str(HERE / "out" / "gs_s0006"))
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--verify", default=None)
    a = ap.parse_args()
    if a.verify:
        verify(Path(a.verify))
    else:
        export(Path(a.run), Path(a.out), a.voxel_mm)
        verify(Path(a.out))
