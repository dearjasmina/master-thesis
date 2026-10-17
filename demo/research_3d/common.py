"""
common.py — shared helpers for the 3-D reconstruction feasibility study (demo/PLAN_3D.md).

All pixel <-> 3-D geometry goes through scripts/training/camera_model.py (lens-shifted
principal point, EEVEE emission gain, planar View-Z depth, pixel centres at +0.5).
Nothing here re-derives the camera.

Coordinates: world millimetres == raw OBJ vertex coordinates (the generator imports the
OBJs with forward_axis='Y', up_axis='Z', global_scale=0.001 -> no axis change).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "training"))
import camera_model as cm  # noqa: E402

RUN = REPO / "demo/runs/20260922-165227_s0006"
SID = "s0006"
GBUF = RUN / "gbuffers" / SID
MESH_DIR = RUN / "meshes" / SID
OUT_DIR = RUN / "output"
GT_DIR = REPO / "data/training_dataset_v25" / SID       # rgb_preview.png per view
HERE = Path(__file__).resolve().parent
RES = HERE / "out"
RES.mkdir(exist_ok=True)

VIEWS = sorted(p.name for p in GBUF.iterdir() if p.is_dir() and p.name.startswith("v"))


def meta(view):
    return json.loads((GBUF / view / "meta.json").read_text())


def tissue_ids():
    return json.loads((GBUF / "tissue_ids.json").read_text())["tissues"]


def load_rgb(path) -> np.ndarray:
    """HxWx3 float32 in [0,1], RGB."""
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return (a[..., ::-1].astype(np.float32) / 255.0)


def save_rgb(path, img01):
    a = (np.clip(img01, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), a[..., ::-1])


class View:
    """Everything one view needs, float32 to keep memory down (~40 MB/view)."""

    def __init__(self, name, need=("depth", "segid", "out")):
        self.name = name
        self.meta = meta(name)
        vd = GBUF / name
        if "depth" in need:
            self.Z = cm.depth_m(vd).astype(np.float32)             # planar, metres
        if "segid" in need:
            self.pid = cm.segid_pid(vd).astype(np.int16)
            self.fg = self.pid > 0
        if "normals" in need:
            self.N = cm.normals_world(vd).astype(np.float32)
        if "out" in need:
            self.out = load_rgb(OUT_DIR / f"{name}.png")
        if "gt" in need:
            p = GT_DIR / name / "rgb_preview.png"
            self.gt = load_rgb(p) if p.exists() else None
        c2w = np.asarray(self.meta["cam_to_world"], np.float64)
        self.C_mm = c2w[:3, 3] * 1000.0                             # camera centre, mm

    def points_mm(self):
        """HxWx3 float32 world mm (valid where self.fg)."""
        return cm.unproject(self.Z.astype(np.float64), self.meta).astype(np.float32)


# ── meshes ──────────────────────────────────────────────────────────────────
def load_obj(path):
    """Minimal OBJ reader for extract_meshes.save_obj output: 'v x y z', 'f a//a b//b c//c'."""
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append(line[2:].split())
            elif line.startswith("f "):
                fs.append([int(t.split("/")[0]) - 1 for t in line[2:].split()][:3])
    return np.asarray(vs, np.float64), np.asarray(fs, np.int64)


def load_all_meshes():
    """Concatenated (V mm, F, face_pid, vert_pid, names) over the subject's organs."""
    ids = tissue_ids()
    Vs, Fs, fp, vp, names = [], [], [], [], []
    off = 0
    for name, pid in ids.items():
        p = MESH_DIR / f"{name}.obj"
        if not p.exists():
            continue
        V, F = load_obj(p)
        Vs.append(V); Fs.append(F + off)
        fp.append(np.full(len(F), pid, np.int16)); vp.append(np.full(len(V), pid, np.int16))
        names.append(name)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(fp), np.concatenate(vp), names


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ── cameras ─────────────────────────────────────────────────────────────────
def lookat_meta(template_meta, az_deg, el_deg, dist_m=None, fov_deg=None):
    """A synthetic camera in the generator's convention (spherical around organ_center,
    Blender -Z forward / +Y up), usable with cm.project/cm.unproject.  K comes from the
    template (or fov); the lens-shift principal point is applied by camera_model."""
    c = np.asarray(template_meta["organ_center"], np.float64)
    d = dist_m or float(template_meta["camera_distance"])
    th, ph = np.radians(az_deg), np.radians(el_deg)
    pos = c + d * np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
    fwd = (c - pos) / np.linalg.norm(c - pos)
    up0 = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ up0) > 0.999:                 # straight down/up: keep the orbit's heading
        up0 = np.array([-np.cos(th), -np.sin(th), 0.0]) * np.sign(el_deg)
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, pos
    K = np.asarray(template_meta["K"], np.float64).copy()
    if fov_deg is not None:
        K[0, 0] = K[1, 1] = 512.0 / np.tan(np.radians(fov_deg) / 2)
    m = dict(template_meta)
    m.update(K=K.tolist(), cam_to_world=c2w.tolist(), camera_pos=pos.tolist())
    return m


# ── rasteriser (numba z-buffer; perspective-correct barycentrics) ────────────
try:
    from numba import njit
except ImportError:          # pragma: no cover
    njit = None


def _raster_impl(u, v, z, F, W, H):
    zbuf = np.full((H, W), np.inf, np.float32)
    fid = np.full((H, W), -1, np.int32)
    bary = np.zeros((H, W, 3), np.float32)
    for t in range(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        za, zb, zc = z[a], z[b], z[c]
        if za <= 1e-6 or zb <= 1e-6 or zc <= 1e-6:
            continue
        ua, va, ub, vb, uc, vc = u[a], v[a], u[b], v[b], u[c], v[c]
        den = (vb - vc) * (ua - uc) + (uc - ub) * (va - vc)
        if abs(den) < 1e-12:
            continue
        x0 = max(int(np.ceil(min(ua, ub, uc))), 0)
        x1 = min(int(np.floor(max(ua, ub, uc))), W - 1)
        y0 = max(int(np.ceil(min(va, vb, vc))), 0)
        y1 = min(int(np.floor(max(va, vb, vc))), H - 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                w0 = ((vb - vc) * (x - uc) + (uc - ub) * (y - vc)) / den
                w1 = ((vc - va) * (x - uc) + (ua - uc) * (y - vc)) / den
                w2 = 1.0 - w0 - w1
                if w0 < -1e-7 or w1 < -1e-7 or w2 < -1e-7:
                    continue
                # perspective-correct: interpolate 1/z linearly in screen space
                iz = w0 / za + w1 / zb + w2 / zc
                zz = 1.0 / iz
                if zz < zbuf[y, x]:
                    zbuf[y, x] = zz
                    fid[y, x] = t
                    bary[y, x, 0] = w0 / za * zz
                    bary[y, x, 1] = w1 / zb * zz
                    bary[y, x, 2] = w2 / zc * zz
    return zbuf, fid, bary


_raster = njit(cache=True)(_raster_impl) if njit else _raster_impl


def rasterize(V_mm, F, m, W=1024, H=1024):
    """Z-buffer the mesh into camera `m` (a meta dict). Pixel (x, y) samples array
    coordinate (x, y), i.e. the pixel centre, consistent with cm.project."""
    u, v, Z = cm.project(V_mm, m, W, H)
    return _raster(u.astype(np.float64), v.astype(np.float64), Z.astype(np.float64),
                   F.astype(np.int64), W, H)


# ── sampling / metrics ──────────────────────────────────────────────────────
def bilinear(img, u, v):
    """Sample HxW[xC] at array coords (u=x, v=y) with border clamp."""
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, W - 2)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, H - 2)
    fx = np.clip(u - x0, 0, 1)[..., None] if img.ndim == 3 else np.clip(u - x0, 0, 1)
    fy = np.clip(v - y0, 0, 1)[..., None] if img.ndim == 3 else np.clip(v - y0, 0, 1)
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def label(img01, text, scale=0.9):
    """Burn a caption into the top-left corner (for side-by-side figures)."""
    a = (np.clip(img01, 0, 1) * 255).astype(np.uint8).copy()
    cv2.putText(a, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return a.astype(np.float32) / 255.0


def mem_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes
