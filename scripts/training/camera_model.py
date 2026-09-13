"""
camera_model.py — the camera and G-buffer encoding that meta.json does NOT describe.

Every consumer of pixel <-> 3D geometry must go through this module. Two facts are missing
from meta.json, and ignoring them put the gbuf_pos1024 position buffer 7-30 mm off the
surface and ~30 mm inconsistent between views of the same point:

1. LENS SHIFT. render_pair_totalseg_v25_textured.py setup_camera() sets shift_x=-0.05,
   shift_y=+0.02, but meta.json's K places the principal point at the image centre.
   Blender shift is in units of the larger sensor dimension, so the true principal
   point is (W/2 - sx*S, H/2 + sy*S) = (563.2, 532.48) at 1024 px.
2. EMISSION GAIN. depth.exr / normals.exr / segid.exr are EEVEE emission renders, and
   colour management multiplies them on write: R and G by 0.99213, B by 0.99213^2.
   Constant over 400 random views (B/R 0.9916-0.9927). Uncorrected, depth reads 0.8%
   short and normals decode to |N| 0.98 instead of 1.

Also: depth is PLANAR 'View Z Depth' in metres (ray distance fits measurably worse), and
pixel centres sit at index + 0.5.

Verified against the subjects' own meshes: corrected reconstruction 0.3-0.9 mm from the
surface (the mesh vertex-spacing floor); cross-view agreement 0.28 mm median on a
0.49 mm-depth-step subject. Guarded by `scripts/analysis/texture_learnability.py geometry`.

dataset.py::unproject_position is deliberately left on the legacy model so the existing
gbuf_pos1024 checkpoint still sees the inputs it was trained on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exr import read_render_exr

CAMERA_SHIFT = (-0.05, 0.02)
EMISSION_GAIN = np.array([0.99213, 0.99213, 0.99213 ** 2], dtype=np.float64)


def read_emission(path) -> np.ndarray:
    """HxWx3 float64 from an EEVEE emission pass, with the colour-management gain removed."""
    a = read_render_exr(str(path), want=("image",))["image"][..., :3].astype(np.float64)
    return a / EMISSION_GAIN


def principal_point(W: int, H: int):
    S = max(W, H)
    return W / 2.0 - CAMERA_SHIFT[0] * S, H / 2.0 + CAMERA_SHIFT[1] * S


def depth_m(view_dir) -> np.ndarray:
    """True planar depth in metres, HxW. Background is ~1e-3, not 0."""
    return read_emission(Path(view_dir) / "depth.exr")[..., 0]


def normals_world(view_dir) -> np.ndarray:
    """Unit world-space normals, HxWx3."""
    n = read_emission(Path(view_dir) / "normals.exr") * 2.0 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def segid_pid(view_dir) -> np.ndarray:
    """Per-subject pass_index, HxW int64 (map to global ids with the subject's LUT)."""
    return np.rint(read_emission(Path(view_dir) / "segid.exr")[..., 0]).astype(np.int64)


def unproject(Z: np.ndarray, meta) -> np.ndarray:
    """Planar depth (metres, gain-corrected) -> world/object-space millimetres, HxWx3.

    Object space == world * 1000: meshes are imported with global_scale=0.001 and no
    other transform, so this is the coordinate v25's BOX-projected textures are sampled in.
    """
    K = np.asarray(meta["K"], np.float64)
    c2w = np.asarray(meta["cam_to_world"], np.float64)
    H, W = Z.shape
    cx, cy = principal_point(W, H)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    cam = np.stack([(u - cx) / K[0, 0] * Z, -(v - cy) / K[1, 1] * Z, -Z], -1)
    return (cam @ c2w[:3, :3].T + c2w[:3, 3]) * 1000.0


def project(P_mm: np.ndarray, meta, W: int = 1024, H: int = 1024):
    """World millimetres -> (u, v) ARRAY coordinates and planar depth Z (metres)."""
    c2w = np.asarray(meta["cam_to_world"], np.float64)
    K = np.asarray(meta["K"], np.float64)
    cam = (P_mm / 1000.0 - c2w[:3, 3]) @ c2w[:3, :3]      # world -> camera (R orthonormal)
    Z = -cam[..., 2]
    cx, cy = principal_point(W, H)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + K[0, 0] * cam[..., 0] / Z - 0.5
        v = cy - K[1, 1] * cam[..., 1] / Z - 0.5
    return u, v, Z
