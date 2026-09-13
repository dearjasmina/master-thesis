"""
texture_lookup.py — deferred texturing: the v25 organ textures, as an input G-buffer.

Every v25 organ is shaded from exactly three image tiles (albedo / normal / roughness)
in BOX projection at --tex_mm; make_material() disables every procedural layer when the
tiles exist, and all 18 organ classes have all three. So the texture at a pixel is a
deterministic function of the object-space point, which camera_model.unproject() recovers.
Handing the network the looked-up texture means it only has to learn shading — it no
longer has to memorise 18 texture sets as a function of position, which a conv generator
cannot do.

The projection below is Cycles' svm_node_tex_image_box. Validated on the server tiles with
texture_learnability.py `oracle`: in every large single-texture region a blind search over
8 uv conventions picks exactly this one, by a 10-20x margin, and the signal vanishes with a
half-tile offset and decays with added position noise.

TILES MUST BE THE ONES THE DATASET WAS RENDERED WITH. synth_tissue_textures.py seeds with
Python's salted hash(), so re-running it produces different vessels; the tiles are not in
git. Use the server copy (checked by data/renders/textures/tissue/MD5SUMS.txt).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import camera_model as cm

# REFERENCE_LOOK keys in render_pair_totalseg_v25_textured.py — the tile class names.
TEX_CLASSES = ["lung", "liver", "spleen", "kidney", "heart", "stomach", "bowel",
               "duodenum", "colon", "pancreas", "gallbladder", "esophagus",
               "urinary_bladder", "aorta", "vena_cava", "portal", "vertebrae",
               "autochthon"]
TEX_MM = 80.0       # generate_training_dataset.py --tex_mm default
BLEND = 0.35        # v25 _box_tex projection_blend
TEXTURE_CHANNELS = ("albedo_r", "albedo_g", "albedo_b",
                    "normalmap_r", "normalmap_g", "normalmap_b", "roughness")


def tex_class(organ: str):
    """Mirror of v25 tissue_textures(): exact name, else the longest class contained."""
    if organ in TEX_CLASSES:
        return organ
    hits = [k for k in TEX_CLASSES if k in organ]
    return max(hits, key=len) if hits else None


# ── Cycles BOX projection (svm_node_tex_image_box) ────────────────────────────
def box_weights(N, blend=BLEND):
    A = np.abs(N)
    A = A / np.maximum(A.sum(-1, keepdims=True), 1e-12)
    x, y, z = A[..., 0], A[..., 1], A[..., 2]
    limit = 0.5 * (1.0 + blend)
    w = np.zeros_like(A)
    cx = (x > limit * (x + y)) & (x > limit * (x + z))
    cy = ~cx & (y > limit * (x + y)) & (y > limit * (y + z))
    cz = ~cx & ~cy & (z > limit * (x + z)) & (z > limit * (y + z))
    w[cx, 0] = 1.0
    w[cy, 1] = 1.0
    w[cz, 2] = 1.0
    rest = ~(cx | cy | cz)
    sat = lambda t: np.clip(t, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = rest & (z < (1 - limit) * (y + x))
        m2 = rest & ~m1 & (x < (1 - limit) * (y + z))
        m3 = rest & ~m1 & ~m2 & (y < (1 - limit) * (x + z))
        m4 = rest & ~m1 & ~m2 & ~m3
        t = sat((x / (x + y) - 0.5 * (1 - blend)) / blend); w[m1, 0] = t[m1]; w[m1, 1] = 1 - t[m1]
        t = sat((y / (y + z) - 0.5 * (1 - blend)) / blend); w[m2, 1] = t[m2]; w[m2, 2] = 1 - t[m2]
        t = sat((x / (x + z) - 0.5 * (1 - blend)) / blend); w[m3, 0] = t[m3]; w[m3, 2] = 1 - t[m3]
        for k, c in enumerate((x, y, z)):
            w[m4, k] = (((2 - limit) * c + (limit - 1)) / (2 * limit - 1))[m4]
    return np.nan_to_num(w)


# uv source axes per projection axis, as in Cycles: X->(y,z)  Y->(x,z)  Z->(y,x)
AXIS_UV = {0: (1, 2), 1: (0, 2), 2: (1, 0)}
# Cycles flips u by the SIGN of the normal: X when N.x<0, Y when N.y>0, Z when N.z>0
CYCLES_FLIP_U = {0: lambda s: s < 0, 1: lambda s: s > 0, 2: lambda s: s > 0}


def sample_wrap(img, u, v, scale=1.0):
    """Bilinear REPEAT sampling of a 1- or 3-channel tile. Blender: v=0 is the BOTTOM row."""
    H, W = img.shape[:2]
    x = np.mod(u, 1.0) * W - 0.5
    y = (1.0 - np.mod(v, 1.0)) * H - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    x0 %= W; y0 %= H
    x1, y1 = (x0 + 1) % W, (y0 + 1) % H
    g = lambda yy, xx: img[yy, xx].astype(np.float64)
    out = ((g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy)
           + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy)
    return out * scale


def project_axis(img, P_mm, N, axis, swap=False, fu=False, fv=False, cycles=False,
                 tex_mm=TEX_MM, scale=1.0):
    """One BOX face. cycles=True applies Cycles' own sign-dependent u flip; otherwise the
    explicit (swap, fu, fv) variant is used (texture_learnability's blind search)."""
    co = P_mm / tex_mm
    b, c = AXIS_UV[axis]
    u, v = co[..., b], co[..., c]
    if swap:
        u, v = v, u
    if cycles:
        u = np.where(CYCLES_FLIP_U[axis](N[..., axis]), 1.0 - u, u)
    else:
        if fu: u = -u
        if fv: v = -v
    return sample_wrap(img, u, v, scale)


def box_project(img, P_mm, N, tex_mm=TEX_MM, blend=BLEND, w=None, scale=1.0):
    """Cycles-exact blended BOX projection at points P_mm (n,3) with normals N (n,3)."""
    if w is None:
        w = box_weights(N, blend)
    C = img.shape[2] if img.ndim == 3 else None
    out = np.zeros(P_mm.shape[:-1] + ((C,) if C else ()), dtype=np.float64)
    for a in range(3):
        m = w[..., a] > 0
        if not m.any():
            continue
        val = project_axis(img, P_mm[m], N[m], a, cycles=True, tex_mm=tex_mm, scale=scale)
        wa = w[m, a]
        out[m] += (wa[:, None] if C else wa) * val
    return out


# ── the buffer ────────────────────────────────────────────────────────────────
class TextureBank:
    """Lazily loaded tiles, kept in their file dtype (uint8: ~126 MB for all 18 classes)."""
    KINDS = ("albedo", "normal", "rough")

    def __init__(self, tex_dir, tex_mm=TEX_MM):
        self.tex_dir = Path(tex_dir)
        self.tex_mm = float(tex_mm)
        self._cache = {}
        if not self.tex_dir.is_dir():
            raise FileNotFoundError(f"texture dir not found: {self.tex_dir.resolve()}")

    def get(self, cls, kind):
        key = (cls, kind)
        if key not in self._cache:
            p = self.tex_dir / f"{cls}_{kind}.png"
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if a is not None:
                if a.ndim == 3:
                    a = np.ascontiguousarray(cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB))
                if kind == "rough" and a.ndim == 3:
                    a = np.ascontiguousarray(a[..., 0])
                scale = 1.0 / (65535.0 if a.dtype == np.uint16 else 255.0)
                self._cache[key] = (a, scale)
            else:
                self._cache[key] = (None, 0.0)
        return self._cache[key]


def texture_buffer(view_dir, meta, pid_to_gid, organ_names, bank: TextureBank) -> np.ndarray:
    """HxWx7 float32 in [-1, 1]: albedo RGB | normal-map RGB | roughness. Background -1.

    pid_to_gid: the subject's per-subject pass_index -> global organ id LUT (1-based).
    organ_names: global id - 1 -> organ name (dataset.CANONICAL_TISSUES).
    The normal-map channels are the raw box-blended tile colour, exactly what Cycles feeds
    its Normal Map node — not a decoded normal, because the tangent frame Blender uses on
    these UV-less meshes is not recorded, so the network has to learn that part.
    """
    view_dir = Path(view_dir)
    P = cm.unproject(cm.depth_m(view_dir), meta)
    N = cm.normals_world(view_dir)
    lut = np.asarray(pid_to_gid).astype(np.int64)
    gid = lut[np.clip(cm.segid_pid(view_dir), 0, len(lut) - 1)]
    H, W = gid.shape
    out = np.full((H, W, 7), -1.0, dtype=np.float32)
    for g in np.unique(gid):
        if g <= 0:
            continue
        cls = tex_class(organ_names[int(g) - 1])
        if cls is None:
            continue
        m = gid == g
        Pm, Nm = P[m], N[m]
        w = box_weights(Nm)
        for kind, sl in (("albedo", slice(0, 3)), ("normal", slice(3, 6)), ("rough", slice(6, 7))):
            img, scale = bank.get(cls, kind)
            if img is None:
                continue
            val = box_project(img, Pm, Nm, tex_mm=bank.tex_mm, w=w, scale=scale)
            if val.ndim == 1:
                val = val[:, None]
            out[m, sl] = (np.clip(val, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)
    return out
