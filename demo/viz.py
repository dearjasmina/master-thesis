"""
viz.py — pure visualisation helpers for the demo: CT windowing, mask overlays,
EXR → PNG colormaps, web-mesh encoding. No thesis pipeline logic lives here —
pipeline.py calls extract_meshes / infer / exr / glint_model directly and only
hands this module arrays to turn into images.
"""
from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

PLUM = (30, 27, 43)          # #1E1B2B, stage-dark image well / EXR background


def hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ── CT windowing / slicing ──────────────────────────────────────────────────
def hu_window(vol: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    """HU volume/slice → uint8, soft-tissue window (default W400/L40)."""
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.clip((vol.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def slice2d(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """Extract a 2-D slice from a canonical (R,A,S)-oriented volume, oriented for display
    (anterior/superior up). `vol` may be the HU volume or a same-shape label volume."""
    if axis == "axial":       # fixed S(Z) -> rows: A(Y) up, cols: R(X)
        sl = vol[:, :, idx]
    elif axis == "coronal":   # fixed A(Y) -> rows: S(Z) up, cols: R(X)
        sl = vol[:, idx, :]
    elif axis == "sagittal":  # fixed R(X) -> rows: S(Z) up, cols: A(Y)
        sl = vol[idx, :, :]
    else:
        raise ValueError(axis)
    return np.flipud(sl.T)


def gray_to_rgb(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return a
    return np.stack([a, a, a], axis=-1)


def resize_rgb(a: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    if a.shape[0] == size and a.shape[1] == size:
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(a, (size, size), interpolation=interp)


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    cv2.imwrite(str(path), bgr)


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def vertical_strip(tiles: list[np.ndarray], tile: int = 256) -> np.ndarray:
    """Stack same-size RGB tiles into one tall strip (for the axial scrubber sprite)."""
    resized = [resize_rgb(gray_to_rgb(t), tile) for t in tiles]
    return np.concatenate(resized, axis=0)


# ── Mask label volumes / overlays ───────────────────────────────────────────
def build_label_volume(masks: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """masks: {tissue name: bool array, same shape}. Returns uint8 label volume,
    label = 1-based index into `order` (0 = background)."""
    shape = next(iter(masks.values())).shape
    label = np.zeros(shape, dtype=np.uint8)
    for i, name in enumerate(order, start=1):
        m = masks.get(name)
        if m is not None:
            label[m] = i
    return label


def build_lut(order: list[str], hexmap: dict[str, str], bg=PLUM) -> np.ndarray:
    """[0]=bg, [i]=colour of order[i-1]. Missing hex → mid-grey."""
    lut = np.zeros((len(order) + 1, 3), dtype=np.uint8)
    lut[0] = bg
    for i, name in enumerate(order, start=1):
        lut[i] = hex_to_rgb(hexmap[name]) if name in hexmap else (150, 150, 150)
    return lut


def apply_lut(id_arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip(id_arr.astype(np.int64), 0, len(lut) - 1)
    return lut[idx]


def display_color(hexstr: str) -> str:
    """Bright pastel of an organ's renderer colour, same hue — for overlays and chips only.
    The renderer's flat colours are dark reds/browns that vanish on a grey CT slice."""
    import colorsys
    r, g, b = (c / 255.0 for c in hex_to_rgb(hexstr))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb(h, min(max(s, 0.45), 0.6), 0.97)
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def overlay(gray_rgb: np.ndarray, label_slice: np.ndarray, lut: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mask overlay for display: `lut` should hold display_color()s; each organ also gets a
    full-strength 1-px outline so small structures stay readable."""
    colored = apply_lut(label_slice, lut).astype(np.float32)
    fg = (label_slice > 0)[..., None]
    base = gray_rgb.astype(np.float32)
    out = np.where(fg, base * (1 - alpha) + colored * alpha, base)
    lab = label_slice.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge &= lab > 0
    out[edge] = colored[edge]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── G-buffer colormaps ───────────────────────────────────────────────────────
def depth_to_pastel(depth2d: np.ndarray, fg_mask: np.ndarray,
                     near=(245, 240, 255), far=(120, 100, 170), bg=PLUM) -> np.ndarray:
    d = depth2d.astype(np.float32)
    out = np.zeros(d.shape + (3,), dtype=np.uint8)
    out[...] = bg
    if fg_mask.any():
        vals = d[fg_mask]
        lo, hi = float(vals.min()), float(vals.max())
        t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)  # 0=near .. 1=far
        t = 1.0 - t  # 1=near (light) .. 0=far
        near_a, far_a = np.array(near, np.float32), np.array(far, np.float32)
        col = (t[..., None] * near_a + (1 - t[..., None]) * far_a)
        out = np.where(fg_mask[..., None], col.astype(np.uint8), np.array(bg, np.uint8))
    return out


def normals_to_rgb(normals3d: np.ndarray, fg_mask: np.ndarray | None = None, bg=PLUM) -> np.ndarray:
    out = np.clip((normals3d + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    if fg_mask is not None:
        out = np.where(fg_mask[..., None], out, np.array(bg, np.uint8))
    return out


def channel_group_to_rgb(chan: np.ndarray, bg=PLUM) -> np.ndarray:
    """chan in [-1,1], HxW or HxWxC (C=1..3); background is exactly -1 in every channel."""
    if chan.ndim == 2:
        chan = chan[..., None]
    fg = np.any(chan > -0.999, axis=-1)
    c = np.clip((chan + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    if c.shape[-1] == 1:
        c = np.repeat(c, 3, axis=-1)
    out = np.where(fg[..., None], c, np.array(bg, np.uint8))
    return out


# ── Web mesh (decimated binary for the WebGL viewer) ────────────────────────
def decimate_for_web(mesh, max_tris: int = 15000):
    """pv.PolyData -> triangulated, decimated to <= max_tris, with point normals."""
    m = mesh.triangulate()
    n_tris = m.n_cells
    if n_tris > max_tris:
        reduction = 1.0 - (max_tris / float(n_tris))
        try:
            m = m.decimate(reduction)
        except Exception:
            m = m.decimate_pro(reduction)
    m = m.compute_normals(cell_normals=False, point_normals=True, consistent_normals=True)
    return m


def write_web_mesh_bin(path: Path, verts_mm: np.ndarray, normals: np.ndarray,
                        idx: np.ndarray, center_mm: np.ndarray) -> tuple[int, int]:
    """little-endian [u32 nV, u32 nI, f32 pos*3nV, f32 nrm*3nV, u32 idx*nI].
    Positions are centred on `center_mm` (the CT volume centre, see do_mesh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    v = (verts_mm - center_mm[None, :]).astype("<f4")
    n = normals.astype("<f4")
    i = idx.astype("<u4")
    nV, nI = v.shape[0], i.size
    with open(path, "wb") as f:
        f.write(struct.pack("<II", nV, nI))
        f.write(v.tobytes())
        f.write(n.tobytes())
        f.write(i.tobytes())
    return nV, nI // 3
