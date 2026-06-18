"""
exr.py — Robust multilayer-EXR reader for render.exr produced by
scripts/training_dataset/generate_training_dataset.py.

The EXR is a MULTILAYER, 32-bit, scene-linear file. Blender names channels like
`ViewLayer.Combined.R`, `ViewLayer.Depth.Z`, `ViewLayer.Normal.X`, `ViewLayer.IndexOB.X`
(the view-layer prefix and casing vary by Blender version), so we DISCOVER channels at
runtime and group them by the pass segment (the token just before the .R/.X/.Z suffix).

Works with both the modern OpenEXR 3.x `File` API and the legacy `InputFile` API.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np

# Pass-name aliases → canonical key. Matched case-insensitively against the pass
# segment of each channel (e.g. "ViewLayer.Combined.R" → pass segment "Combined").
_PASS_ALIASES = {
    "image":   ["combined", "image", "rgba", "rgb"],
    "depth":   ["depth", "z"],
    "normal":  ["normal"],
    "indexob": ["indexob", "index", "objectindex"],
    "diffcol": ["diffcol"],
}

_RGB_ORDER = ["R", "G", "B"]
_XYZ_ORDER = ["X", "Y", "Z"]


def _split_channel(name: str):
    """'ViewLayer.Combined.R' → (pass_segment='Combined', component='R')."""
    parts = name.split(".")
    if len(parts) == 1:
        return parts[0], parts[0]            # e.g. bare 'R'
    return parts[-2], parts[-1]


def _read_all_channels(path: str) -> Dict[str, np.ndarray]:
    """Return {full_channel_name: HxW float32 array} using whichever API is available."""
    import OpenEXR

    # Modern API (OpenEXR >= 3.x): File(...).channels() → {name: Channel(.pixels)}
    if hasattr(OpenEXR, "File"):
        try:
            with OpenEXR.File(path) as f:
                ch = f.channels(0)
                out = {}
                for name, chan in ch.items():
                    px = np.asarray(chan.pixels)
                    out[name] = px.astype(np.float32, copy=False)
                if out:
                    return out
        except Exception:
            pass  # fall through to legacy

    # Legacy API
    import Imath
    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    names = list(header["channels"].keys())
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = f.channels(names, FLOAT)
    out = {n: np.frombuffer(b, dtype=np.float32).reshape(H, W) for n, b in zip(names, raw)}
    f.close()
    return out


def _group_by_pass(channels: Dict[str, np.ndarray]):
    """
    Group raw channels by pass segment, handling BOTH conventions:
      - per-component 2D arrays ('ViewLayer.Combined.R')  → {pass: {comp: 2D}}
      - pre-merged 3D arrays    ('ViewLayer.Combined' HxWxC, modern OpenEXR API)
                                                          → {pass: 3D}
    Returns (single, multi) dicts keyed by lowercased pass segment.
    """
    single: Dict[str, Dict[str, np.ndarray]] = {}
    multi: Dict[str, np.ndarray] = {}
    for name, arr in channels.items():
        if arr.ndim == 3:                       # already merged (RGB/RGBA/XYZ)
            seg = name.split(".")[-1].lower()
            multi[seg] = arr
        else:
            seg, comp = _split_channel(name)
            single.setdefault(seg.lower(), {})[comp.upper()] = arr
    return single, multi


def _find(single, multi, aliases: List[str]):
    """Return ('multi', arr) or ('single', comps) for the first matching pass."""
    for seg, arr in multi.items():
        if any(seg == a or seg.endswith(a) for a in aliases):
            return "multi", arr
    for seg, comps in single.items():
        if any(seg == a or seg.endswith(a) for a in aliases):
            return "single", comps
    return None, None


def _stack(comps: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    arrs = [comps[c] for c in order if c in comps]
    if not arrs:
        arrs = list(comps.values())[:1]
    return np.stack(arrs, axis=-1)


def read_render_exr(path: str, want=("image", "depth", "normal", "indexob")) -> Dict[str, np.ndarray]:
    """
    Read selected passes from a multilayer render.exr.

    Returns a dict with any of:
      image   : HxWx3 float32, scene-linear radiance
      depth   : HxW    float32, metric Z (metres); background ~1e10
      normal  : HxWx3  float32, normals in [-1, 1] (world-space as exported)
      indexob : HxW    float32, per-object pass_index (round to int for class id)
    Missing passes are simply absent from the returned dict.
    """
    channels = _read_all_channels(path)
    single, multi = _group_by_pass(channels)
    out: Dict[str, np.ndarray] = {}

    if "image" in want:
        kind, val = _find(single, multi, _PASS_ALIASES["image"])
        if kind == "multi":
            out["image"] = val[..., :3].astype(np.float32)
        elif kind == "single":
            out["image"] = _stack(val, _RGB_ORDER).astype(np.float32)

    if "depth" in want:
        kind, val = _find(single, multi, _PASS_ALIASES["depth"])
        if kind == "multi":
            out["depth"] = val[..., 0].astype(np.float32)
        elif kind == "single":
            arr = val.get("Z") if "Z" in val else next(iter(val.values()))
            out["depth"] = arr.astype(np.float32)

    if "normal" in want:
        kind, val = _find(single, multi, _PASS_ALIASES["normal"])
        if kind == "multi":
            out["normal"] = val[..., :3].astype(np.float32)
        elif kind == "single":
            out["normal"] = _stack(val, _XYZ_ORDER).astype(np.float32)

    if "indexob" in want:
        kind, val = _find(single, multi, _PASS_ALIASES["indexob"])
        if kind == "multi":
            out["indexob"] = val[..., 0].astype(np.float32)
        elif kind == "single":
            out["indexob"] = next(iter(val.values())).astype(np.float32)

    return out


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    chans = _read_all_channels(p)
    print(f"{len(chans)} channels:")
    for k in sorted(chans):
        print("  ", k, chans[k].shape, chans[k].dtype)
    single, multi = _group_by_pass(chans)
    print("single-channel passes:", sorted(single.keys()))
    print("merged passes:", sorted(multi.keys()))
    got = read_render_exr(p)
    for k, v in got.items():
        print(f"  {k}: {v.shape}  min={v.min():.4g} max={v.max():.4g}")
