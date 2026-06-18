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


_COMPRESSION = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
                5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}
_PIXTYPE = {0: "UINT", 1: "HALF", 2: "FLOAT"}


def peek_exr_header(path: str) -> dict:
    """
    Parse an EXR header with the STANDARD LIBRARY ONLY (no OpenEXR, no numpy).
    Use this to confirm which passes/channels a render.exr actually contains and
    its compression codec — works on any Python, no installs.
    """
    with open(path, "rb") as f:
        buf = f.read(1 << 20)  # 1 MB is far more than any header needs
    import struct
    if struct.unpack("<I", buf[:4])[0] != 20000630:
        raise ValueError("not an EXR file (bad magic)")
    pos = 8  # skip magic(4) + version(4)

    def cstr(p):
        e = buf.index(b"\x00", p)
        return buf[p:e].decode("latin-1"), e + 1

    channels, compression, data_window = [], None, None
    while pos < len(buf) and buf[pos] != 0:
        name, pos = cstr(pos)
        atype, pos = cstr(pos)
        size = struct.unpack("<i", buf[pos:pos + 4])[0]; pos += 4
        val = buf[pos:pos + size]; pos += size
        if name == "channels":
            cp = 0
            while cp < len(val) and val[cp] != 0:
                e = val.index(b"\x00", cp)
                cname = val[cp:e].decode("latin-1"); cp = e + 1
                ptype = struct.unpack("<i", val[cp:cp + 4])[0]; cp += 4
                cp += 1 + 3 + 4 + 4  # pLinear + reserved + x/ySampling
                channels.append((cname, _PIXTYPE.get(ptype, ptype)))
        elif name == "compression" and size >= 1:
            compression = _COMPRESSION.get(val[0], val[0])
        elif name == "dataWindow" and size >= 16:
            xmin, ymin, xmax, ymax = struct.unpack("<iiii", val[:16])
            data_window = (xmax - xmin + 1, ymax - ymin + 1)
    return {"channels": channels, "compression": compression, "size": data_window}


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
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv
    p = args[0]

    # Default: stdlib-only header peek (NO OpenEXR, NO numpy needed).
    hdr = peek_exr_header(p)
    print(f"compression={hdr['compression']}  size={hdr['size']}  "
          f"{len(hdr['channels'])} channels:")
    for name, ptype in hdr["channels"]:
        print(f"   {name:<32} {ptype}")
    passes = sorted({n.split('.')[-2] if '.' in n else n for n, _ in hdr["channels"]})
    print("pass segments:", passes)

    if full:  # requires OpenEXR
        got = read_render_exr(p)
        print("decoded:")
        for k, v in got.items():
            print(f"   {k}: {v.shape}  min={v.min():.4g} max={v.max():.4g}")
