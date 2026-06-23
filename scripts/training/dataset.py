"""
dataset.py — Paired G-buffer → photoreal dataset.

Reads the nested layout written by generate_training_dataset.py:

    {root}/{subject}/
        v{NN}_az{az}_el{el}/
            render.exr        multilayer, scene-linear (Image, Depth, Normal, IndexOB, ...)
            seg.png           flat EEVEE semantic render  — the "simple input"
            rgb_preview.png   AgX + fog-glow + CA          — default GT target
            meta.json         camera K / cam_to_world / depth_metric_ref / scene_scale ...
        tissue_ids.json       {background_id, tissues:{name:id}}
        summary.json

INPUT  = configurable stack of {seg_rgb, depth, normals, segid}, each normalised to [-1,1].
TARGET = preview_png (default) | exr_agx | exr_linear, in [-1,1].

Consistency-first design: the input buffers are exact, noise-free functions of the
fixed geometry + fixed light rig, so a deterministic generator produces view-consistent
output by construction (3rd.md). Augmentation is intentionally minimal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from exr import read_render_exr
from splits import assign_splits

cv2.setNumThreads(0)  # avoid oversubscription with DataLoader workers


# ── Tonemapping ───────────────────────────────────────────────────────────────
def agx_tonemap(linear_rgb: np.ndarray) -> np.ndarray:
    """
    Approximate AgX (Troy Sobotka) display transform, linear sRGB → display sRGB [0,1].
    NOTE: this is an APPROXIMATION of Blender's "AgX - Medium Contrast" look and does
    NOT include the fog-glow / chromatic-aberration compositor effects. For an exact
    target use target="preview_png". Used only when target="exr_agx".
    """
    x = np.clip(linear_rgb, 0.0, None)
    # AgX log2 encoding over [-12.47, 4.026] EV.
    min_ev, max_ev = -12.47393, 4.026069
    x = np.where(x > 0, np.log2(x + 1e-10), min_ev)
    x = np.clip((x - min_ev) / (max_ev - min_ev), 0.0, 1.0)
    # 6th-order sigmoid polynomial fit (Sobotka / Mitchell).
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    y = (-17.86 * x6 + 78.01 * x5 - 126.7 * x4 + 92.06 * x3
         - 28.72 * x2 + 4.361 * x - 0.1718)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _resize(arr: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    interp = cv2.INTER_NEAREST if nearest else (
        cv2.INTER_AREA if arr.shape[0] > size else cv2.INTER_LINEAR)
    out = cv2.resize(arr, (size, size), interpolation=interp)
    if out.ndim == 2 and arr.ndim == 3:
        out = out[..., None]
    return out


class PairedRenderDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.dcfg = cfg.data
        self.split = split

        root = Path(self.dcfg.root)
        if not root.exists():
            raise FileNotFoundError(f"data root not found: {root.resolve()}")

        subjects = [p.name for p in sorted(root.iterdir())
                    if p.is_dir() and any(p.glob("v*/meta.json"))]
        if not subjects:
            raise RuntimeError(f"no subjects with v*/meta.json under {root.resolve()}")

        # Subject selection. Explicit --subjects overrides the split entirely
        # (used for the overfit sanity test); otherwise filter by split.
        if self.dcfg.subjects:
            requested = list(self.dcfg.subjects)
            self.subjects = [s for s in requested if s in subjects]
            missing = [s for s in requested if s not in subjects]
            if missing:
                print(f"[data] WARNING: requested subjects not found: {missing}")
        else:
            split_map = assign_splits(subjects, cfg)
            self.subjects = [s for s in subjects if split_map.get(s) == split]
        if self.dcfg.max_subjects > 0:
            self.subjects = self.subjects[:self.dcfg.max_subjects]

        # Optional per-view exclude list ("subject/view") — e.g. badly-framed views
        # flagged by score_framing.py.
        exclude = set()
        if self.dcfg.exclude_file:
            try:
                exclude = set(json.load(open(self.dcfg.exclude_file)))
            except Exception as e:
                print(f"[data] WARNING: could not read exclude_file {self.dcfg.exclude_file}: {e}")

        # Build flat list of view samples.
        self.samples: List[Dict] = []
        self.num_classes = 1
        n_excluded = 0
        for s in self.subjects:
            subj_dir = root / s
            ids_path = subj_dir / "tissue_ids.json"
            n_cls = 29
            if ids_path.exists():
                try:
                    ids = json.load(open(ids_path))
                    n_cls = max(ids.get("tissues", {}).values(), default=29)
                except Exception:
                    pass
            self.num_classes = max(self.num_classes, n_cls)
            for view_dir in sorted(subj_dir.glob("v*")):
                if not (view_dir / "meta.json").exists():
                    continue
                if f"{s}/{view_dir.name}" in exclude:
                    n_excluded += 1
                    continue
                self.samples.append({"subject": s, "dir": view_dir})
        if exclude:
            print(f"[data] excluded {n_excluded} badly-framed views (from {self.dcfg.exclude_file})")

        if not self.samples:
            raise RuntimeError(f"split '{split}' has 0 samples")

    def __len__(self):
        return len(self.samples)

    # ── per-buffer loaders (return HxWxC float32, normalised to [-1,1]) ───────
    def _load_seg_rgb(self, d: Path, exr) -> np.ndarray:
        img = cv2.imread(str(d / "seg.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img * 2.0 - 1.0

    def _load_depth(self, d: Path, exr, meta) -> np.ndarray:
        depth = exr["depth"].astype(np.float32)
        ref = float(meta.get("depth_metric_ref", 1.0)) or 1.0
        depth = np.clip(depth / ref, 0.0, 1.0)[..., None]
        return depth * 2.0 - 1.0

    def _load_normals(self, d: Path, exr, meta) -> np.ndarray:
        n = exr["normal"].astype(np.float32)              # world-space [-1,1]
        if self.dcfg.normals_space == "camera":
            c2w = np.asarray(meta["cam_to_world"], dtype=np.float32)
            R = c2w[:3, :3]                                # cam→world rotation
            flat = n.reshape(-1, 3) @ R                    # world→cam = n @ R (= R^T·n)
            n = flat.reshape(n.shape).astype(np.float32)
        return np.clip(n, -1.0, 1.0)

    def _load_segid(self, d: Path, exr) -> np.ndarray:
        ids = np.rint(exr["indexob"]).astype(np.float32)
        ids = ids / float(max(self.num_classes, 1))
        return np.clip(ids, 0.0, 1.0)[..., None] * 2.0 - 1.0

    def _load_target(self, d: Path, exr) -> np.ndarray:
        t = self.dcfg.target
        if t == "preview_png":
            img = cv2.imread(str(d / "rgb_preview.png"), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        elif t == "exr_agx":
            img = agx_tonemap(exr["image"])
        elif t == "exr_linear":
            # Crude display clamp of scene-linear radiance (see README; not recommended).
            img = np.clip(exr["image"], 0.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"unknown target '{t}'")
        return img * 2.0 - 1.0

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        d: Path = rec["dir"]
        meta = json.load(open(d / "meta.json"))

        # Decide which EXR passes we actually need.
        need = set()
        bufs = self.dcfg.input_buffers
        if "depth" in bufs:   need.add("depth")
        if "normals" in bufs: need.add("normal")
        if "segid" in bufs:   need.add("indexob")
        if self.dcfg.target in ("exr_agx", "exr_linear"): need.add("image")
        exr = read_render_exr(str(d / "render.exr"), want=tuple(need)) if need else {}

        # Build input stack in configured order.
        planes: List[np.ndarray] = []
        for b in bufs:
            if b == "seg_rgb":  planes.append(self._load_seg_rgb(d, exr))
            elif b == "depth":  planes.append(self._load_depth(d, exr, meta))
            elif b == "normals":planes.append(self._load_normals(d, exr, meta))
            elif b == "segid":  planes.append(self._load_segid(d, exr))
            else: raise ValueError(f"unknown input buffer '{b}'")
        target = self._load_target(d, exr)

        size = self.dcfg.size
        planes = [_resize(p, size, nearest=False) for p in planes]
        target = _resize(target, size, nearest=False)

        x = np.concatenate(planes, axis=-1) if planes else np.zeros((size, size, 0), np.float32)

        # Minimal augmentation: opt-in hflip, only sound for camera-space normals.
        if self.split == "train" and self.dcfg.hflip and np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
            if self.dcfg.normals_space == "camera" and "normals" in bufs:
                off = sum({"seg_rgb": 3, "depth": 1, "normals": 3, "segid": 1}[b]
                          for b in bufs[:bufs.index("normals")])
                x[:, :, off] *= -1.0   # negate camera-space normal X

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1)))
        return {"input": x.float(), "target": y.float(),
                "subject": rec["subject"], "view": d.name}
