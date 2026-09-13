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

from config import buffer_channels
from exr import read_render_exr
from texture_lookup import TextureBank, texture_buffer
from splits import assign_splits

cv2.setNumThreads(0)  # avoid oversubscription with DataLoader workers

# Wavelength of the first Fourier band on the `position` buffer; band k uses
# this / 4**k, i.e. 256 / 64 / 16 / 4 mm — bracketing whole-organ lobulation down to
# the vessel calibres measured from HiP-CT (0.2-1.5 mm).
POSITION_BASE_LAMBDA_MM = 256.0


def unproject_position(d: Path, meta, bands: int) -> np.ndarray:
    """Object-space XYZ per pixel, unprojected from depth.exr via meta.json.

    Module-level rather than a method because infer.py builds its input stack without a
    Dataset (it runs on the cheap G-buffers alone, with no ground truth). Both callers
    share this one implementation so inference cannot drift from training.

    v25 samples every organ texture through ShaderNodeTexCoord's *Object* output in BOX
    projection, so the GT vessel/mottle pattern at a pixel is a deterministic function of
    this coordinate — and nothing in {seg_rgb, depth, normals, segid} determines it,
    because recovering the 3D point needs the camera pose. Supplying it is what makes the
    texture predictable rather than an average over positions.

    Meshes are imported with global_scale=0.001 and no other transform, so object-space
    mm == world metres * 1000. Verified on s0050: the reconstruction lands at 118..348 mm
    against the known local vertex range 155..340, and every foreground point falls inside
    the organ bounding sphere (p99 0.303 m vs r 0.331).

    depth.exr holds Blender's 'View Z Depth' — PLANAR distance along the camera's -Z
    axis, NOT distance along the ray. The ray convention warps the result by up to 4% at
    the frame corners (~12 mm, an order of magnitude more than a vessel diameter);
    measured against the stored normals, planar is the better fit.

    No per-subject offset is subtracted: the texture is a fixed function of the ABSOLUTE
    object coordinate, so re-centring per subject would make the same input coordinate
    mean different things in different subjects and destroy exactly the determinism this
    buffer exists to create.
    """
    if "K" not in meta or "cam_to_world" not in meta:
        raise KeyError(f"{d}/meta.json lacks K / cam_to_world — the 'position' buffer "
                       f"needs the camera pose to unproject depth")
    Z = read_render_exr(str(d / "depth.exr"), want=("image",))["image"][..., 0].astype(np.float32)
    K = np.asarray(meta["K"], dtype=np.float32)
    c2w = np.asarray(meta["cam_to_world"], dtype=np.float32)
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # Blender camera: +X right, +Y up, looking down -Z; image row 0 is the top.
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * Z,
                    -(v - K[1, 2]) / K[1, 1] * Z,
                    -Z], axis=-1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]

    # These EEVEE emission passes leave the background near 1e-3, NOT 0, so a ">0"
    # test would admit the whole background and unproject it onto the camera.
    cam_dist, radius = float(meta.get("camera_distance", 0.0)), float(meta.get("radius", 0.0))
    fg = (Z > (0.5 * (cam_dist - radius) if cam_dist > 0 else 0.5))[..., None]

    planes = [np.clip(world, -2.0, 2.0)]            # metres, absolute scale kept
    mm = world * 1000.0
    for k in range(int(bands)):
        w = 2.0 * np.pi / (POSITION_BASE_LAMBDA_MM / (4.0 ** k))
        planes.append(np.sin(mm * w))
        planes.append(np.cos(mm * w))
    return (np.concatenate(planes, axis=-1) * fg).astype(np.float32)

# Canonical organ → fixed global ID (1-based), matching the TISSUES order in
# generate_training_dataset.py. The render stores per-subject pass_index (assigned by
# load order among PRESENT organs), so the same organ gets different IDs across subjects.
# We remap each subject's pass_index → this fixed global ID via its tissue_ids.json, so
# segid is a CONSISTENT organ-identity signal (organ X = same value in every subject).
CANONICAL_TISSUES = [
    "autochthon_left", "autochthon_right",
    "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_upper_lobe_left", "lung_upper_lobe_right",
    "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "heart", "esophagus", "liver", "stomach", "gallbladder", "spleen",
    "kidney_right", "kidney_left", "pancreas", "duodenum", "small_bowel", "colon",
    "urinary_bladder", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "superior_vena_cava",
]
ORGAN_TO_GID = {name: i + 1 for i, name in enumerate(CANONICAL_TISSUES)}  # 0 = background
NUM_CANONICAL = len(CANONICAL_TISSUES)  # 29


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

        # Build flat list of view samples + per-subject segid remap LUT
        # (pass_index → fixed global organ ID).
        self.samples: List[Dict] = []
        self.num_classes = NUM_CANONICAL
        self._segid_lut: Dict[str, np.ndarray] = {}
        n_excluded = 0
        for s in self.subjects:
            subj_dir = root / s
            ids_path = subj_dir / "tissue_ids.json"
            tissues = {}
            if ids_path.exists():
                try:
                    tissues = json.load(open(ids_path)).get("tissues", {})
                except Exception:
                    pass
            # LUT indexed by per-subject pass_index → global organ id (0 = background)
            max_pid = max(tissues.values(), default=0)
            lut = np.zeros(max_pid + 1, dtype=np.float32)
            for name, pid in tissues.items():
                lut[int(pid)] = ORGAN_TO_GID.get(name, 0)
            self._segid_lut[s] = lut
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

        # `position` is only as precise as depth.exr, and these passes were written as
        # HALF floats (Blender ignored color_depth='32'), quantising Z to ~2 mm at a 2 m
        # camera distance. A Fourier band shorter than a few times that step samples
        # below Nyquist and encodes quantisation noise instead of a coordinate — so
        # measure the real step and refuse an aliased setting before spending GPU days.
        if "position" in self.dcfg.input_buffers and self.dcfg.position_bands > 0:
            z = read_render_exr(str(self.samples[0]["dir"] / "depth.exr"),
                                want=("image",))["image"][..., 0]
            zf = z[z > 0.5 * z.max()]                  # drop the ~1e-3 background
            steps = np.diff(np.unique(zf))
            step_mm = float(np.median(steps)) * 1000.0 if steps.size else 0.0
            fine_mm = POSITION_BASE_LAMBDA_MM / (4.0 ** (self.dcfg.position_bands - 1))
            per_cycle = fine_mm / max(step_mm, 1e-6)
            print(f"[data] position: depth step ~{step_mm:.2f} mm, finest band "
                  f"{fine_mm:.0f} mm → {per_cycle:.1f} samples/cycle")
            if per_cycle < 3.0:
                raise ValueError(
                    f"position_bands={self.dcfg.position_bands} puts the finest band at "
                    f"{fine_mm:.0f} mm — only {per_cycle:.1f} samples per cycle against a "
                    f"{step_mm:.2f} mm depth quantisation. That encodes quantisation "
                    f"noise, not position. Use --position-bands "
                    f"{max(0, self.dcfg.position_bands - 1)} or fewer.")

    def __len__(self):
        return len(self.samples)

    # ── per-buffer loaders (return HxWxC float32, normalised to [-1,1]) ───────
    # G-buffers are SEPARATE single-layer EXRs written by generate_training_dataset.py
    # (depth.exr / normals.exr / segid.exr) — each stores its data in the RGB channels.
    def _load_seg_rgb(self, d: Path) -> np.ndarray:
        img = cv2.imread(str(d / "seg.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img * 2.0 - 1.0

    def _load_depth(self, d: Path, meta) -> np.ndarray:
        got = read_render_exr(str(d / "depth.exr"), want=("image",))
        depth = got["image"][..., 0].astype(np.float32)   # metric Z (metres)
        ref = float(meta.get("depth_metric_ref", 1.0)) or 1.0
        depth = np.clip(depth / ref, 0.0, 1.0)[..., None]
        return depth * 2.0 - 1.0

    def _load_normals(self, d: Path, meta) -> np.ndarray:
        got = read_render_exr(str(d / "normals.exr"), want=("image",))
        n = np.clip(got["image"][..., :3].astype(np.float32) * 2.0 - 1.0, -1.0, 1.0)  # (N+1)/2 → [-1,1]
        if self.dcfg.normals_space == "camera":
            c2w = np.asarray(meta["cam_to_world"], dtype=np.float32)
            R = c2w[:3, :3]                                # cam→world rotation
            n = (n.reshape(-1, 3) @ R).reshape(n.shape).astype(np.float32)  # world→cam
        return n

    def _load_segid(self, d: Path) -> np.ndarray:
        got = read_render_exr(str(d / "segid.exr"), want=("image",))
        pid = np.rint(got["image"][..., 0]).astype(np.int64)   # per-subject pass_index
        lut = self._segid_lut.get(d.parent.name)
        if lut is not None:
            pid = np.clip(pid, 0, len(lut) - 1)
            gid = lut[pid]                                       # → fixed global organ id
        else:
            gid = pid.astype(np.float32)
        gid = gid / float(NUM_CANONICAL)                        # consistent across subjects
        return np.clip(gid, 0.0, 1.0)[..., None] * 2.0 - 1.0

    def _load_position(self, d: Path, meta) -> np.ndarray:
        return unproject_position(d, meta, self.dcfg.position_bands)

    def _load_texture(self, d: Path, meta) -> np.ndarray:
        # Created lazily so each DataLoader worker holds its own tile cache.
        if getattr(self, "_tex_bank", None) is None:
            self._tex_bank = TextureBank(self.dcfg.tex_dir, self.dcfg.tex_mm)
        return texture_buffer(d, meta, self._segid_lut[d.parent.name], CANONICAL_TISSUES,
                              self._tex_bank)

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

        # GT image from render.exr only needed for the exr_* targets (preview_png doesn't).
        bufs = self.dcfg.input_buffers
        exr = (read_render_exr(str(d / "render.exr"), want=("image",))
               if self.dcfg.target in ("exr_agx", "exr_linear") else {})

        # Build input stack in configured order (G-buffers read their own EXR files).
        planes: List[np.ndarray] = []
        for b in bufs:
            if b == "seg_rgb":  planes.append(self._load_seg_rgb(d))
            elif b == "depth":  planes.append(self._load_depth(d, meta))
            elif b == "normals":planes.append(self._load_normals(d, meta))
            elif b == "segid":  planes.append(self._load_segid(d))
            elif b == "position": planes.append(self._load_position(d, meta))
            elif b == "texture":  planes.append(self._load_texture(d, meta))
            else: raise ValueError(f"unknown input buffer '{b}'")
        target = self._load_target(d, exr)

        size = self.dcfg.size
        planes = [_resize(p, size, nearest=False) for p in planes]
        target = _resize(target, size, nearest=False)

        x = np.concatenate(planes, axis=-1) if planes else np.zeros((size, size, 0), np.float32)

        # Minimal augmentation: opt-in hflip, only sound for camera-space normals.
        # It is never sound with `position`: mirroring the image does not mirror the
        # object-space coordinate, so the flipped pixels would carry the coordinate of
        # the point on the OTHER side of the organ. Fail loudly rather than train on it.
        if self.dcfg.hflip and ("position" in bufs or "texture" in bufs):
            raise ValueError("hflip is invalid with the 'position'/'texture' buffers — "
                             "mirroring the image does not mirror object-space coordinates")
        if self.split == "train" and self.dcfg.hflip and np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
            if self.dcfg.normals_space == "camera" and "normals" in bufs:
                off = sum(buffer_channels(b, self.dcfg.position_bands)
                          for b in bufs[:bufs.index("normals")])
                x[:, :, off] *= -1.0   # negate camera-space normal X

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1)))
        return {"input": x.float(), "target": y.float(),
                "subject": rec["subject"], "view": d.name}
