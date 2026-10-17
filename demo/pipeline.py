"""
pipeline.py — the Run object: stages, event emission, timing, run dir.

Orchestrates the thesis code (extract_meshes, generate_training_dataset.py via Blender
subprocess, infer.load_input, the trained generator) end to end for one subject. Does
NOT reimplement any of it — see the imports below, all resolved from scripts/training
and scripts/training_dataset via sys.path.

Event contract: see demo/PLAN.md §3.1. Every event is a JSON object with `t` (seconds
since run start), `type`, and `stage` when relevant; both appended to
<run_dir>/events.jsonl and pushed to any live SSE subscribers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent


def _load_demo_config():
    """Load demo/config.py under a private module name so it never claims the bare
    'config' entry in sys.modules — scripts/training/config.py needs that name for its
    own internal `from config import preset` and we must not shadow it (see report)."""
    spec = importlib.util.spec_from_file_location("_demo_config", DEMO_DIR / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load_demo_config()

sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(cfg.TRAINING_DIR))
sys.path.insert(0, str(cfg.TRAINING_DATASET_DIR))

import viz  # noqa: E402  (demo/viz.py)
import recon  # noqa: E402  (demo/recon.py — the 3-D view stage, PLAN_3D.md §4.2)

from config import preset as train_preset  # noqa: E402  scripts/training/config.py
from exr import read_render_exr  # noqa: E402
from splits import _bucket  # noqa: E402
import glint_model  # noqa: E402
import infer as infer_mod  # noqa: E402
from extract_meshes import extract_mesh, save_obj, TISSUE_NAMES  # noqa: E402
import nibabel as nib  # noqa: E402

CANONICAL_ORDER = TISSUE_NAMES  # identical order to dataset.CANONICAL_TISSUES (verified)

STAGES = ("ct", "masks", "mesh", "gbuffer", "buffers", "infer", "recon")

_HINTS = {
    "ct": "check the CT is a valid NIfTI (.nii/.nii.gz) with a real 3-D volume",
    "masks": "check segmentations/ exists next to ct.nii.gz and holds the 29 tissue masks",
    "mesh": "marching-cubes/meshing failed — check the segmentation masks are not empty",
    "gbuffer": "Blender failed — check BLENDER path, the run's mesh dir, and --tex_dir "
               "(must be data/renders/textures/tissue_server)",
    "buffers": "G-buffer -> network input failed — check the texture dir and that all "
               "gbuffer views wrote seg/depth/normals/segid + meta.json",
    "infer": "the model is not ready (still loading / failed to load) or a saved stack "
             "is missing — check /api/status",
    "recon": "3-D view could not be built — the renders are unaffected",
}


# ── Model, loaded once at server start (§3.7) ───────────────────────────────
class ModelHolder:
    def __init__(self):
        self.status = "loading"  # loading|ready|error
        self.device = None
        self.error = None
        self.G = None
        self.gcfg = None
        self.params_m = None
        self.lock = threading.Lock()

    def load(self):
        try:
            import torch
            from networks import define_G
            gcfg = train_preset(cfg.PRESET)
            gcfg.data.tex_dir = str(cfg.TEX_DIR)
            if torch.cuda.is_available():                       # server (L4)
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():                # this laptop
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            G = define_G(gcfg).eval()
            # mmap: the 2.1 GB file also holds D and both optimiser states; mapping it
            # means only the G tensors are ever paged in (~0.7 GB), not the whole file.
            ckpt = torch.load(str(cfg.CKPT), map_location="cpu", weights_only=False, mmap=True)
            G.load_state_dict(ckpt["G"])
            del ckpt
            import gc
            gc.collect()
            G = G.to(device)
            n_params = sum(p.numel() for p in G.parameters())
            dummy = torch.zeros(1, gcfg.input_nc(), gcfg.data.size, gcfg.data.size, device=device)
            with torch.no_grad():
                G(dummy)
            del dummy
            with self.lock:
                self.G, self.device, self.gcfg = G, device, gcfg
                self.params_m = n_params / 1e6
                self.status = "ready"
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error = f"{e}\n{traceback.format_exc()}"

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "device": str(self.device) if self.device else None,
                    "error": self.error, "ckpt": str(cfg.CKPT), "preset": cfg.PRESET,
                    "params_m": self.params_m}


MODEL = ModelHolder()


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ── CT inspect (also used standalone by /api/inspect, cached) ───────────────
_INSPECT_CACHE: dict[str, dict] = {}
_INSPECT_LOCK = threading.Lock()

_SID_RE = re.compile(r"^s\d{4}$")


def _sanitize_stem(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s or "subject"


def resolve_subject(path_str: str):
    """path -> (ct_path, subject_root, sid). Raises FileNotFoundError with a clear message."""
    p = Path(path_str).expanduser()
    if p.is_dir():
        ct_path = p / "ct.nii.gz"
        subject_root = p
    elif p.is_file():
        ct_path = p
        subject_root = p.parent
    else:
        raise FileNotFoundError(f"no such file or directory: {p}")
    if not ct_path.exists():
        raise FileNotFoundError(f"no ct.nii.gz found at {ct_path}")
    name = subject_root.name
    sid = name if _SID_RE.match(name) else _sanitize_stem(ct_path.stem.replace(".nii", ""))
    return ct_path, subject_root, sid


def inspect_ct(path_str: str, force: bool = False) -> dict:
    """CT stage result (§3.2), cached per resolved ct_path. Writes previews under
    demo/runs/_inspect_<sid>/ct/ so a subsequent real Run can just copy them."""
    ct_path, subject_root, sid = resolve_subject(path_str)
    # keyed on the folder as given (not resolved): a symlinked CT in a folder WITHOUT
    # segmentations/ must not inherit the cached result of the folder it points into.
    cache_key = f"{ct_path.absolute()}:{ct_path.stat().st_mtime_ns}"
    with _INSPECT_LOCK:
        cached = _INSPECT_CACHE.get(cache_key)
        if cached is not None and not force:
            return cached

    t0 = time.time()
    img = nib.load(str(ct_path))
    zooms = img.header.get_zooms()[:3]
    shape = img.shape[:3]
    data_dtype = str(img.get_data_dtype())
    orientation = "".join(nib.aff2axcodes(img.affine))

    canon = nib.as_closest_canonical(img)
    cdata = np.asarray(canon.dataobj, dtype=np.float32)
    load_time = time.time() - t0

    sample = cdata[::4, ::4, ::4]
    hu_min = float(sample.min())
    hu_p1 = float(np.percentile(sample, 1))
    hu_p99 = float(np.percentile(sample, 99))
    hu_max = float(sample.max())

    extent_cm = [round(float(shape[i]) * float(zooms[i]) / 10.0, 2) for i in range(3)]
    file_size = ct_path.stat().st_size

    seg_dir = subject_root / "segmentations"
    has_seg = seg_dir.is_dir()
    n_tissues = 0
    if has_seg:
        n_tissues = sum(1 for t in TISSUE_NAMES if (seg_dir / f"{t}.nii.gz").exists())

    # The split is defined only for TotalSegmentator ids; any other CT was never in the
    # dataset, so it is held out by construction.
    split = (_bucket(sid, cfg.SPLIT_SEED, cfg.SPLIT_VAL_FRAC, cfg.SPLIT_TEST_FRAC)
             if _SID_RE.match(sid) else "external")
    gt_available = any((root / sid).is_dir() for root in cfg.GT_ROOTS)

    checks = []
    checks.append({"name": "3-D volume", "status": "pass" if cdata.ndim == 3 else "fail",
                    "reason": f"shape {shape}"})
    spacing_ok = all(0.3 <= z <= 10 for z in zooms)
    checks.append({"name": "voxel spacing", "status": "pass" if spacing_ok else "fail",
                    "reason": f"{tuple(round(float(z), 3) for z in zooms)} mm"})
    checks.append({"name": "HU range looks like CT", "status": "pass" if hu_min <= -900 else "warn",
                    "reason": f"min HU {hu_min:.0f}" + ("" if hu_min <= -900 else " (expected <= -900)")})
    checks.append({"name": "segmentations/ present", "status": "pass" if has_seg else "fail",
                    "reason": str(seg_dir) if has_seg else "needs TotalSegmentator masks"})
    if has_seg:
        checks.append({"name": "tissue coverage",
                        "status": "pass" if n_tissues == len(TISSUE_NAMES) else
                                   ("warn" if n_tissues > 0 else "fail"),
                        "reason": f"{n_tissues}/{len(TISSUE_NAMES)} tissues present"})
    else:
        checks.append({"name": "tissue coverage", "status": "fail", "reason": "no segmentations/"})
    checks.append({"name": "split", "status": "warn" if split == "train" else "pass",
                    "reason": {"train": "train — seen in training", "external": "external — not in the dataset"}
                              .get(split, f"{split} — held out")})
    checks.append({"name": "GT reference", "status": "pass" if gt_available else "warn",
                    "reason": (f"found in {cfg.GT_ROOTS[0].name}" if gt_available
                               else "no reference render for this subject")})

    out_dir = cfg.RUNS / f"_inspect_{sid}" / "ct"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windowed = viz.hu_window(cdata)
    mid = {"axial": shape[2] // 2, "coronal": shape[1] // 2, "sagittal": shape[0] // 2}
    slice_urls = {}
    for axis in ("axial", "coronal", "sagittal"):
        sl = viz.slice2d(windowed, axis, mid[axis])
        rgb = viz.resize_rgb(viz.gray_to_rgb(sl), 512)
        viz.save_png(out_dir / f"{axis}.png", rgb)
        slice_urls[axis] = f"ct/{axis}.png"

    n_sprite = min(64, shape[2])
    sprite_idx = sorted(set(int(round(x)) for x in np.linspace(0, shape[2] - 1, n_sprite)))
    tiles = [viz.slice2d(windowed, "axial", k) for k in sprite_idx]
    strip = viz.vertical_strip(tiles, 256)
    viz.save_jpg(out_dir / "axial_stack.jpg", strip, 85)

    info = {
        "sid": sid, "ct_path": str(ct_path), "subject_root": str(subject_root),
        "shape": list(shape), "spacing_mm": [round(float(z), 4) for z in zooms],
        "extent_cm": extent_cm, "dtype": data_dtype, "orientation": orientation,
        "hu_min": hu_min, "hu_p1": hu_p1, "hu_p99": hu_p99, "hu_max": hu_max,
        "file_size": file_size, "load_time_s": round(load_time, 3),
        "split": split, "gt_available": gt_available, "n_tissues": n_tissues,
        "n_tissues_total": len(TISSUE_NAMES), "has_segmentations": has_seg,
        "checks": checks,
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))

    result = {
        "info": info, "checks": checks, "slice_urls": slice_urls,
        "axial_stack_url": "ct/axial_stack.jpg", "sprite_n": len(sprite_idx),
        "sprite_idx": sprite_idx, "mid_idx": mid, "shape": list(shape),
        "cache_dir": str(out_dir.parent),
    }
    with _INSPECT_LOCK:
        _INSPECT_CACHE[cache_key] = result
    return result


# ── One run per machine ──────────────────────────────────────────────────────
class Busy(RuntimeError):
    """Another run holds the machine-wide lock, or there is not enough free memory."""


def _try_flock(path: Path):
    """Non-blocking exclusive flock; returns the open file (keep it!) or None if held.
    The OS drops the lock when the holder exits or crashes, so it can never go stale."""
    import fcntl
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
    return f


def _holder_pid(path: Path) -> str:
    try:
        return path.read_text().strip() or "?"
    except OSError:
        return "?"


# ── Run ──────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, path_str: str):
        self.ct_path, self.subject_root, self.sid = resolve_subject(path_str)
        # A pipeline run peaks at several GB (Blender, the buffer worker, the model);
        # two at once on a 16 GB laptop is what froze the machine. The flock is across
        # processes, so a second server (any port) cannot start a run either.
        lock_path = cfg.RUNS / ".run.lock"
        self._lock_file = _try_flock(lock_path)
        if self._lock_file is None:
            raise Busy(f"another run is in progress (pid {_holder_pid(lock_path)})")
        free = cfg.available_gb()
        if free < cfg.MIN_FREE_GB:
            self._release_lock()
            raise Busy(f"only {free:.1f} GB of memory free (need {cfg.MIN_FREE_GB:.0f} GB) — "
                       "close some apps and try again")
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        for n in range(100):
            self.run_id = f"{stamp}_{self.sid}" + (f"_{n}" if n else "")
            self.run_dir = cfg.RUNS / self.run_id
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        self.events_path = self.run_dir / "events.jsonl"
        self.t0 = time.time()
        self.status = "running"
        self.error = None
        self.timings: dict[str, float] = {}
        self._events_lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._line_no = 0
        self.split = None
        # populated by do_ct, consumed by do_masks (slice alignment)
        self._mid_idx = None
        self._sprite_idx = None
        self._ct_shape = None
        self._masks_cache = None  # {name: bool array (canonical)}
        self._gbuf_subject_dir = None

    # -- event plumbing --------------------------------------------------
    def _write(self, ev: dict):
        line = json.dumps(ev)
        with self._events_lock:
            with open(self.events_path, "a") as f:
                f.write(line + "\n")
            self._line_no += 1
            for q in list(self._subscribers):
                q.put(line)

    def emit(self, type_, **kw):
        ev = {"t": round(time.time() - self.t0, 3), "type": type_}
        ev.update(kw)
        self._write(ev)

    def progress(self, stage, done, total, label):
        self.emit("progress", stage=stage, done=done, total=total, label=label)

    def stat(self, stage, key, label, value, unit="", fmt="text"):
        self.emit("stat", stage=stage, key=key, label=label, value=value, unit=unit, fmt=fmt)

    def artifact(self, stage, key, url, view=None, meta=None):
        ev = {"stage": stage, "key": key, "url": f"/runs/{self.run_id}/{url}"}
        if view is not None:
            ev["view"] = view
        if meta is not None:
            ev["meta"] = meta
        self.emit("artifact", **ev)

    def log(self, stage, line):
        self.emit("log", stage=stage, line=line)

    def subscribe(self):
        q = queue.Queue()
        with self._events_lock:
            with open(self.events_path) as f:
                backlog = f.read().splitlines()
            self._subscribers.append(q)
        return backlog, q

    def unsubscribe(self, q):
        with self._events_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _close_subscribers(self):
        with self._events_lock:
            for q in list(self._subscribers):
                q.put(None)

    def _release_lock(self):
        if getattr(self, "_lock_file", None) is not None:
            self._lock_file.close()          # closing the fd releases the flock
            self._lock_file = None

    # -- driver ------------------------------------------------------------
    def run(self):
        try:
            self._run()
        finally:
            self._release_lock()

    def _run(self):
        self.emit("run_start", run_id=self.run_id, subject=self.sid,
                   ct_path=str(self.ct_path), split=self.split)
        ok_all = True
        for name, fn in (("ct", self.do_ct), ("masks", self.do_masks), ("mesh", self.do_mesh),
                         ("gbuffer", self.do_gbuffer), ("buffers", self.do_buffers),
                         ("infer", self.do_infer), ("recon", self.do_recon)):
            ok = self._run_stage(name, fn)
            if not ok:
                # recon is a bonus visualisation, not part of the neural-rendering result:
                # its failure must not turn an otherwise-successful run into "error".
                if name != "recon":
                    ok_all = False
                break
        self.status = "done" if ok_all else "error"
        total = time.time() - self.t0
        self.emit("run_done", seconds_total=round(total, 3),
                   timings={k: round(v, 3) for k, v in self.timings.items()})
        self._write_run_json(total)
        self._close_subscribers()

    def _run_stage(self, name, fn) -> bool:
        self.emit("stage_start", stage=name)
        t0 = time.time()
        try:
            summary = fn() or {}
            dt = time.time() - t0
            self.timings[name] = dt
            self.emit("stage_done", stage=name, seconds=round(dt, 3), summary=summary)
            return True
        except Exception as e:
            dt = time.time() - t0
            self.timings[name] = dt
            self.error = {"stage": name, "message": str(e), "traceback": traceback.format_exc()}
            self.emit("error", stage=name, message=str(e), hint=_HINTS.get(name, ""))
            return False

    def _write_run_json(self, total):
        doc = {
            "run_id": self.run_id, "subject": self.sid, "ct_path": str(self.ct_path),
            "split": self.split, "status": self.status,
            "seconds_total": round(total, 3),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "error": self.error,
        }
        (self.run_dir / "run.json").write_text(json.dumps(doc, indent=2))

    # -- stage: ct -----------------------------------------------------------
    def do_ct(self):
        result = inspect_ct(str(self.ct_path))
        self.split = result["info"]["split"]
        src_dir = Path(result["cache_dir"]) / "ct"
        dst_dir = self.run_dir / "ct"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            shutil.copy2(f, dst_dir / f.name)
        self._mid_idx = result["mid_idx"]
        self._sprite_idx = result["sprite_idx"]
        self._ct_shape = result["shape"]

        info = result["info"]
        for chk in result["checks"]:
            self.log("ct", f"[{chk['status']}] {chk['name']}: {chk['reason']}")
        self.stat("ct", "shape", "shape", info["shape"], "voxels", "text")
        self.stat("ct", "spacing", "spacing", info["spacing_mm"], "mm", "text")
        self.stat("ct", "extent", "extent", info["extent_cm"], "cm", "text")
        self.stat("ct", "hu_range", "HU range", [round(info["hu_min"]), round(info["hu_max"])], "HU", "text")
        self.stat("ct", "orientation", "orientation", info["orientation"], "", "text")
        self.stat("ct", "split", "split", info["split"], "", "text")
        self.stat("ct", "load_time", "load time", info["load_time_s"], "s", "float2")
        for axis, url in result["slice_urls"].items():
            self.artifact("ct", axis, url)
        self.artifact("ct", "axial_stack", result["axial_stack_url"],
                       meta={"n": result["sprite_n"], "w": 256, "h": 256})
        self.artifact("ct", "info", "ct/info.json")
        if not info["has_segmentations"]:
            raise RuntimeError(f"no segmentations/ next to {self.ct_path}")
        return {"sid": self.sid, "split": self.split, "n_tissues": info["n_tissues"]}

    # -- stage: masks ---------------------------------------------------------
    def do_masks(self):
        seg_dir = self.subject_root / "segmentations"
        img = nib.load(str(self.ct_path))
        zooms = img.header.get_zooms()[:3]
        voxel_ml = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0
        tp = glint_model.tissue_params()

        masks = {}
        present = [t for t in TISSUE_NAMES if (seg_dir / f"{t}.nii.gz").exists()]
        for i, name in enumerate(present):
            m_img = nib.load(str(seg_dir / f"{name}.nii.gz"))
            m_canon = nib.as_closest_canonical(m_img)
            data = np.asarray(m_canon.dataobj) > 0.5
            masks[name] = data
            self.progress("masks", i + 1, len(present), name)
        self._masks_cache = masks

        volumes = {}
        total_l = 0.0
        for name in TISSUE_NAMES:
            hexcol = tp.get(name, {}).get("hex", "#999999")
            if name in masks:
                vol_ml = float(masks[name].sum()) * voxel_ml
                total_l += vol_ml / 1000.0
                # A mask file can exist yet be empty (organ outside the scan's field of
                # view, e.g. lungs in an abdominal CT) — count it only if it holds tissue.
                volumes[name] = {"present": vol_ml >= 1.0, "volume_ml": round(vol_ml, 2), "color": hexcol,
                                 "display": viz.display_color(hexcol)}
            else:
                volumes[name] = {"present": False, "volume_ml": 0.0, "color": hexcol,
                                 "display": viz.display_color(hexcol)}
        out_dir = self.run_dir / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "volumes.json").write_text(json.dumps(volumes, indent=2))
        self.artifact("masks", "volumes", "masks/volumes.json")

        if masks:
            label_vol = viz.build_label_volume(masks, TISSUE_NAMES)
            lut = viz.build_lut(TISSUE_NAMES, {n: viz.display_color(v["hex"]) for n, v in tp.items()})
            windowed = viz.hu_window(np.asarray(nib.as_closest_canonical(img).dataobj, dtype=np.float32))
            mid = self._mid_idx or {"axial": label_vol.shape[2] // 2,
                                     "coronal": label_vol.shape[1] // 2,
                                     "sagittal": label_vol.shape[0] // 2}
            for axis in ("axial", "coronal", "sagittal"):
                gray = viz.gray_to_rgb(viz.slice2d(windowed, axis, mid[axis]))
                lab = viz.slice2d(label_vol, axis, mid[axis])
                over = viz.overlay(gray, lab, lut)
                viz.save_png(out_dir / f"overlay_{axis}.png", viz.resize_rgb(over, 512))
                self.artifact("masks", f"overlay_{axis}", f"masks/overlay_{axis}.png")

            sprite_idx = self._sprite_idx or list(range(0, label_vol.shape[2],
                                                         max(1, label_vol.shape[2] // 64)))
            tiles = []
            for k in sprite_idx:
                gray = viz.gray_to_rgb(viz.slice2d(windowed, "axial", k))
                lab = viz.slice2d(label_vol, "axial", k)
                tiles.append(viz.overlay(gray, lab, lut))
            viz.save_jpg(out_dir / "overlay_stack.jpg", viz.vertical_strip(tiles, 256), 85)
            self.artifact("masks", "overlay_stack", "masks/overlay_stack.jpg",
                           meta={"n": len(sprite_idx), "w": 256, "h": 256})

        n_in_scan = sum(1 for v in volumes.values() if v["present"])
        self.stat("masks", "found", "organs in scan", f"{n_in_scan}/{len(TISSUE_NAMES)}", "", "text")
        self.stat("masks", "total_volume", "total volume", round(total_l, 2), "L", "float2")
        return {"tissues_found": len(present), "total_volume_l": round(total_l, 2)}

    # -- stage: mesh ------------------------------------------------------------
    def do_mesh(self):
        seg_dir = self.subject_root / "segmentations"
        img = nib.load(str(self.ct_path))
        zooms = img.header.get_zooms()[:3]
        tp = glint_model.tissue_params()

        out_dir = self.run_dir / "meshes" / self.sid
        out_dir.mkdir(parents=True, exist_ok=True)

        present = [t for t in TISSUE_NAMES if (seg_dir / f"{t}.nii.gz").exists()]
        # Web meshes are written per organ as soon as it exists, so the viewer can pop
        # organs in live. They are centred on the CT volume centre (OBJ coordinates are
        # voxel index x spacing from 0, so that centre is known before any organ); each
        # organ also carries its bbox so the viewer can frame the union seen so far.
        center = np.asarray(img.shape[:3], np.float64) * np.asarray(zooms, np.float64) / 2.0
        web_dir = self.run_dir / "web" / "meshes"
        web_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"center_mm": center.tolist(), "organs": []}
        n_organs = total_v = total_t = 0
        for i, name in enumerate(present):
            t0 = time.time()
            mesh = extract_mesh(seg_dir / f"{name}.nii.gz", zooms)
            ms = (time.time() - t0) * 1000.0
            if mesh is None:
                self.log("mesh", f"{name}: empty, skipped")
                self.progress("mesh", i + 1, len(present), name)
                continue
            save_obj(mesh, out_dir / f"{name}.obj")
            tri = mesh.triangulate()             # save_obj writes exactly these points/faces
            n_v, n_t = int(tri.n_points), int(tri.n_cells)
            n_organs += 1
            total_v += n_v
            total_t += n_t

            dm = viz.decimate_for_web(mesh, max_tris=15000)
            pts = np.asarray(dm.points, dtype=np.float32)
            faces = dm.faces.reshape(-1, 4)[:, 1:].astype(np.uint32).ravel()
            nV, nT = viz.write_web_mesh_bin(web_dir / f"{name}.bin", pts,
                                            np.asarray(dm.point_normals, dtype=np.float32), faces, center)
            hexcol = tp.get(name, {}).get("hex", "#999999")
            entry = {"name": name, "color": hexcol, "verts": n_v, "tris": n_t, "ms": round(ms, 1),
                     "web": f"/runs/{self.run_id}/web/meshes/{name}.bin", "web_verts": int(nV), "web_tris": int(nT),
                     "bbox_min": (pts.min(axis=0) - center).round(2).tolist(),
                     "bbox_max": (pts.max(axis=0) - center).round(2).tolist()}
            manifest["organs"].append(entry)
            self.artifact("mesh", "organ", f"meshes/{self.sid}/{name}.obj", meta=entry)
            self.progress("mesh", i + 1, len(present), name)

        (self.run_dir / "web" / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.artifact("mesh", "manifest", "web/manifest.json")

        self.stat("mesh", "organs", "organs", n_organs, "", "int")
        self.stat("mesh", "verts", "total vertices", total_v, "", "int")
        self.stat("mesh", "tris", "total triangles", total_t, "", "int")
        return {"organs": n_organs, "verts": total_v, "tris": total_t}

    # -- stage: gbuffer (Blender subprocess) -------------------------------
    def do_gbuffer(self):
        mesh_dir = self.run_dir / "meshes"
        out_dir = self.run_dir / "gbuffers"
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = self.run_dir / "preview"

        blender = cfg.BLENDER
        if not blender or not Path(blender).exists():
            import shutil as _sh
            blender = _sh.which("blender")
        if not blender:
            raise RuntimeError("Blender executable not found")

        # Blender block-buffers stdout into a pipe, so every "View [k/20]" line arrived at
        # the very end (0.0 s/view, no live progress). Line-buffer it before the script runs.
        cmd = [blender, "--background",
               "--python-expr", "import sys; sys.stdout.reconfigure(line_buffering=True)",
               "--python", str(cfg.GENERATE_SCRIPT), "--",
               "--subject", self.sid, "--dataset", str(self.subject_root.parent),
               "--mesh_dir", str(mesh_dir), "--output_dir", str(out_dir),
               "--tex_dir", str(cfg.TEX_DIR), "--size", str(cfg.GBUFFER_SIZE),
               "--gbuffer_only"]
        self.log("gbuffer", " ".join(cmd))

        proc = subprocess.Popen(cmd, cwd=str(cfg.REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)

        subj_dir = out_dir / self.sid
        self._gbuf_subject_dir = subj_dir
        view_re = re.compile(r"--- View \[(\d+)/(\d+)\] (\S+)")
        bbox_re = re.compile(r"organ bbox: (.+)")
        sampling_re = re.compile(r"sampling\s*: (.+)")
        n_total = 20
        view_start_t = {}
        view_durations = []
        current_view = None
        pid_lut = None

        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            m = view_re.search(line)
            if m:
                view_id, n_total_s, label = int(m.group(1)), int(m.group(2)), m.group(3)
                n_total = n_total_s
                current_view = label
                view_start_t[label] = time.time()
                self.progress("gbuffer", view_id - 1, n_total, label)
                continue
            m = bbox_re.search(line)
            if m:
                self.log("gbuffer", f"organ bbox: {m.group(1)}")
                continue
            m = sampling_re.search(line)
            if m:
                self.log("gbuffer", f"sampling: {m.group(1)}")
                mm_px_m = re.search(r"([\d.]+)\s*mm/px", m.group(1))
                if mm_px_m:
                    self.stat("gbuffer", "mm_per_px", "mm/px", float(mm_px_m.group(1)), "mm", "float2")
                continue
            if "gbuffers →" in line and current_view is not None:
                dt = time.time() - view_start_t.get(current_view, time.time())
                view_durations.append(dt)
                view_dir = subj_dir / current_view
                try:
                    pid_lut = self._make_gbuffer_previews(view_dir, preview_dir / current_view, pid_lut)
                except Exception as e:
                    self.log("gbuffer", f"[warn] preview failed for {current_view}: {e}")
                done_n = len(view_durations)
                self.progress("gbuffer", done_n, n_total, current_view)
                self._emit_cameras(subj_dir, out_dir)
                current_view = None
                continue
            if line.startswith("[dataset-gen]") or line.startswith("ERROR") or "[warn]" in line:
                self.log("gbuffer", line)

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Blender exited with code {ret}")

        n_views = len(view_durations)
        mean_s = float(np.mean(view_durations)) if view_durations else 0.0
        self.stat("gbuffer", "views", "views", n_views, "", "int")
        self.stat("gbuffer", "s_per_view", "s/view (mean)", round(mean_s, 2), "s", "float2")
        self.stat("gbuffer", "resolution", "resolution", cfg.GBUFFER_SIZE, "px", "int")
        return {"views": n_views, "s_per_view": round(mean_s, 2)}

    def _make_gbuffer_previews(self, view_dir: Path, preview_dir: Path, pid_lut):
        segid = read_render_exr(str(view_dir / "segid.exr"), want=("image",))["image"][..., 0]
        pid = np.rint(segid).astype(np.int64)
        fg = pid > 0
        if pid_lut is None:
            pid_lut = self._build_pid_lut(view_dir.parent)
        segid_rgb = viz.apply_lut(pid, pid_lut)

        depth = read_render_exr(str(view_dir / "depth.exr"), want=("image",))["image"][..., 0]
        depth_rgb = viz.depth_to_pastel(depth, fg)

        normals = read_render_exr(str(view_dir / "normals.exr"), want=("image",))["image"][..., :3]
        normals_rgb = viz.normals_to_rgb(normals, fg)

        seg_bgr = cv2.imread(str(view_dir / "seg.png"), cv2.IMREAD_COLOR)
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

        view = view_dir.name
        for key, arr in (("seg", seg_rgb), ("depth", depth_rgb), ("normals", normals_rgb),
                         ("segid", segid_rgb)):
            out = viz.resize_rgb(arr, 512)
            viz.save_png(preview_dir / f"{key}.png", out)
            rel = str((preview_dir / f"{key}.png").relative_to(self.run_dir))
            self.artifact("gbuffer", key, rel, view=view)
        return pid_lut

    def _build_pid_lut(self, subj_dir: Path):
        lut_gid = infer_mod.build_segid_lut(subj_dir)  # pid -> global id (float)
        tp = glint_model.tissue_params()
        from dataset import CANONICAL_TISSUES
        max_pid = len(lut_gid) - 1
        rgb = np.zeros((max_pid + 1, 3), dtype=np.uint8)
        rgb[0] = viz.PLUM
        for pid in range(1, max_pid + 1):
            gid = int(round(lut_gid[pid]))
            if 1 <= gid <= len(CANONICAL_TISSUES):
                name = CANONICAL_TISSUES[gid - 1]
                rgb[pid] = viz.hex_to_rgb(tp.get(name, {}).get("hex", "#969696"))
            else:
                rgb[pid] = (150, 150, 150)
        return rgb

    def _emit_cameras(self, subj_dir: Path, out_dir: Path):
        cams = []
        organ_center = None
        radius = None
        for vdir in sorted(subj_dir.glob("v*")):
            mp = vdir / "meta.json"
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            cams.append({"view": vdir.name, "cam_pos": meta.get("camera_pos"),
                        "target": meta.get("target_pos"), "az": meta.get("azimuth_actual"),
                        "el": meta.get("elevation_actual")})
            organ_center = meta.get("organ_center", organ_center)
            radius = meta.get("radius", radius)
        doc = {"cameras": cams, "organ_center": organ_center, "radius": radius}
        (out_dir / "cameras.json").write_text(json.dumps(doc, indent=2))
        self.artifact("gbuffer", "cameras", f"gbuffers/cameras.json")

    # -- stage: buffers (subprocess: demo/buffer_worker.py) ------------------
    def do_buffers(self):
        subj_dir = self._gbuf_subject_dir or (self.run_dir / "gbuffers" / self.sid)
        views = sorted(p for p in subj_dir.glob("v*") if (p / "seg.png").exists())
        stack_dir = self.run_dir / "stack"
        preview_dir = self.run_dir / "preview"
        stack_dir.mkdir(parents=True, exist_ok=True)

        jobs = [[str(vdir), str(subj_dir), str(stack_dir / f"{vdir.name}.npy"),
                 str(preview_dir / vdir.name), cfg.PRESET, str(cfg.TEX_DIR)] for vdir in views]
        jobs_path = self.run_dir / "buffer_jobs.json"
        jobs_path.write_text(json.dumps(jobs))

        # Its own process (see buffer_worker.py): bounded memory, freed when it exits.
        workers = cfg.buffer_workers()
        self.log("buffers", f"{len(jobs)} views on {workers} worker(s)")
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "buffer_worker.py"),
               str(jobs_path), "--workers", str(workers)]
        proc = subprocess.Popen(cmd, cwd=str(cfg.REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        tex_times, glint_times = [], []
        n_done, n_failed = 0, 0
        for line in proc.stdout:
            try:
                res = json.loads(line)
            except ValueError:
                continue                     # stray library output
            view = res["view"]
            n_done += 1
            if "error" in res:
                n_failed += 1
                self.log("buffers", f"[warn] {view} failed: {res['error']}")
            else:
                tex_times.append(res["texture_ms"])
                glint_times.append(res["glint_ms"])
                for key in ("albedo", "nmap", "rough", "glint", "highlight"):
                    self.artifact("buffers", key, f"preview/{view}/{key}.png", view=view)
            self.progress("buffers", n_done, len(jobs), view)
        err = proc.stderr.read()
        if proc.wait() != 0 or n_failed == len(jobs):
            tail = "\n".join(err.strip().splitlines()[-5:])
            raise RuntimeError(f"buffer worker failed (exit {proc.returncode}): {tail}")

        channels = [("seg_rgb", 3), ("depth", 1), ("normals", 3), ("segid", 1),
                   ("texture", 7), ("glint", 2)]
        self.stat("buffers", "channels", "channels", sum(c for _, c in channels), "", "int")
        self.log("buffers", "groups: " + ", ".join(f"{n}={c}ch" for n, c in channels))
        if tex_times:
            self.stat("buffers", "texture_ms", "ms/view texture", round(float(np.mean(tex_times)), 1), "ms", "float1")
        if glint_times:
            self.stat("buffers", "glint_ms", "ms/view glint", round(float(np.mean(glint_times)), 1), "ms", "float1")
        return {"views": n_done, "channels": sum(c for _, c in channels)}

    # -- stage: infer ---------------------------------------------------------
    def do_infer(self):
        import torch
        if MODEL.status != "ready":
            raise RuntimeError(f"model not ready (status={MODEL.status})")
        G, device, gcfg = MODEL.G, MODEL.device, MODEL.gcfg

        stack_dir = self.run_dir / "stack"
        out_dir = self.run_dir / "output"
        thumb_dir = out_dir / "thumb"
        gt_dir = self.run_dir / "gt"
        out_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        stacks = sorted(stack_dir.glob("v*.npy"))
        times_ms = []
        gt_root = cfg.GT_ROOTS[0] / self.sid if cfg.GT_ROOTS else None
        for i, sp in enumerate(stacks):
            view = sp.stem
            x = np.load(sp).astype(np.float32)
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            _sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                fake = G(t)
            _sync(device)                    # MPS/CUDA are async: time the kernels, not the enqueue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)
            img = ((fake[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
            img = img.transpose(1, 2, 0)  # HWC RGB

            viz.save_png(out_dir / f"{view}.png", img)
            viz.save_jpg(thumb_dir / f"{view}.jpg", viz.resize_rgb(img, 320), 88)
            self.artifact("infer", "output", f"output/{view}.png", view=view)
            self.artifact("infer", "thumb", f"output/thumb/{view}.jpg", view=view)
            self.progress("infer", i + 1, len(stacks), view)

            if gt_root is not None:
                ref_dir = gt_root / view
                ref_png = ref_dir / "rgb_preview.png"
                ref_seg = ref_dir / "seg.png"
                our_seg = (self.run_dir / "gbuffers" / self.sid / view / "seg.png")
                if ref_png.exists() and ref_seg.exists() and our_seg.exists():
                    a = cv2.imread(str(ref_seg), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                    b = cv2.imread(str(our_seg), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                    if a.shape == b.shape:
                        diff = np.abs(a - b)
                        mad = float(diff.mean())
                        pct = float((diff.max(axis=-1) > (2.0 / 255.0)).mean())
                        if mad < cfg.GT_MEAN_ABS_DIFF_MAX and pct < cfg.GT_PCT_DIFFERING_MAX:
                            gt_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(ref_png, gt_dir / f"{view}.png")
                            self.artifact("infer", "gt", f"gt/{view}.png", view=view)
                        else:
                            self.log("infer", f"{view}: GT hidden — seg mismatch "
                                              f"(mad={mad:.4f}, pct={pct:.4f})")
                    else:
                        self.log("infer", f"{view}: GT hidden — seg shape mismatch")
                else:
                    self.log("infer", f"{view}: no GT reference available")

        mean_ms = float(np.mean(times_ms[1:])) if len(times_ms) > 1 else (times_ms[0] if times_ms else 0.0)
        self.stat("infer", "ms_per_view", "ms/view", round(mean_ms, 1), "ms", "float1")
        self.stat("infer", "device", "device", str(device), "", "text")
        self.stat("infer", "params", "params", round(MODEL.params_m, 1) if MODEL.params_m else None, "M", "float1")
        return {"views": len(stacks), "ms_per_view": round(mean_ms, 1)}

    # -- stage: recon (3-D view — PLAN_3D.md §4.2/§4.3) ------------------------
    def do_recon(self):
        def emit(kind, **kw):
            if kind == "progress":
                self.progress("recon", kw["done"], kw["total"], kw["label"])
            elif kind == "stat":
                self.stat("recon", kw["key"], kw["label"], kw["value"], kw.get("unit", ""), kw.get("fmt", "text"))
            elif kind == "artifact":
                self.artifact("recon", kw["key"], kw["url"], kw.get("view"), kw.get("meta"))
            elif kind == "log":
                self.log("recon", kw["line"])

        stats = recon.build(self.run_dir, self.sid, emit=emit)
        return {"vertices": stats["vertices"], "seen_frac": stats.get("seen_frac_area_weighted")}


# ── run listing (for /api/runs) ──────────────────────────────────────────────
def list_runs():
    runs = []
    if not cfg.RUNS.is_dir():
        return runs
    for d in sorted(cfg.RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.name.startswith("_inspect_") or not d.is_dir():
            continue
        rj = d / "run.json"
        if not rj.exists():
            continue
        try:
            doc = json.loads(rj.read_text())
        except Exception:
            continue
        runs.append({"run_id": doc.get("run_id", d.name), "subject": doc.get("subject"),
                    "status": doc.get("status"), "seconds_total": doc.get("seconds_total"),
                    "date": datetime.fromtimestamp(d.stat().st_mtime).isoformat()})
    return runs
