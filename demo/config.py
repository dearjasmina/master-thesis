"""
config.py — every path/constant the demo backend needs, in one place.

Nothing here re-implements thesis code; it only points at it. See demo/PLAN.md §1.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Reused thesis code lives under these dirs — added to sys.path by pipeline.py so
# `import infer`, `import config as train_config`, etc. resolve without copying code.
TRAINING_DIR = REPO / "scripts" / "training"
TRAINING_DATASET_DIR = REPO / "scripts" / "training_dataset"

# Each path may be overridden by an environment variable, so the same code runs on this
# laptop and on the GPU server without editing the file.
import os

CKPT = Path(os.environ.get("DEMO_CKPT", REPO / "results/training_runs/gbuf_glint/checkpoints/epoch_060.pt"))
PRESET = os.environ.get("DEMO_PRESET", "full1024_glint")
TEX_DIR = Path(os.environ.get("DEMO_TEX_DIR", REPO / "data/renders/textures/tissue_server"))
GT_ROOTS = [Path(p) for p in os.environ.get("DEMO_GT_ROOTS", str(REPO / "data/training_dataset_v25")).split(os.pathsep)]

_blender_default = os.environ.get("DEMO_BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
BLENDER = _blender_default if Path(_blender_default).exists() else (shutil.which("blender") or _blender_default)

BROWSE_HOME = Path("/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
RUNS = REPO / "demo/runs"
RUNS.mkdir(parents=True, exist_ok=True)

GENERATE_SCRIPT = TRAINING_DATASET_DIR / "generate_training_dataset.py"

SERVER_REF = {
    "cycles_s_per_frame": 53.025,
    "cycles_where": "server, Cycles (supplied)",
    "net_ms_per_frame_l4": 98.3,
}

# 29-tissue extraction order — must match extract_meshes.TISSUE_NAMES (imported directly
# where needed; kept here only as a doc pointer, not duplicated).

HOST = "127.0.0.1"
PORT = 8765

# Blender render size for the gbuffer stage (§3.5).
GBUFFER_SIZE = 1024

# ── Memory budget ───────────────────────────────────────────────────────────
# This is a 16 GB laptop. Measured peaks: one `buffers` worker ~1.0 GB (infer.load_input
# at 1024 px); the generator on MPS ~0.8 GB; Blender (G-buffers only) ~1-2 GB. A run is
# refused below MIN_FREE_GB, and the worker count follows what is actually free.
MIN_FREE_GB = 3.0
BUFFER_WORKER_GB = 1.5        # measured 1.0 GB + headroom
BUFFER_WORKERS_MAX = 2


def available_gb() -> float:
    """Free + inactive + speculative pages from vm_stat (what macOS can hand out now)."""
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return float("inf")          # unknown → do not block
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1))
                for k in ("Pages free", "Pages inactive", "Pages speculative"))
    return pages * page / 1e9


def buffer_workers() -> int:
    """1-2 workers, never more than free memory allows (leaves MIN_FREE_GB spare)."""
    spare = available_gb() - MIN_FREE_GB
    return int(max(1, min(BUFFER_WORKERS_MAX, spare // BUFFER_WORKER_GB)))

# GT alignment guard thresholds (§3.7).
GT_MEAN_ABS_DIFF_MAX = 2.0 / 255.0
GT_PCT_DIFFERING_MAX = 0.01

# splits._bucket params (mirrors scripts/training/config.py DataConfig defaults).
SPLIT_SEED = 1337
SPLIT_VAL_FRAC = 0.075
SPLIT_TEST_FRAC = 0.075
