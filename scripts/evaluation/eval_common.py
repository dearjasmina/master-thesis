"""
eval_common.py — shared loading/plumbing for the evaluation suite.

Every script in scripts/evaluation/ uses this so they agree on how the checkpoint is
loaded, how the dataset is opened, and how results are written. Keeping it in one place
means a change to the model or config surface breaks loudly in one file rather than
silently diverging between metrics.

The suite is deliberately additive: it does not modify scripts/training/evaluate.py,
which already produces PSNR/SSIM/LPIPS. These add what a thesis on replacing path
tracing actually needs and that file does not have:

    eval_fid_kid.py      distribution-level realism (FID / KID)
    eval_multiview.py    cross-view consistency — implements the placeholder gate
    eval_per_tissue.py   which ORGANS are learned well, via the IndexOB pass
    eval_speed.py        the headline claim: network ms/frame vs Cycles s/frame
    eval_report.py       aggregates everything with bootstrap CIs

Run scripts/training/evaluate.py first for the baseline metrics; these complement it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# The training code is a flat package; import it the same way train.py/evaluate.py do.
_TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from config import preset, PRESET_NAMES        # noqa: E402
from dataset import PairedRenderDataset        # noqa: E402
from networks import define_G                  # noqa: E402


def add_common_args(ap):
    """Arguments every evaluation script shares."""
    ap.add_argument("--preset", default="full1024",
                    choices=PRESET_NAMES)
    ap.add_argument("--checkpoint", required=True,
                    help="e.g. results/training_runs/full1024/checkpoints/latest.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--target", default=None,
                    choices=["preview_png", "exr_agx", "exr_linear"])
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="EVALUATION/suite",
                    help="output directory for CSV/JSON/figures")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return ap


def build_cfg(args):
    cfg = preset(args.preset)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.target:
        cfg.data.target = args.target
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",")
                                  if b.strip()]
    return cfg


def pick_device(arg=None):
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(cfg, checkpoint, device):
    """Load G exactly as scripts/training/evaluate.py does, so results are comparable."""
    G = define_G(cfg).to(device).eval()
    ck = torch.load(checkpoint, map_location=device)
    state = ck["G"] if isinstance(ck, dict) and "G" in ck else ck
    G.load_state_dict(state)
    n_par = sum(p.numel() for p in G.parameters())
    print(f"  generator: {n_par/1e6:.1f} M parameters   checkpoint: {checkpoint}")
    return G


def open_dataset(cfg, split):
    ds = PairedRenderDataset(cfg, split=split)
    print(f"  dataset: split={split}  n={len(ds)}  target={cfg.data.target}")
    return ds


def to01(t):
    """CxHxW in [-1,1] (torch) -> HxWxC in [0,1] (numpy float32)."""
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).numpy().transpose(1, 2, 0)


def sample_indices(n, max_samples, seed=0):
    """Deterministic evenly-spread subset, so repeated runs are comparable."""
    if max_samples <= 0 or max_samples >= n:
        return list(range(n))
    return list(np.linspace(0, n - 1, max_samples).astype(int))


def view_dir_of(ds, idx):
    """Directory on disk backing sample `idx`, for reading meta.json / render.exr.

    PairedRenderDataset stores its sample list under one of a few attribute names
    depending on version; probe rather than assume, and fail loudly if none match —
    a silently wrong path would produce confident nonsense.
    """
    for attr in ("samples", "views", "items", "records", "paths"):
        seq = getattr(ds, attr, None)
        if seq is None:
            continue
        rec = seq[idx]
        if isinstance(rec, (str, Path)):
            return Path(rec)
        if isinstance(rec, dict):
            for k in ("dir", "path", "view_dir", "root"):
                if k in rec:
                    return Path(rec[k])
        if isinstance(rec, (tuple, list)) and rec:
            return Path(rec[0])
    raise RuntimeError(
        "could not locate the on-disk directory for a dataset sample. "
        "Inspect PairedRenderDataset and add its attribute name to view_dir_of()."
    )


def read_meta(view_dir):
    f = Path(view_dir) / "meta.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean.

    Reported instead of a bare mean because per-view metrics are correlated within a
    subject and far from normal; a mean without an interval invites over-reading small
    differences between models.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def ensure_out(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
