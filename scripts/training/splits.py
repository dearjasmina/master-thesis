"""
splits.py — Volume-level train/val/test assignment.

Deterministic, hash-based, STABLE under dataset growth: each subject's bucket is a
pure function of its own id, so as you keep generating subjects the existing
train/val/test assignments never change (no val/test contamination over time).
This was chosen because (a) the dataset is still being generated, and (b) tissue
coverage is a non-issue — the rarest organ is in ~46% of subjects, so a plain
hash split already lands 0 tissues missing from val/test (see results/split_analysis).

Splits are BY VOLUME (subject): all views of a subject go to the same split, so the
network never sees 19 views of an organ at train and is tested on the 20th.

Low-tissue subjects are KEPT by default (min_tissues=0): they were confirmed to be
legitimate limited-FOV scans (coherent upper-chest / pelvis organ clusters), not
broken extractions. Raise cfg.data.min_tissues to exclude degenerate cases.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List

VALID_SPLITS = ("train", "val", "test")


def _bucket(subject: str, seed: int, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(f"{seed}:{subject}".encode()).hexdigest(), 16) % 10000 / 10000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def _tissue_count(root: Path, subject: str) -> int:
    p = root / subject / "tissue_ids.json"
    if p.exists():
        try:
            return len(json.load(open(p)).get("tissues", {}))
        except Exception:
            pass
    return 1_000  # unknown → keep


def assign_splits(subjects: List[str], cfg=None) -> Dict[str, str]:
    """Map each subject id → 'train' | 'val' | 'test' | 'exclude'."""
    if cfg is not None:
        d = cfg.data
        val_frac, test_frac = d.val_frac, d.test_frac
        seed, min_tissues, root = d.split_seed, d.min_tissues, Path(d.root)
    else:
        val_frac = test_frac = 0.075
        seed, min_tissues, root = 1337, 0, Path("data/training_dataset")

    out: Dict[str, str] = {}
    for s in sorted(subjects):
        if min_tissues > 0 and _tissue_count(root, s) < min_tissues:
            out[s] = "exclude"
        else:
            out[s] = _bucket(s, seed, val_frac, test_frac)
    return out
