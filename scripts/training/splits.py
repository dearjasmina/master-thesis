"""
splits.py — Train/val/test assignment.

>>> PLACEHOLDER <<<
Per the current request, data is NOT split yet. Every subject is assigned to
"train" so the full pipeline runs end-to-end on all data. Fill this in later.

When you implement it, split BY VOLUME (subject), never by view (3rd.md):
all 20 views of a held-out subject must go entirely to val or test, otherwise the
network sees 19 views of an organ at train time and the held-out 20th leaks.
Recommended split: 85 / 7.5 / 7.5 % of *subjects* (3rd.md), with a fixed seed.

A reference split already exists at scripts/training_dataset/split_training_dataset.py
and/or data/splits.json — wire that in here when ready.
"""
from __future__ import annotations

from typing import Dict, List


VALID_SPLITS = ("train", "val", "test")


def assign_splits(subjects: List[str], cfg=None) -> Dict[str, str]:
    """
    Map each subject id → split name.

    PLACEHOLDER: returns everything as "train".

    TODO (later): replace with a deterministic volume-level 85/7.5/7.5 split, e.g.

        import hashlib
        def bucket(s):
            h = int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000
            return "test" if h < 75 else "val" if h < 150 else "train"
        return {s: bucket(s) for s in subjects}

    or load from data/splits.json.
    """
    return {s: "train" for s in sorted(subjects)}
