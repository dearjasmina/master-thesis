"""
split_training_dataset.py — Create subject-level train/val/test splits.

Splits must happen at the SUBJECT level (not sample level). If you split by
individual sample dirs, views of the same patient appear in both train and test,
the network memorises that anatomy, and your test metrics are meaningless.

Input:  data/training_dataset/  (or --dataset_dir)
Output: data/training_dataset/splits.json

splits.json structure:
  {
    "train": ["s0001", "s0050", ...],    # subject IDs
    "val":   ["s0100", ...],
    "test":  ["s0200", ...],
    "train_samples": ["s0001_v01_az-45_el-20", ...],  # all sample dirs in train
    "val_samples":   [...],
    "test_samples":  [...],
  }

Usage:
    python scripts/split_training_dataset.py
    python scripts/split_training_dataset.py --dataset_dir data/training_dataset --ratio 80 10 10 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/training_dataset")
    ap.add_argument("--ratio",  nargs=3, type=int, default=[80, 10, 10],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Split ratio in integer percentages, must sum to 100")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert sum(args.ratio) == 100, f"Ratio must sum to 100, got {args.ratio}"
    train_r, val_r, test_r = [r / 100 for r in args.ratio]

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset_dir not found: {dataset_dir}")
        return

    # Collect all sample directories (name pattern: {subject}_v{N}_az{az}_el{el})
    all_sample_dirs = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and "_v" in d.name
    ])

    if not all_sample_dirs:
        print(f"No sample directories found in {dataset_dir}")
        print("Run generate_training_dataset.py for at least one subject first.")
        return

    # Group samples by subject prefix (everything before the first "_v{N:02d}")
    subject_to_samples: dict[str, list[str]] = {}
    for sample in all_sample_dirs:
        # e.g. "s0050_v01_az-45_el-20" → subject "s0050"
        subject = sample.split("_v")[0]
        subject_to_samples.setdefault(subject, []).append(sample)

    all_subjects = sorted(subject_to_samples.keys())
    n = len(all_subjects)

    rng = random.Random(args.seed)
    shuffled = all_subjects.copy()
    rng.shuffle(shuffled)

    n_val  = max(1, round(n * val_r))
    n_test = max(1, round(n * test_r))
    n_train = n - n_val - n_test

    if n_train < 1:
        print(f"ERROR: Only {n} subjects — not enough for a meaningful split.")
        print("Generate data for more subjects before splitting.")
        return

    train_subjects = sorted(shuffled[:n_train])
    val_subjects   = sorted(shuffled[n_train:n_train + n_val])
    test_subjects  = sorted(shuffled[n_train + n_val:])

    # Verify no overlap
    assert not (set(train_subjects) & set(val_subjects)),  "Train/val overlap!"
    assert not (set(train_subjects) & set(test_subjects)), "Train/test overlap!"
    assert not (set(val_subjects)   & set(test_subjects)), "Val/test overlap!"

    def flatten(subjects):
        return [s for subj in subjects for s in subject_to_samples[subj]]

    splits = {
        "train":          train_subjects,
        "val":            val_subjects,
        "test":           test_subjects,
        "train_samples":  flatten(train_subjects),
        "val_samples":    flatten(val_subjects),
        "test_samples":   flatten(test_subjects),
        "meta": {
            "seed":           args.seed,
            "ratio":          args.ratio,
            "total_subjects": n,
            "total_samples":  len(all_sample_dirs),
            "views_per_subject": 15,
        },
    }

    out_path = dataset_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    total_s = len(all_sample_dirs)
    print(f"\nSplit saved → {out_path}")
    print(f"\n  Subjects:    {n} total")
    print(f"  train: {len(train_subjects):>4} subjects  ({len(splits['train_samples']):>5} samples)")
    print(f"  val:   {len(val_subjects):>4} subjects  ({len(splits['val_samples']):>5} samples)")
    print(f"  test:  {len(test_subjects):>4} subjects  ({len(splits['test_samples']):>5} samples)")
    print(f"  total: {n:>4} subjects  ({total_s:>5} samples)")
    print(f"\n  Actual split: "
          f"{len(train_subjects)/n*100:.1f}% / "
          f"{len(val_subjects)/n*100:.1f}% / "
          f"{len(test_subjects)/n*100:.1f}%")


if __name__ == "__main__":
    main()
