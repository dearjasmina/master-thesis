"""
Split preprocessed CQ500 volumes into train / val / test at the VOLUME level.

Why volume-level (not image-level)?
  Each volume produces 64 rendered views. If you split by image, the same patient's
  views could appear in both train and test — the model memorises that patient and
  you get falsely inflated test scores. Splitting by volume guarantees the model
  has never seen any view of a test patient during training.

Ratio: 70 / 15 / 15  |  Random seed: 42 (fixed for reproducibility)

Usage:
    python scripts/split_dataset.py --input data/processed/ --output data/splits.json
"""

import argparse
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Train/val/test split at volume level")
    parser.add_argument("--input",  required=True, help="Directory of preprocessed .nii.gz files")
    parser.add_argument("--output", required=True, help="Output path for splits.json")
    args = parser.parse_args()

    processed_dir = Path(args.input)
    output_path   = Path(args.output)

    # Collect all volume IDs (filename without .nii.gz), skip strays
    all_ids = sorted([
        f.name.replace(".nii.gz", "")
        for f in processed_dir.glob("*.nii.gz")
        if f.name.startswith("CQ500")   # skip Unknown Study.nii.gz etc.
    ])

    print(f"Found {len(all_ids)} volumes")

    # First split off test (15%), then split remainder into train/val (82%/18% ≈ 70/15 of total)
    train_val, test  = train_test_split(all_ids, test_size=0.15, random_state=42)
    train,     val   = train_test_split(train_val, test_size=0.176, random_state=42)
    # 0.176 of 85% ≈ 15% of total → gives us ~70/15/15

    # Sanity checks — these must never fail
    assert len(set(train) & set(val))  == 0, "Overlap between train and val!"
    assert len(set(train) & set(test)) == 0, "Overlap between train and test!"
    assert len(set(val)   & set(test)) == 0, "Overlap between val and test!"
    assert len(train) + len(val) + len(test) == len(all_ids), "Missing volumes!"

    splits = {"train": train, "val": val, "test": test}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    total = len(all_ids)
    print(f"\nSplit saved → {output_path}")
    print(f"  train : {len(train):>3} volumes  ({len(train)/total*100:.1f}%)")
    print(f"  val   : {len(val):>3} volumes  ({len(val)/total*100:.1f}%)")
    print(f"  test  : {len(test):>3} volumes  ({len(test)/total*100:.1f}%)")
    print(f"  total : {total}")
    print(f"\n  Estimated render pairs (×64 views each):")
    print(f"  train : {len(train)*64:>6} image pairs")
    print(f"  val   : {len(val)*64:>6} image pairs")
    print(f"  test  : {len(test)*64:>6} image pairs")


if __name__ == "__main__":
    main()
