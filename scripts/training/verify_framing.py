"""
verify_framing.py — Numerically check camera framing from meta.json (no rendering, no pixels).

For each view, the organ bounding-sphere fills a fraction of the frame equal to
    fill = 2 * arcsin(radius / camera_distance) / FOV
using values stored in meta.json (radius, camera_distance, fov_deg). This is exact
geometry, far more reliable than seg-pixel heuristics.

Use it to:
  - confirm the NEW mesh-bbox camera frames consistently (fill ≈ --fill for ALL views,
    low variance) on your 10 validation subjects BEFORE the full re-render;
  - quantify the OLD data (CT-volume framing shows wild fill / high variance).

    python scripts/training/verify_framing.py --render-dir /mnt/data/$USER/runs/reframe_test
"""
from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np


def fill_fraction(meta: dict) -> float:
    r = float(meta.get("radius", 0.0))
    d = float(meta.get("camera_distance", 0.0))
    fov = math.radians(float(meta.get("fov_deg", 0.0)))
    if d <= 0 or fov <= 0:
        return float("nan")
    ratio = min(max(r / d, 0.0), 1.0)        # clamp to arcsin domain
    return 2.0 * math.asin(ratio) / fov      # sphere angular diameter / FOV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", default="data/training_dataset",
                    help="dir of {subject}/v*/meta.json to check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lo", type=float, default=0.40, help="flag below (too zoomed out / tiny)")
    ap.add_argument("--hi", type=float, default=1.00, help="flag above (sphere exceeds frame → clip risk)")
    args = ap.parse_args()

    root = Path(args.render_dir)
    metas = sorted(root.glob("*/v*/meta.json"))
    if not metas:
        print(f"no */v*/meta.json under {root.resolve()}"); return

    rows, per_subj = [], {}
    for mp in metas:
        try:
            meta = json.load(open(mp))
        except Exception:
            continue
        subj = mp.parent.parent.name
        f = fill_fraction(meta)
        flag = "ok"
        if math.isnan(f):      flag = "no-data"
        elif f < args.lo:      flag = "TOO_SMALL"
        elif f > args.hi:      flag = "TOO_BIG"
        rows.append({"subject": subj, "view": mp.parent.name, "fill": round(f, 4), "flag": flag})
        per_subj.setdefault(subj, []).append(f)

    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "framing_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "view", "fill", "flag"])
        w.writeheader(); w.writerows(rows)

    fills = np.array([r["fill"] for r in rows if not math.isnan(r["fill"])])
    nbad = sum(1 for r in rows if r["flag"] in ("TOO_SMALL", "TOO_BIG"))
    print(f"\n[framing-check] {len(rows)} views | {len(per_subj)} subjects")
    print(f"  fill fraction: mean={fills.mean():.3f}  std={fills.std():.3f}  "
          f"min={fills.min():.3f}  max={fills.max():.3f}")
    print(f"  flagged (fill<{args.lo} or >{args.hi}): {nbad} ({100*nbad/max(len(rows),1):.1f}%)")
    print("  per-subject mean fill (worst spread first):")
    spread = sorted(per_subj.items(), key=lambda kv: -(max(kv[1]) - min(kv[1])))
    for s, fs in spread[:12]:
        fa = np.array(fs)
        print(f"    {s}: mean={fa.mean():.3f} min={fa.min():.3f} max={fa.max():.3f} (n={len(fs)})")
    print(f"  → {out_dir/'framing_check.csv'}")
    print("\n  GOOD = mean ≈ your --fill, low std, ~0 flagged, tight per-subject spread.")


if __name__ == "__main__":
    main()
