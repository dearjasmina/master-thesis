"""
score_framing.py — Detect badly-framed views from seg.png (no EXR, no re-render).

The render camera is framed from the CT volume bbox, which doesn't transfer across
subjects → some views are too zoomed-in (organ clipped at the frame edge) or too
zoomed-out / off-centre (organ tiny). This scores EVERY view's framing from the flat
seg.png alone, so you can quantify the problem and exclude bad views without manually
inspecting 14k frames or re-rendering anything.

Per view it measures (organs are coloured, background is near-black):
  fg_frac   fraction of the frame that is foreground (organ)
  edge_frac fraction of the 1-pixel-wide border that is foreground (clipping indicator)
  off_ctr   distance of the foreground centroid from frame centre (0..~0.7)

A view is flagged BAD if:  fg_frac < min_fg  (tiny/off-screen)
                       or  fg_frac > max_fg  (overflowing)
                       or  edge_frac > max_edge (clipped at the border)

Run on the server:

    python scripts/training/score_framing.py --data-root data/training_dataset \
        --out results/framing

Outputs:
    results/framing/framing.csv        per-view fg_frac / edge_frac / off_ctr / flag
    results/framing/exclude.json       list of "subject/view" flagged BAD (feed to training)
    results/framing/summary.json       % bad, per-subject counts
    results/framing/montage_bad.png    sample of flagged-BAD views  (eyeball to tune thresholds)
    results/framing/montage_ok.png     sample of flagged-OK views
"""
from __future__ import annotations

import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def fg_mask(seg_path: Path, thresh: float):
    img = np.asarray(Image.open(seg_path).convert("RGB"), dtype=np.float32) / 255.0
    return img.max(axis=2) > thresh, img   # organ pixels = brighter than near-black bg


def measure(seg_path: Path, thresh: float):
    mask, _ = fg_mask(seg_path, thresh)
    h, w = mask.shape
    fg_frac = float(mask.mean())
    border = np.zeros_like(mask)
    b = max(1, int(round(0.01 * h)))        # ~1% border band
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    edge_frac = float(mask[border].mean()) if border.any() else 0.0
    if fg_frac > 0:
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean() / h, xs.mean() / w
        off_ctr = float(np.hypot(cy - 0.5, cx - 0.5))
    else:
        off_ctr = 0.71
    return fg_frac, edge_frac, off_ctr


def montage(items, data_root: Path, path: Path, cols=8, thumb=224):
    if not items:
        return
    tiles = []
    for subj, view, tag in items:
        try:
            im = Image.open(data_root / subj / view / "rgb_preview.png").convert("RGB").resize((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (0, 0, 0))
        canvas = Image.new("RGB", (thumb, thumb + 22), (25, 25, 25))
        canvas.paste(im, (0, 22))
        from PIL import ImageDraw
        ImageDraw.Draw(canvas).text((4, 5), f"{subj}/{view[:14]} {tag}", fill=(230, 230, 120))
        tiles.append(np.asarray(canvas))
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    grid = np.full((rows * th, cols * tw, 3), 25, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
    Image.fromarray(grid).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/training_dataset")
    ap.add_argument("--out", default="results/framing")
    ap.add_argument("--thresh", type=float, default=0.06, help="luminance bg/fg cutoff (max-channel)")
    ap.add_argument("--min-fg", type=float, default=0.04, help="below = organ too tiny/off-screen")
    ap.add_argument("--max-fg", type=float, default=0.75, help="above = overflowing frame")
    ap.add_argument("--max-edge", type=float, default=0.06, help="above = clipped at border")
    ap.add_argument("--montage-n", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, bad, per_subj = [], [], {}
    subj_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for si, subj_dir in enumerate(subj_dirs):
        s = subj_dir.name
        for vdir in sorted(subj_dir.glob("v*")):
            seg = vdir / "seg.png"
            if not seg.exists():
                continue
            fg, edge, off = measure(seg, args.thresh)
            is_bad = (fg < args.min_fg) or (fg > args.max_fg) or (edge > args.max_edge)
            reason = ("tiny" if fg < args.min_fg else "overflow" if fg > args.max_fg
                      else "clipped" if edge > args.max_edge else "")
            rows.append({"subject": s, "view": vdir.name, "fg_frac": round(fg, 4),
                         "edge_frac": round(edge, 4), "off_ctr": round(off, 4),
                         "flag": "BAD" if is_bad else "ok", "reason": reason})
            if is_bad:
                bad.append(f"{s}/{vdir.name}")
                per_subj[s] = per_subj.get(s, 0) + 1
        if (si + 1) % 100 == 0:
            print(f"[framing] {si+1}/{len(subj_dirs)} subjects")

    with open(out / "framing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "view", "fg_frac", "edge_frac", "off_ctr", "flag", "reason"])
        w.writeheader(); w.writerows(rows)
    json.dump(sorted(bad), open(out / "exclude.json", "w"), indent=2)

    n = len(rows); nbad = len(bad)
    summary = {
        "n_views": n, "n_bad": nbad, "pct_bad": round(100 * nbad / max(n, 1), 2),
        "thresholds": {"min_fg": args.min_fg, "max_fg": args.max_fg, "max_edge": args.max_edge},
        "subjects_with_bad": len(per_subj),
        "worst_subjects": sorted(per_subj.items(), key=lambda kv: -kv[1])[:20],
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)

    # validation montages: a sample of flagged BAD and flagged OK
    bad_items = [(r["subject"], r["view"], f"{r['reason']} fg{r['fg_frac']:.2f}")
                 for r in rows if r["flag"] == "BAD"][:args.montage_n]
    ok_items = [(r["subject"], r["view"], f"fg{r['fg_frac']:.2f}")
                for r in rows if r["flag"] == "ok"][:args.montage_n]
    montage(bad_items, root, out / "montage_bad.png")
    montage(ok_items, root, out / "montage_ok.png")

    print(f"\n[framing] {n} views | {nbad} BAD ({summary['pct_bad']}%) across {len(per_subj)} subjects")
    print(f"  exclude list → {out/'exclude.json'}")
    print(f"  eyeball      → {out/'montage_bad.png'} (should look badly framed) and montage_ok.png")
    print(f"  feed to training: --exclude-file {out/'exclude.json'}")


if __name__ == "__main__":
    main()
