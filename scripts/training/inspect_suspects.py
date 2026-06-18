"""
inspect_suspects.py — Eyeball low-tissue subjects to confirm valid-limited-FOV vs broken.

Builds a contact sheet of one rgb_preview.png per low-tissue subject (annotated with the
subject id, tissue count, organ list) and, by default, measures the on-screen FOREGROUND
pixel fraction from the seg-ID pass — so a genuinely empty/broken frame (organ off-screen
or extraction failed) is flagged automatically, while a real but sparse scan is not.

Run on the server, then pull the montage:

    venv/bin/python scripts/training/inspect_suspects.py \
        --data-root data/training_dataset --out results/suspects --max-tissues 5

Outputs:
    results/suspects/montage.png    contact sheet (pull this and look)
    results/suspects/suspects.csv   subject, n_tissues, tissues, fg_frac, flag
"""
from __future__ import annotations

import sys
import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr import read_render_exr

THUMB = 256
HEADER = 54
EMPTY_FRAC = 0.01   # below this foreground fraction → flag as likely broken/empty


def tissue_map(subj_dir: Path):
    p = subj_dir / "tissue_ids.json"
    return json.load(open(p)).get("tissues", {}) if p.exists() else {}


def fg_fraction(view_dir: Path) -> float:
    try:
        got = read_render_exr(str(view_dir / "render.exr"), want=("indexob",))
        ids = np.rint(got["indexob"]).astype(np.int64)
        return float((ids > 0).mean())
    except Exception:
        return float("nan")


def thumb_with_header(view_dir: Path, subject: str, tissues, fg: float):
    img = cv2.imread(str(view_dir / "rgb_preview.png"), cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((THUMB, THUMB, 3), np.uint8)
        cv2.putText(img, "NO PREVIEW", (20, THUMB // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    img = cv2.resize(img, (THUMB, THUMB))
    canvas = np.full((THUMB + HEADER, THUMB, 3), 30, np.uint8)
    canvas[HEADER:] = img
    broken = (not np.isnan(fg)) and fg < EMPTY_FRAC
    col = (0, 0, 255) if broken else (255, 255, 255)
    fgtxt = "n/a" if np.isnan(fg) else f"{fg*100:.1f}%"
    cv2.putText(canvas, f"{subject}  ({len(tissues)}t)  fg={fgtxt}{'  BROKEN?' if broken else ''}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    short = ", ".join(t.replace("_", " ")[:12] for t in tissues)[:46]
    cv2.putText(canvas, short, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/training_dataset")
    ap.add_argument("--out", default="results/suspects")
    ap.add_argument("--max-tissues", type=int, default=5, help="show subjects with <= this many tissues")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--no-coverage", action="store_true", help="skip foreground-fraction measurement")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    suspects = []
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tmap = tissue_map(subj_dir)
        if not tmap or len(tmap) > args.max_tissues:
            continue
        views = sorted(v for v in subj_dir.glob("v*") if (v / "rgb_preview.png").exists())
        if not views:
            continue
        suspects.append((subj_dir.name, sorted(tmap.keys()), views[0]))
    suspects.sort(key=lambda x: len(x[1]))
    print(f"[suspects] {len(suspects)} subjects with <= {args.max_tissues} tissues")

    rows, tiles = [], []
    for subj, tissues, vdir in suspects:
        fg = float("nan") if args.no_coverage else fg_fraction(vdir)
        tiles.append(thumb_with_header(vdir, subj, tissues, fg))
        rows.append([subj, len(tissues), "|".join(tissues),
                     "" if np.isnan(fg) else round(fg, 5),
                     "BROKEN?" if (not np.isnan(fg) and fg < EMPTY_FRAC) else "ok"])

    with open(out / "suspects.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "n_tissues", "tissues", "fg_frac", "flag"]); w.writerows(rows)

    if tiles:
        cols = args.cols
        rows_n = (len(tiles) + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        grid = np.full((rows_n * h, cols * w, 3), 30, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = t
        cv2.imwrite(str(out / "montage.png"), grid)
        print(f"[done] montage → {out/'montage.png'}  ({rows_n}x{cols})")

    flagged = [r[0] for r in rows if r[4] == "BROKEN?"]
    print(f"[flags] {len(flagged)} subject(s) below {EMPTY_FRAC*100:.0f}% foreground"
          f"{': ' + ', '.join(flagged) if flagged else ' (none — all render real anatomy)'}")
    print(f"[csv]  {out/'suspects.csv'}")


if __name__ == "__main__":
    main()
