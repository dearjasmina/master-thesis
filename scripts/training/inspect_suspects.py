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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr import read_render_exr

THUMB = 256
HEADER = 54
EMPTY_FRAC = 0.01   # below this foreground fraction → flag as likely broken/empty

try:
    _font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    _font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    _font_lg = ImageFont.load_default()
    _font_sm = ImageFont.load_default()


def tissue_map(subj_dir: Path):
    p = subj_dir / "tissue_ids.json"
    if not p.exists():
        return {}
    try:
        return json.load(open(p)).get("tissues", {})
    except Exception:
        print(f"[warn] {p}: empty or invalid JSON — skipped")
        return {}


def fg_fraction(view_dir: Path) -> float:
    try:
        got = read_render_exr(str(view_dir / "render.exr"), want=("indexob",))
        ids = np.rint(got["indexob"]).astype(np.int64)
        return float((ids > 0).mean())
    except Exception:
        return float("nan")


def thumb_with_header(view_dir: Path, subject: str, tissues, fg: float):
    try:
        pil_img = Image.open(view_dir / "rgb_preview.png").convert("RGB")
        pil_img = pil_img.resize((THUMB, THUMB), Image.LANCZOS)
    except Exception:
        pil_img = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
        ImageDraw.Draw(pil_img).text((20, THUMB // 2), "NO PREVIEW", fill=(255, 0, 0), font=_font_lg)
    canvas = Image.new("RGB", (THUMB, THUMB + HEADER), (30, 30, 30))
    canvas.paste(pil_img, (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    broken = (not np.isnan(fg)) and fg < EMPTY_FRAC
    col = (255, 0, 0) if broken else (255, 255, 255)
    fgtxt = "n/a" if np.isnan(fg) else f"{fg*100:.1f}%"
    draw.text((6, 4), f"{subject}  ({len(tissues)}t)  fg={fgtxt}{'  BROKEN?' if broken else ''}",
              fill=col, font=_font_lg)
    short = ", ".join(t.replace("_", " ")[:12] for t in tissues)[:46]
    draw.text((6, 22), short, fill=(180, 220, 180), font=_font_sm)
    return np.array(canvas)


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
        Image.fromarray(grid).save(str(out / "montage.png"))
        print(f"[done] montage → {out/'montage.png'}  ({rows_n}x{cols})")

    flagged = [r[0] for r in rows if r[4] == "BROKEN?"]
    print(f"[flags] {len(flagged)} subject(s) below {EMPTY_FRAC*100:.0f}% foreground"
          f"{': ' + ', '.join(flagged) if flagged else ' (none — all render real anatomy)'}")
    print(f"[csv]  {out/'suspects.csv'}")


if __name__ == "__main__":
    main()
