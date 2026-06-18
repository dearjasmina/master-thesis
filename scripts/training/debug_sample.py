"""
debug_sample.py — Sanity-check the data path on REAL data before training.

Loads a few samples through the actual PairedRenderDataset, prints per-channel tensor
stats, and dumps each input buffer + the target as labelled PNGs. This is the cheapest
way to catch the failures that otherwise waste GPU-days:
  - EXR reader not finding Depth/Normal/IndexOB on the real files (channels mis-named)
  - depth all -1 / all 1 (bad normalisation, wrong depth_metric_ref)
  - normals outside [-1,1] or all zero
  - seg-ID not scaled sensibly
  - target wrong (preview vs exr_agx vs linear)

Run on the server:

    venv/bin/python scripts/training/debug_sample.py --preset full1024 \
        --data-root data/training_dataset --n 3 --out results/debug_sample

Then pull results/debug_sample/*.png and eyeball; the console stats already tell you
if a channel is degenerate.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import preset, INPUT_CHANNELS
from dataset import PairedRenderDataset


def chan_to_png(arr: np.ndarray) -> Image.Image:
    """arr: HxW or HxWxC in [-1,1] → uint8 PNG (grayscale for 1ch)."""
    a = (np.clip(arr, -1, 1) + 1) * 127.5
    a = a.astype(np.uint8)
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full1024")
    ap.add_argument("--data-root", default="data/training_dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target", default=None)
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results/debug_sample")
    args = ap.parse_args()

    cfg = preset(args.preset)
    cfg.data.root = args.data_root
    if args.target: cfg.data.target = args.target
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]
    print(f"[cfg] {cfg.summary()}")

    ds = PairedRenderDataset(cfg, split=args.split)
    print(f"[data] split={args.split}: {len(ds)} views, num_classes={ds.num_classes}, input_nc={cfg.input_nc()}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # channel offsets per buffer (to slice the stacked input tensor)
    offsets, o = {}, 0
    for b in cfg.data.input_buffers:
        offsets[b] = (o, o + INPUT_CHANNELS[b]); o += INPUT_CHANNELS[b]

    for i in range(min(args.n, len(ds))):
        s = ds[i]
        x = s["input"].numpy()      # CxHxW
        y = s["target"].numpy()
        tag = f"{s['subject']}_{s['view']}"
        print(f"\n=== sample {i}: {tag} ===  input {x.shape}  target {y.shape}")
        for b, (a, bb) in offsets.items():
            sub = x[a:bb]
            print(f"  {b:<8} ch[{a}:{bb}]  min={sub.min():+.3f} max={sub.max():+.3f} "
                  f"mean={sub.mean():+.3f}  {'DEGENERATE?' if sub.max()-sub.min() < 1e-3 else ''}")
            img = sub[0] if sub.shape[0] == 1 else sub.transpose(1, 2, 0)
            chan_to_png(img).save(out / f"{i}_{tag}_{b}.png")
        print(f"  target   min={y.min():+.3f} max={y.max():+.3f} mean={y.mean():+.3f}")
        chan_to_png(y.transpose(1, 2, 0)).save(out / f"{i}_{tag}_TARGET.png")

    print(f"\n[done] dumped to {out.resolve()} — pull and eyeball; watch for DEGENERATE? flags.")


if __name__ == "__main__":
    main()
