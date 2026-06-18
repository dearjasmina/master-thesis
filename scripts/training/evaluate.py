"""
evaluate.py — Fidelity evaluation against the Cycles GT (3rd.md eval protocol).

    python scripts/training/evaluate.py --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
        --split test --preset full1024

Reports per-image PSNR / SSIM / LPIPS vs GT (primary ranking metric), writes a CSV
and sample grids. The cross-view reprojection consistency gate is stubbed below
(Stage-2 placeholder) since it needs known-pose paired-view batching.

The volume-level split is implemented in splits.py, so --split test / val evaluate
genuinely held-out subjects.
"""
from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import preset, ALL_INPUT_BUFFERS
from dataset import PairedRenderDataset
from networks import define_G
from losses import build_lpips


def to01(t):  # CxHxW [-1,1] → HxWxC [0,1]
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).numpy().transpose(1, 2, 0)


def cross_view_reprojection_gate(*args, **kwargs):
    """
    >>> Stage-2 PLACEHOLDER (3rd.md advisor gate). <<<
    For each test subject, take overlapping view pairs (i, j), warp generated_i into
    frame j via known pose (meta.json: K, cam_to_world) + depth_i, mask occlusions,
    and compute masked photometric + LPIPS error on co-visible pixels — benchmarked
    AGAINST the same metric on the Cycles GT pairs (the GT is the floor, since genuine
    view-dependent specular/Fresnel differs). Pass if generated error ≈ GT error.
    Needs paired-view batching; not wired up yet.
    """
    raise NotImplementedError("cross-view reprojection gate is a Stage-2 placeholder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="full1024", choices=["proto512", "full1024", "rgb_only"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--target", default=None, choices=["preview_png", "exr_agx", "exr_linear"])
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = preset(args.preset)
    if args.data_root: cfg.data.root = args.data_root
    if args.target: cfg.data.target = args.target
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = define_G(cfg).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ck["G"])
    print(f"[eval] loaded {args.checkpoint} (epoch {ck.get('epoch','?')})")

    ds = PairedRenderDataset(cfg, split=args.split)
    print(f"[eval] split={args.split}: {len(ds)} views")

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    lpips_net = build_lpips(device)

    out_dir = Path(args.out) if args.out else Path(cfg.train.output_dir) / "eval"
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(ds) if args.max_samples == 0 else min(args.max_samples, len(ds))

    import cv2
    with torch.no_grad():
        for i in range(n):
            b = ds[i]
            inp = b["input"].unsqueeze(0).to(device)
            real = b["target"].unsqueeze(0).to(device)
            fake = G(inp)

            f01 = np.clip(to01(fake[0]), 0, 1)
            r01 = np.clip(to01(real[0]), 0, 1)
            psnr = float(psnr_fn(r01, f01, data_range=1.0))
            ssim = float(ssim_fn(r01, f01, data_range=1.0, channel_axis=2))
            lp = float(lpips_net(fake, real).mean()) if lpips_net is not None else float("nan")
            rows.append({"subject": b["subject"], "view": b["view"],
                         "psnr": psnr, "ssim": ssim, "lpips": lp})

            if i < 16:
                grid = np.concatenate([r01, f01], axis=1)
                cv2.imwrite(str(out_dir / "grids" / f"{b['subject']}_{b['view']}.png"),
                            cv2.cvtColor((grid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    csv_path = out_dir / f"metrics_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "view", "psnr", "ssim", "lpips"])
        w.writeheader(); w.writerows(rows)

    arr = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    print(f"\n[eval] N={len(rows)}")
    print(f"  PSNR  {arr('psnr').mean():.3f} ± {arr('psnr').std():.3f}")
    print(f"  SSIM  {arr('ssim').mean():.4f} ± {arr('ssim').std():.4f}")
    if lpips_net is not None:
        print(f"  LPIPS {arr('lpips').mean():.4f} ± {arr('lpips').std():.4f}")
    print(f"  → {csv_path}")


if __name__ == "__main__":
    main()
