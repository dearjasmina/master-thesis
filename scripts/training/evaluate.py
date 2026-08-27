"""
evaluate.py — Fidelity evaluation against the Cycles GT (3rd.md eval protocol).

Single checkpoint, detailed (per-view CSV + sample grids + worst-K diagnosis):
    python scripts/training/evaluate.py --preset full1024 \
        --checkpoint /mnt/data/$USER/runs/full1024/checkpoints/latest.pt \
        --split test --worst 12

Track ALL checkpoints — train vs val curves for overfitting analysis (run this every
few epochs; it caches and only evaluates NEW checkpoints, then prints the full table):
    python scripts/training/evaluate.py --preset full1024 \
        --run-dir /mnt/data/$USER/runs/full1024 --track

Reports PSNR / SSIM / LPIPS vs GT. The volume-level split (splits.py) means --split
val/test are genuinely held-out subjects, so the train-vs-val gap = the overfitting signal.
"""
from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import preset, PRESET_NAMES
from dataset import PairedRenderDataset
from networks import define_G
from losses import build_lpips


def to01(t):  # CxHxW [-1,1] → HxWxC [0,1]
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).numpy().transpose(1, 2, 0)


def cross_view_reprojection_gate(*args, **kwargs):
    """
    >>> Stage-2 PLACEHOLDER (3rd.md advisor gate). <<<
    Warp generated_i into frame j via known pose (meta.json K, cam_to_world) + depth_i,
    mask occlusions, compare masked photometric/LPIPS on co-visible pixels against the
    same metric on the Cycles GT pairs. Needs paired-view batching; not wired up yet.
    """
    raise NotImplementedError("cross-view reprojection gate is a Stage-2 placeholder")


def eval_metrics(G, ds, device, lpips_net, max_samples=0):
    """Mean PSNR/SSIM/LPIPS over (a subset of) a dataset. Returns dict + per-view rows."""
    n = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
    rows = []
    with torch.no_grad():
        for i in range(n):
            b = ds[i]
            real = b["target"].unsqueeze(0).to(device)
            fake = G(b["input"].unsqueeze(0).to(device))
            f01 = np.clip(to01(fake[0]), 0, 1)
            r01 = np.clip(to01(real[0]), 0, 1)
            lp = float(lpips_net(fake, real).mean()) if lpips_net is not None else float("nan")
            rows.append({"idx": i, "subject": b["subject"], "view": b["view"],
                         "psnr": float(psnr_fn(r01, f01, data_range=1.0)),
                         "ssim": float(ssim_fn(r01, f01, data_range=1.0, channel_axis=2)),
                         "lpips": lp})
    a = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    summary = {"n": len(rows),
               "psnr": float(a("psnr").mean()), "ssim": float(a("ssim").mean()),
               "lpips": float(a("lpips").mean())}
    return summary, rows


# ── Track mode: train-vs-val across all checkpoints ───────────────────────────
def track(args, cfg, device):
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg.train.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print(f"[track] no epoch_*.pt under {ckpt_dir}"); return
    out_dir = Path(args.out) if args.out else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "metrics_history.csv"

    # Resume: load what we've already computed so we only eval NEW checkpoints.
    done = {}
    if hist_path.exists():
        for r in csv.DictReader(open(hist_path)):
            done[(int(r["epoch"]), r["split"])] = {k: r[k] for k in r}

    splits = ["train", "val"]
    datasets = {}
    for s in splits:
        try:
            datasets[s] = PairedRenderDataset(cfg, split=s)
        except Exception as e:
            print(f"[track] split '{s}' unavailable ({e})")
    lpips_net = build_lpips(device)
    G = define_G(cfg).to(device).eval()

    for ck in ckpts:
        epoch = int(ck.stem.split("_")[1])
        todo = [s for s in datasets if (epoch, s) not in done]
        if not todo:
            continue
        G.load_state_dict(torch.load(ck, map_location=device)["G"])
        for s in todo:
            m, _ = eval_metrics(G, datasets[s], device, lpips_net, args.track_samples)
            done[(epoch, s)] = {"epoch": epoch, "split": s, **m}
            print(f"[track] epoch {epoch:>3} {s:<5}  PSNR {m['psnr']:.2f}  "
                  f"SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}  (n={m['n']})")

    # Persist full history.
    rows = [done[k] for k in sorted(done, key=lambda x: (x[0], x[1]))]
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "split", "n", "psnr", "ssim", "lpips"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # Print the train-vs-val table.
    def g(ep, s, k):
        v = done.get((ep, s))
        return float(v[k]) if v else None
    def fmt(x, d=2):
        return f"{x:.{d}f}" if x is not None else "  -  "
    print(f"\n  epoch | train PSNR  SSIM   LPIPS | val   PSNR  SSIM   LPIPS | ΔPSNR(tr−val)")
    print(f"  ------+-------------------------+-------------------------+--------------")
    for ep in sorted({k[0] for k in done}):
        tp, vp = g(ep, "train", "psnr"), g(ep, "val", "psnr")
        gap = f"{tp - vp:+.2f}" if (tp is not None and vp is not None) else "  -"
        print(f"  {ep:5d} | {fmt(tp)}   {fmt(g(ep,'train','ssim'),3)}  {fmt(g(ep,'train','lpips'),3)} "
              f"| {fmt(vp)}   {fmt(g(ep,'val','ssim'),3)}  {fmt(g(ep,'val','lpips'),3)} | {gap}")
    print(f"\n  → {hist_path}")
    print("  Read: val improving = keep going. val plateaus while ΔPSNR grows = overfitting → "
          "use the best-val checkpoint.")


# ── Single-checkpoint detailed eval ───────────────────────────────────────────
def eval_single(args, cfg, device):
    import cv2
    G = define_G(cfg).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ck["G"])
    print(f"[eval] loaded {args.checkpoint} (epoch {ck.get('epoch', '?')})")

    ds = PairedRenderDataset(cfg, split=args.split)
    print(f"[eval] split={args.split}: {len(ds)} views")
    lpips_net = build_lpips(device)
    _, rows = eval_metrics(G, ds, device, lpips_net, args.max_samples)

    out_dir = Path(args.out) if args.out else Path(cfg.train.output_dir) / "eval"
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)

    # Choose which views to save as seg|fake|GT grids:
    #   --grid-subjects "s0809,s0445" → ALL views of those subjects (multi-view showcase)
    #   otherwise                     → --grids N views spread EVENLY across the split
    #                                   (variety of subjects, not just the first one)
    if args.grid_subjects:
        keep = {s.strip() for s in args.grid_subjects.split(",") if s.strip()}
        grid_rows = [r for r in rows if r["subject"] in keep]
    else:
        idxs = np.unique(np.linspace(0, len(rows) - 1, min(args.grids, len(rows))).astype(int))
        grid_rows = [rows[i] for i in idxs]

    tiles = []
    with torch.no_grad():
        for r in grid_rows:
            b = ds[r["idx"]]
            fake = G(b["input"].unsqueeze(0).to(device))
            grid = np.concatenate([np.clip(to01(b["input"][:3]), 0, 1),
                                   np.clip(to01(fake[0]), 0, 1),
                                   np.clip(to01(b["target"]), 0, 1)], axis=1)  # seg|fake|GT
            g8 = (grid * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "grids" / f"{b['subject']}_{b['view']}.png"),
                        cv2.cvtColor(g8, cv2.COLOR_RGB2BGR))
            th = 240  # downscale for the montage
            tiles.append(cv2.resize(g8, (int(g8.shape[1] * th / g8.shape[0]), th)))

    # One montage image (seg|fake|GT stacked) — the "send to prof" artifact.
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=20) for t in tiles]
        montage = np.concatenate(tiles, axis=0)
        cv2.imwrite(str(out_dir / "montage.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"[eval] {len(tiles)} grids + montage → {out_dir/'montage.png'} "
              f"(left=seg input · middle=neural render · right=Cycles GT)")

    if args.worst > 0:
        worst = sorted(rows, key=lambda r: r["lpips"], reverse=True)[:args.worst]
        (out_dir / "worst").mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for r in worst:
                b = ds[r["idx"]]
                fake = G(b["input"].unsqueeze(0).to(device))
                grid = np.concatenate([np.clip(to01(b["input"][:3]), 0, 1),
                                       np.clip(to01(fake[0]), 0, 1),
                                       np.clip(to01(b["target"]), 0, 1)], axis=1)
                name = f"lpips{r['lpips']:.3f}_psnr{r['psnr']:.1f}_{r['subject']}_{r['view']}.png"
                cv2.imwrite(str(out_dir / "worst" / name),
                            cv2.cvtColor((grid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"[eval] dumped {len(worst)} worst-LPIPS grids → {out_dir/'worst'}")

    csv_path = out_dir / f"metrics_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "view", "psnr", "ssim", "lpips"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    a = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    print(f"\n[eval] N={len(rows)}")
    print(f"  PSNR  {a('psnr').mean():.3f} ± {a('psnr').std():.3f}")
    print(f"  SSIM  {a('ssim').mean():.4f} ± {a('ssim').std():.4f}")
    print(f"  LPIPS {a('lpips').mean():.4f} ± {a('lpips').std():.4f}")
    print(f"  → {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full1024", choices=PRESET_NAMES)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--target", default=None, choices=["preview_png", "exr_agx", "exr_linear"])
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--exclude-file", default=None)
    ap.add_argument("--out", default=None)
    # single-checkpoint mode
    ap.add_argument("--checkpoint", default=None, help="single-checkpoint detailed eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--worst", type=int, default=0, help="dump K worst-LPIPS seg|fake|GT grids")
    ap.add_argument("--grids", type=int, default=16,
                    help="how many seg|fake|GT grids to save, spread evenly across subjects")
    ap.add_argument("--grid-subjects", default=None,
                    help="comma list of subjects → save ALL their views as grids (multi-view showcase)")
    # track mode
    ap.add_argument("--track", action="store_true",
                    help="evaluate ALL checkpoints in --run-dir on train+val and print the curve")
    ap.add_argument("--run-dir", default=None, help="training output dir (contains checkpoints/)")
    ap.add_argument("--track-samples", type=int, default=300,
                    help="views/split to sample per checkpoint in --track (speed)")
    args = ap.parse_args()

    cfg = preset(args.preset)
    if args.data_root: cfg.data.root = args.data_root
    if args.target: cfg.data.target = args.target
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]
    if args.exclude_file: cfg.data.exclude_file = args.exclude_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.track:
        track(args, cfg, device)
    else:
        if not args.checkpoint:
            ap.error("single-checkpoint mode needs --checkpoint (or use --track --run-dir ...)")
        eval_single(args, cfg, device)


if __name__ == "__main__":
    main()
