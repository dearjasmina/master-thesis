"""
eval_speed.py — the headline measurement: learned renderer vs path tracing

This is the claim the thesis actually makes. PSNR/SSIM/LPIPS say the network reproduces
the ground truth; only this says it is worth doing. Everything else in the suite is
supporting evidence for a speed/quality trade-off that has to be quantified here.

Measures:
  · network latency per frame (ms), batched and unbatched, with correct CUDA
    synchronisation and warm-up — without both, a naive timing is wrong by an order
    of magnitude because kernel launches are asynchronous
  · throughput (frames/s), and peak GPU memory
  · Cycles reference cost per frame, taken from the dataset's own metadata (spp,
    resolution) plus a measured or supplied seconds-per-frame
  · the resulting speed-up, and the break-even dataset size at which training cost is
    amortised

Cycles timing is NOT measured here — it needs Blender and the meshes. Supply it with
--cycles-sec-per-frame from your own generation logs (logs/gpu*.log records per-subject
wall time), or pass --cycles-log to have it parsed. Guessing it would make the headline
number fiction.

Usage
-----
    python scripts/evaluation/eval_speed.py \
        --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
        --cycles-sec-per-frame 21.5 \
        --train-gpu-hours 96
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from eval_common import (add_common_args, build_cfg, pick_device, load_generator,
                         open_dataset, ensure_out)


def time_generator(G, cfg, device, batch_sizes, iters, warmup):
    """Latency/throughput per batch size, with proper synchronisation."""
    c_in = cfg.input_nc()
    res = getattr(cfg.data, "resolution", None) or getattr(cfg, "resolution", 1024)
    results = []
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, c_in, res, res, device=device)
        except RuntimeError as e:
            print(f"  batch {bs}: skipped ({e.__class__.__name__}: out of memory?)")
            continue
        # Warm-up matters: the first calls include cuDNN autotuning and lazy kernel
        # compilation, which can be 10-100x the steady-state cost.
        with torch.no_grad():
            for _ in range(warmup):
                G(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        with torch.no_grad():
            for _ in range(iters):
                t0 = time.perf_counter()
                G(x)
                # CUDA kernels are asynchronous; without this the timer measures how
                # long it took to QUEUE the work, not to do it.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        per_batch = statistics.median(times)
        peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else float("nan")
        results.append({
            "batch": bs,
            "resolution": res,
            "ms_per_batch": round(per_batch * 1000, 3),
            "ms_per_frame": round(per_batch * 1000 / bs, 3),
            "frames_per_sec": round(bs / per_batch, 2),
            "peak_mem_MiB": round(peak, 1),
        })
        print(f"  batch {bs:>3}: {per_batch*1000:8.2f} ms/batch  "
              f"{per_batch*1000/bs:7.2f} ms/frame  {bs/per_batch:7.1f} fps  "
              f"peak {peak:.0f} MiB")
    return results


def parse_cycles_log(path):
    """Extract per-subject wall time from run_dataset_generation.sh output.

    That script prints 'Started: HH:MM:SS' / 'Finished: HH:MM:SS' around each subject.
    Returns seconds per SUBJECT; divide by views/subject for per-frame.
    """
    txt = Path(path).read_text(errors="ignore")
    starts = re.findall(r"Started:\s*(\d\d):(\d\d):(\d\d)", txt)
    ends = re.findall(r"Finished:\s*(\d\d):(\d\d):(\d\d)", txt)
    secs = []
    for s, e in zip(starts, ends):
        a = int(s[0]) * 3600 + int(s[1]) * 60 + int(s[2])
        b = int(e[0]) * 3600 + int(e[1]) * 60 + int(e[2])
        if b < a:                      # crossed midnight
            b += 24 * 3600
        secs.append(b - a)
    return secs


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--cycles-sec-per-frame", type=float, default=None,
                    help="measured Cycles cost per rendered view. Required for the "
                         "speed-up figure — not guessed.")
    ap.add_argument("--cycles-log", default=None,
                    help="a logs/gpu*.log from dataset generation, parsed for timing")
    ap.add_argument("--views-per-subject", type=int, default=20)
    ap.add_argument("--train-gpu-hours", type=float, default=None,
                    help="total GPU-hours spent training, for the break-even estimate")
    args = ap.parse_args()

    out = ensure_out(args.out)
    device = pick_device(args.device)
    cfg = build_cfg(args)
    print(f"eval_speed — device={device}")
    G = load_generator(cfg, args.checkpoint, device)

    bss = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    print("\nnetwork inference:")
    net = time_generator(G, cfg, device, bss, args.iters, args.warmup)
    if not net:
        raise SystemExit("no batch size fitted in memory")
    best = min(r["ms_per_frame"] for r in net)

    # ── Cycles reference ──
    cyc = args.cycles_sec_per_frame
    cyc_source = "supplied"
    if cyc is None and args.cycles_log:
        secs = parse_cycles_log(args.cycles_log)
        if secs:
            per_subject = statistics.median(secs)
            cyc = per_subject / max(args.views_per_subject, 1)
            cyc_source = (f"parsed from {args.cycles_log}: median {per_subject:.0f} s/subject "
                          f"over {len(secs)} subjects / {args.views_per_subject} views")
            print(f"\ncycles: {cyc_source}")

    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "network": net,
        "best_ms_per_frame": best,
    }

    if cyc:
        speedup = (cyc * 1000.0) / best
        report["cycles_sec_per_frame"] = cyc
        report["cycles_source"] = cyc_source
        report["speedup_x"] = round(speedup, 1)
        print(f"\n  Cycles      : {cyc*1000:10.1f} ms/frame")
        print(f"  network     : {best:10.1f} ms/frame")
        print(f"  SPEED-UP    : {speedup:10.1f}x")

        if args.train_gpu_hours:
            # Amortisation: training is a one-off cost paid in GPU-hours; each rendered
            # frame thereafter is cheaper. Break-even is where the saved render time
            # equals the training time. Worth stating explicitly — a speed-up that
            # never amortises is not an engineering win.
            saved_per_frame_s = cyc - best / 1000.0
            if saved_per_frame_s > 0:
                be = args.train_gpu_hours * 3600.0 / saved_per_frame_s
                report["train_gpu_hours"] = args.train_gpu_hours
                report["break_even_frames"] = int(be)
                print(f"  break-even  : {be:,.0f} frames "
                      f"({be/args.views_per_subject:,.0f} subjects) to amortise "
                      f"{args.train_gpu_hours} GPU-h of training")
    else:
        print("\n  [!] no Cycles reference given — speed-up NOT computed.")
        print("      Pass --cycles-sec-per-frame or --cycles-log. A guessed baseline")
        print("      would make the headline number of this thesis fiction.")

    f = out / "speed.json"
    f.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
