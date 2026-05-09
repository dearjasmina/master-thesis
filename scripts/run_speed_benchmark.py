"""
1.5.5 — Speed benchmark: confirms networks are too slow for interactive use.

Tests three configurations (10 timed runs each where practical):
  A. SD 1.5 + ControlNet @ 20 steps
  B. SD 1.5 + ControlNet @ 50 steps
  C. IP-Adapter + ControlNet @ 20 steps (proxy for 30-step cluster target)

Interactive threshold: < 33ms per frame = 30 FPS.
3DGS target:          ≥ 100 FPS = < 10ms per frame.

Usage:
    python scripts/run_speed_benchmark.py --volume CT95
    python scripts/run_speed_benchmark.py --volume CT95 --runs 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


INTERACTIVE_MS = 33.0    # 30 FPS threshold
REALTIME_MS    = 10.0    # 100 FPS (3DGS target)
LUMA_LATENCY_S = 30.0    # observed cloud API latency (manual timing, rough)


def pick_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_controlnet_pipe(device, dtype=torch.float32):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(device)


def load_cond_image(path: Path, size=512) -> Image.Image:
    arr = cv2.imread(str(path))
    arr = cv2.resize(arr, (size, size))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def benchmark(pipe, cond_img, steps: int, n_runs: int, device) -> dict:
    """Time n_runs inference calls, return stats in seconds."""
    times = []
    gen = torch.Generator(device="cpu").manual_seed(42)
    for i in range(n_runs):
        # Warm-up on first run (don't count it)
        t0 = time.perf_counter()
        pipe(prompt="", image=cond_img,
             num_inference_steps=steps, generator=gen,
             controlnet_conditioning_scale=1.0).images[0]
        elapsed = time.perf_counter() - t0
        if i > 0:   # skip warm-up
            times.append(elapsed)
        gen = torch.Generator(device="cpu").manual_seed(42 + i)
        print(f"    run {i+1}/{n_runs}: {elapsed:.1f}s", flush=True)

    times = np.array(times)
    return {
        "mean_s":  float(times.mean()),
        "std_s":   float(times.std()),
        "min_s":   float(times.min()),
        "max_s":   float(times.max()),
        "n_runs":  len(times),
        "steps":   steps,
    }


def fps(mean_s: float) -> float:
    return 1.0 / mean_s if mean_s > 0 else 0.0


def slowdown(mean_s: float, threshold_ms: float) -> float:
    return (mean_s * 1000) / threshold_ms


def make_table(results: list[dict]) -> str:
    header = f"{'Method':<35} {'Steps':>5} {'Mean':>8} {'Std':>6} {'FPS':>6} {'vs 30FPS':>10}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for r in results:
        rows.append(
            f"  {r['label']:<33} {r['steps']:>5} "
            f"{r['mean_s']:>7.2f}s {r['std_s']:>5.2f}s "
            f"{fps(r['mean_s']):>5.2f} "
            f"{slowdown(r['mean_s'], INTERACTIVE_MS):>8.0f}×"
        )
    return "\n".join(rows)


def make_chart(results: list[dict], out_path: Path):
    labels  = [r["label"] for r in results]
    means   = [r["mean_s"] for r in results]
    stds    = [r.get("std_s", 0) for r in results]
    colors  = ["#2d7ab5", "#1a5276", "#e07b39", "#c0392b", "#27ae60"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time in seconds
    bars = axes[0].bar(labels, means, yerr=stds, capsize=5,
                       color=colors[:len(labels)], alpha=0.85)
    axes[0].axhline(INTERACTIVE_MS / 1000, color="red",
                    linestyle="--", label=f"30 FPS ({INTERACTIVE_MS}ms)")
    axes[0].set_ylabel("Inference time (s)")
    axes[0].set_title("Inference time per image")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, m + 0.5,
                     f"{m:.1f}s", ha="center", va="bottom", fontsize=8)

    # Right: slowdown vs interactive threshold (log scale)
    slowdowns = [slowdown(m, INTERACTIVE_MS) for m in means]
    axes[1].bar(labels, slowdowns, color=colors[:len(labels)], alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].axhline(1, color="red", linestyle="--", label="Interactive threshold (1×)")
    axes[1].set_ylabel("Slowdown vs 30 FPS (log scale)")
    axes[1].set_title("How far from real-time?")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)
    for i, (bar, s) in enumerate(zip(axes[1].patches, slowdowns)):
        axes[1].text(bar.get_x() + bar.get_width()/2, s * 1.1,
                     f"{s:.0f}×", ha="center", va="bottom", fontsize=8)

    plt.suptitle("1.5.5 Speed benchmark — networks vs interactive rate\n"
                 f"Device: {pick_device()}  |  Interactive threshold: <{INTERACTIVE_MS}ms (30 FPS)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--runs",   type=int, default=5,
                        help="Timed runs per config (first is warm-up, not counted)")
    args = parser.parse_args()

    base = Path("data/renders/simple")
    vol_dir  = next(base.glob(f"*{args.volume}*"))
    out_dir  = Path("results/speed_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_path = vol_dir / "depth.png"
    device     = pick_device()
    cond_img   = load_cond_image(depth_path)

    print(f"Device : {device}")
    print(f"Volume : {vol_dir.name}")
    print(f"Runs   : {args.runs - 1} timed (+ 1 warm-up)\n")

    all_results = []

    # ── A: ControlNet @ 20 steps ─────────────────────────────────────────────
    print("=" * 55)
    print("A. SD 1.5 + ControlNet-depth @ 20 steps")
    pipe = load_controlnet_pipe(device)
    r_a  = benchmark(pipe, cond_img, steps=20, n_runs=args.runs, device=device)
    r_a["label"] = "SD1.5 + ControlNet"
    all_results.append(r_a)
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── B: ControlNet @ 50 steps ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("B. SD 1.5 + ControlNet-depth @ 50 steps")
    pipe = load_controlnet_pipe(device)
    r_b  = benchmark(pipe, cond_img, steps=50, n_runs=args.runs, device=device)
    r_b["label"] = "SD1.5 + ControlNet"
    all_results.append(r_b)
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── C: IP-Adapter — use timings from 1.5.4 (already ran, very slow on MPS)
    print("\n" + "=" * 55)
    print("C. IP-Adapter + ControlNet (from 1.5.4 timings, scale=0.3, 20 steps)")
    ip_metrics_path = vol_dir / "ip_adapter" / "metrics.json"
    if ip_metrics_path.exists():
        with open(ip_metrics_path) as f:
            ip_data = json.load(f)
        ip_times = [r["time_s"] for r in ip_data["results"]]
        r_c = {
            "label":  "IP-Adapter + ControlNet",
            "steps":  20,
            "mean_s": float(np.mean(ip_times)),
            "std_s":  float(np.std(ip_times)),
            "min_s":  float(np.min(ip_times)),
            "max_s":  float(np.max(ip_times)),
            "n_runs": len(ip_times),
            "note":   "Reused from 1.5.4 (5 scale runs @ 20 steps)",
        }
        print(f"  Loaded {len(ip_times)} runs from 1.5.4: mean={r_c['mean_s']:.1f}s ± {r_c['std_s']:.1f}s")
    else:
        print("  No 1.5.4 metrics found — skipping IP-Adapter row")
        r_c = None
    if r_c:
        all_results.append(r_c)

    # ── D: Luma Ray (manual observation) ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("D. Luma Ray (cloud API — manual timing observation)")
    r_d = {
        "label":  "Luma Ray 2 (cloud)",
        "steps":  0,
        "mean_s": LUMA_LATENCY_S,
        "std_s":  0.0,
        "min_s":  LUMA_LATENCY_S,
        "max_s":  LUMA_LATENCY_S,
        "n_runs": 1,
        "note":   "Cloud API; latency includes network + generation; manual estimate ~30s",
    }
    all_results.append(r_d)

    # ── 3DGS reference row (literature value) ────────────────────────────────
    r_3dgs = {
        "label":  "3DGS (real-time target)",
        "steps":  0,
        "mean_s": 0.010,   # 100 FPS = 10ms
        "std_s":  0.0,
        "min_s":  0.010,
        "max_s":  0.010,
        "n_runs": 0,
        "note":   "Literature value: ≥100 FPS @ 1080p on RTX 3090 (Kerbl et al., SIGGRAPH 2023)",
    }
    all_results.append(r_3dgs)

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "device":             str(device),
            "interactive_fps":    30,
            "interactive_ms":     INTERACTIVE_MS,
            "realtime_fps":       100,
            "realtime_ms":        REALTIME_MS,
            "results":            all_results,
        }, f, indent=2)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SPEED BENCHMARK RESULTS")
    print("=" * 70)
    print(make_table(all_results))
    print("=" * 70)
    print(f"\nInteractive threshold : <{INTERACTIVE_MS}ms = 30 FPS")
    print(f"3DGS real-time target : <{REALTIME_MS}ms = 100 FPS")
    print(f"\nNote: benchmarked on {device}. Cluster GPU (24GB CUDA) expected")
    print("  ~10–20× faster for ControlNet, ~5–10× faster for IP-Adapter.")

    # ── Chart ─────────────────────────────────────────────────────────────────
    chart_results = [r for r in all_results if r["label"] != "3DGS (real-time target)"]
    make_chart(chart_results, out_dir / "speed_chart.png")

    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()
