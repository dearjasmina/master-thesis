"""
Compare three ControlNet conditioning types on the same 6 lighting prompts.

  depth   → control_v11f1p_sd15_depth   ← depth.png
  edges   → control_v11p_sd15_canny     ← edges.png
  normals → control_v11p_sd15_normalbae ← normals.png

Runs all 6 prompts × 3 conditioning types, seed=42.
Saves per-condition grids + a combined comparison grid.

Usage:
    python scripts/run_controlnet_conditioning.py --volume CT95
    python scripts/run_controlnet_conditioning.py --volume CT95 --conditions depth edges normals
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


PROMPTS = [
    ("warm_studio",    "photorealistic CT render, warm studio lighting, bone structure"),
    ("cold_overhead",  "photorealistic medical visualization, cold blue overhead lighting"),
    ("dramatic_side",  "cinematic render of human skull, dramatic side lighting, high contrast"),
    ("soft_diffuse",   "photorealistic bone structure, soft diffuse lighting, clinical setting"),
    ("xray_glow",      "X-ray style medical render, glowing internal structure"),
    ("empty_baseline", ""),
]

NEGATIVE = "blurry, low quality, distorted, deformed, painting, cartoon"

CONDITION_CONFIG = {
    "depth":   ("lllyasviel/control_v11f1p_sd15_depth",   "depth.png"),
    "edges":   ("lllyasviel/control_v11p_sd15_canny",     "edges.png"),
    "normals": ("lllyasviel/control_v11p_sd15_normalbae", "normals.png"),
}


def pick_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(model_id: str, device: torch.device):
    dtype = torch.float32
    controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(device)


def load_cond_image(path: Path, size: int = 512) -> Image.Image:
    arr = cv2.imread(str(path))
    arr = cv2.resize(arr, (size, size))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def compute_ssim(simple_path: Path, out: Image.Image, size: int = 512) -> float:
    sg = cv2.imread(str(simple_path), cv2.IMREAD_GRAYSCALE)
    og = np.array(out.convert("L").resize((size, size)))
    return float(ssim(sg, og, data_range=255))


def run_condition(condition: str, vol_dir: Path, out_root: Path,
                  device: torch.device, steps: int, size: int) -> list[dict]:
    model_id, cond_file = CONDITION_CONFIG[condition]
    cond_path   = vol_dir / cond_file
    simple_path = vol_dir / "color.png"

    if not cond_path.exists():
        print(f"  [SKIP] {condition} — {cond_file} not found")
        return []

    print(f"\n{'='*60}")
    print(f"  Condition: {condition}  |  model: {model_id.split('/')[-1]}")
    print(f"  Input: {cond_file}")
    print(f"{'='*60}")

    pipe     = load_pipeline(model_id, device)
    cond_img = load_cond_image(cond_path, size)
    out_dir  = out_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    results   = []
    grid_imgs = []

    for slug, prompt in PROMPTS:
        gen = torch.Generator(device="cpu").manual_seed(42)
        t0  = time.perf_counter()

        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=cond_img,
            num_inference_steps=steps,
            generator=gen,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        elapsed  = time.perf_counter() - t0
        geo_ssim = compute_ssim(simple_path, out, size)

        out.save(out_dir / f"{slug}.png")
        print(f"  [{slug:<18}]  SSIM={geo_ssim:.3f}  {elapsed:.1f}s")

        results.append({
            "condition": condition,
            "slug":      slug,
            "prompt":    prompt,
            "ssim_vs_color": round(geo_ssim, 4),
            "time_s":    round(elapsed, 1),
        })
        grid_imgs.append((slug, out))

    # Per-condition grid
    cols = 3
    rows = (len(grid_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    axes = np.array(axes).flatten()
    for ax, (label, img) in zip(axes, grid_imgs):
        ax.imshow(img); ax.set_title(label, fontsize=8); ax.axis("off")
    for ax in axes[len(grid_imgs):]: ax.axis("off")
    plt.suptitle(f"{condition} conditioning — 6 prompts", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Free VRAM before next model
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results


def make_comparison_grid(all_results: list[dict], out_path: Path):
    """SSIM bar chart comparing all conditions × prompts."""
    conditions = list({r["condition"] for r in all_results})
    slugs      = [s for s, _ in PROMPTS]
    colors     = {"depth": "#2d7ab5", "edges": "#e07b39", "normals": "#5cb85c"}

    x     = np.arange(len(slugs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, cond in enumerate(sorted(conditions)):
        scores = []
        for slug in slugs:
            match = [r for r in all_results if r["condition"] == cond and r["slug"] == slug]
            scores.append(match[0]["ssim_vs_color"] if match else 0)
        bars = ax.bar(x + i * width, scores, width, label=cond,
                      color=colors.get(cond, "#999"))
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, s + 0.005,
                    f"{s:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(slugs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("SSIM vs color render")
    ax.set_ylim(0, 1)
    ax.set_title("ControlNet conditioning comparison — SSIM geometry check\n"
                 "(higher = output closer to original bone render)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume",     required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=["depth", "edges", "normals"],
                        choices=["depth", "edges", "normals"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size",  type=int, default=512)
    args = parser.parse_args()

    base = Path("data/renders/simple")
    vol_dir = next(base.glob(f"*{args.volume}*"))
    out_root = vol_dir / "controlnet_conditioning"
    out_root.mkdir(exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    all_results = []
    for condition in args.conditions:
        results = run_condition(condition, vol_dir, out_root, device, args.steps, args.size)
        all_results.extend(results)

    # Save combined metrics
    with open(out_root / "metrics.json", "w") as f:
        json.dump({
            "volume": args.volume, "steps": args.steps,
            "seed": 42, "device": str(device),
            "results": all_results,
        }, f, indent=2)

    if len(args.conditions) > 1:
        make_comparison_grid(all_results, out_root / "comparison_ssim.png")

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Condition':<10} {'Prompt':<20} {'SSIM':>6}")
    print("=" * 65)
    for r in all_results:
        print(f"  {r['condition']:<10} {r['slug']:<20} {r['ssim_vs_color']:>6.3f}")
    print("=" * 65)
    print(f"\nAll outputs → {out_root}/")


if __name__ == "__main__":
    main()
