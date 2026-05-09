"""
1.5.3 — Run zero-shot ControlNet on all 5 signal-strength variants (empty prompt).

For each variant image we use it as the ControlNet conditioning input directly
(treating each variant as the "simple image" signal). Empty prompt isolates the
effect of the input signal richness — no text guidance allowed.

LPIPS vs Mitsuba GT is deferred (GT not yet rendered). We record SSIM vs V1 (flat
baseline) as a proxy for how much the richer variants shift the output.

Usage:
    python scripts/run_controlnet_variants.py --volume CT95
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


NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, painting, cartoon"

VARIANT_SLUGS = [
    "V1_flat",
    "V2_shading",
    "V3_ao",
    "V4_shading_ao",
    "V5_clahe",
]


def pick_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(device):
    print("Loading pipeline …")
    dtype = torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(device)


def img_to_pil_rgb(path: Path, size: int = 512) -> Image.Image:
    arr = cv2.imread(str(path))
    arr = cv2.resize(arr, (size, size))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    bg = cv2.resize(bg, (ag.shape[1], ag.shape[0]))
    return float(ssim(ag, bg, data_range=255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--steps",  type=int, default=20)
    parser.add_argument("--size",   type=int, default=512)
    args = parser.parse_args()

    base = Path("data/renders/simple")
    vol_dir = next(base.glob(f"*{args.volume}*"))
    variants_dir = vol_dir / "variants"
    out_dir      = vol_dir / "controlnet_variants"
    out_dir.mkdir(exist_ok=True)

    device = pick_device()
    pipe   = load_pipeline(device)

    results = []
    output_imgs = []

    print(f"\nRunning ControlNet (empty prompt) on 5 variants — {args.volume}\n")

    for slug in VARIANT_SLUGS:
        variant_path = variants_dir / f"{slug}.png"
        if not variant_path.exists():
            print(f"  [SKIP] {slug} — file not found, run render_variants.py first")
            continue

        cond_img = img_to_pil_rgb(variant_path, args.size)
        variant_arr = np.array(cond_img)

        print(f"  [{slug}]")
        t0 = time.perf_counter()
        gen = torch.Generator(device="cpu").manual_seed(42)

        out = pipe(
            prompt="",
            negative_prompt=NEGATIVE_PROMPT,
            image=cond_img,
            num_inference_steps=args.steps,
            generator=gen,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        elapsed = time.perf_counter() - t0
        out_arr = np.array(out)

        # SSIM: output vs its own variant input (geometry fidelity)
        geo_ssim = compute_ssim(variant_arr, out_arr)

        out.save(out_dir / f"{slug}_output.png")
        print(f"    → {elapsed:.1f}s  |  SSIM(output vs variant)={geo_ssim:.3f}")

        results.append({
            "slug":      slug,
            "ssim_vs_variant": round(geo_ssim, 4),
            "time_s":    round(elapsed, 1),
        })
        output_imgs.append((slug, variant_arr, out_arr))

    # ── Save metrics ─────────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "volume":  args.volume,
            "steps":   args.steps,
            "device":  str(device),
            "model":   "SD 1.5 + ControlNet-depth",
            "prompt":  "(empty)",
            "seed":    42,
            "note_lpips": "LPIPS vs GT deferred — Mitsuba GT not yet rendered (Phase 3)",
            "results": results,
        }, f, indent=2)

    # ── Side-by-side grid: variant input | ControlNet output ─────────────────
    n = len(output_imgs)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))
    for col, (slug, var_arr, out_arr) in enumerate(output_imgs):
        axes[0, col].imshow(var_arr); axes[0, col].set_title(slug, fontsize=8); axes[0, col].axis("off")
        axes[1, col].imshow(out_arr); axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Input variant", fontsize=9)
    axes[1, 0].set_ylabel("ControlNet output", fontsize=9)
    plt.suptitle(f"1.5.3 Signal strength — {args.volume} (empty prompt, seed=42)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── SSIM bar chart ────────────────────────────────────────────────────────
    slugs  = [r["slug"] for r in results]
    scores = [r["ssim_vs_variant"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(slugs, scores, color=["#aac4dd", "#5b9abf", "#2d7ab5", "#145a8c", "#0a3558"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("SSIM (ControlNet output vs variant input)")
    ax.set_title(f"1.5.3 Signal strength experiment — {args.volume}\n"
                 "(higher = output closer to conditioning input)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "ssim_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Variant':<20}  {'SSIM vs input':>13}  {'Time':>7}")
    print("=" * 55)
    for r in results:
        print(f"  {r['slug']:<18}  {r['ssim_vs_variant']:>13.3f}  {r['time_s']:>6.1f}s")
    print("=" * 55)
    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
