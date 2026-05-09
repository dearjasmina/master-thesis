"""
1.5.2 — Zero-shot ControlNet: prompt-based lighting control experiment.

Runs 6 lighting prompts through SD 1.5 + ControlNet-depth on the same depth map,
seed fixed at 42, to test whether text alone can steer illumination.

Usage:
    python scripts/run_controlnet_prompts.py --volume CT95
    python scripts/run_controlnet_prompts.py --volume CT106
    python scripts/run_controlnet_prompts.py --volume CT95 --steps 30
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


PROMPTS = [
    ("warm_studio",    "photorealistic CT render, warm studio lighting, bone structure"),
    ("cold_overhead",  "photorealistic medical visualization, cold blue overhead lighting"),
    ("dramatic_side",  "cinematic render of human skull, dramatic side lighting, high contrast"),
    ("soft_diffuse",   "photorealistic bone structure, soft diffuse lighting, clinical setting"),
    ("xray_glow",      "X-ray style medical render, glowing internal structure"),
    ("empty_baseline", ""),
]

NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, extra limbs, painting, cartoon"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(device: torch.device) -> StableDiffusionControlNetPipeline:
    print("Loading ControlNet depth model …")
    dtype = torch.float32  # float16 unstable on MPS; fine on CUDA

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()  # lower peak VRAM — safe on all devices
    pipe = pipe.to(device)
    print(f"  Pipeline on {device}")
    return pipe


def prepare_depth(depth_path: Path, size: int = 512) -> Image.Image:
    """Load depth map, resize to model input size, convert to RGB PIL (ControlNet expects 3-ch)."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.resize(depth, (size, size))
    depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)


def compute_ssim(simple_path: Path, output_img: Image.Image) -> float:
    simple = cv2.imread(str(simple_path), cv2.IMREAD_GRAYSCALE)
    out    = np.array(output_img.convert("L").resize((simple.shape[1], simple.shape[0])))
    return float(ssim(simple, out, data_range=255))


def make_grid(images: list[tuple[str, Image.Image]], out_path: Path, cols: int = 3):
    """Save a labelled side-by-side grid of all outputs."""
    import matplotlib.pyplot as plt
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    axes = np.array(axes).flatten()
    for ax, (label, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(label, fontsize=8, wrap=True)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grid saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True, help="CT95 or CT106")
    parser.add_argument("--steps",  type=int, default=20, help="Diffusion steps (default 20)")
    parser.add_argument("--size",   type=int, default=512)
    args = parser.parse_args()

    # Resolve volume directory
    base = Path("data/renders/simple")
    matches = list(base.glob(f"*{args.volume}*"))
    if not matches:
        raise FileNotFoundError(f"No render directory found matching '{args.volume}'")
    vol_dir = matches[0]
    print(f"Volume dir: {vol_dir}")

    out_dir = vol_dir / "controlnet"
    out_dir.mkdir(exist_ok=True)

    depth_path  = vol_dir / "depth.png"
    simple_path = vol_dir / "color.png"

    device = pick_device()
    pipe   = load_pipeline(device)
    depth_img = prepare_depth(depth_path, args.size)

    generator = torch.Generator(device="cpu").manual_seed(42)  # CPU generator works on all devices

    results  = []
    grid_imgs = []

    print(f"\nRunning {len(PROMPTS)} prompts (steps={args.steps}, seed=42) …\n")

    for slug, prompt in PROMPTS:
        print(f"  [{slug}]  '{prompt[:60]}'")
        t0 = time.perf_counter()

        out = pipe(
            prompt          = prompt,
            negative_prompt = NEGATIVE_PROMPT,
            image           = depth_img,
            num_inference_steps = args.steps,
            generator       = generator,
            controlnet_conditioning_scale = 1.0,
        ).images[0]

        elapsed = time.perf_counter() - t0

        # Save individual output
        prompt_dir = out_dir / slug
        prompt_dir.mkdir(exist_ok=True)
        out.save(prompt_dir / "output.png")

        # Geometry check: SSIM vs simple color render
        geo_ssim = compute_ssim(simple_path, out)

        print(f"    → {elapsed:.1f}s  |  SSIM(vs simple)={geo_ssim:.3f}")

        results.append({
            "slug":       slug,
            "prompt":     prompt,
            "ssim_vs_simple": round(geo_ssim, 4),
            "time_s":     round(elapsed, 1),
            "output":     str(prompt_dir / "output.png"),
        })
        grid_imgs.append((slug if prompt else "empty (baseline)", out))

        # Reset generator to same seed for next prompt (isolates prompt effect)
        generator = torch.Generator(device="cpu").manual_seed(42)

    # Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "volume":   args.volume,
            "steps":    args.steps,
            "device":   str(device),
            "model":    "runwayml/stable-diffusion-v1-5 + lllyasviel/control_v11f1p_sd15_depth",
            "seed":     42,
            "note_lpips": "LPIPS vs Mitsuba GT skipped — GT not yet rendered (Phase 3)",
            "results":  results,
        }, f, indent=2)
    print(f"\n  Metrics → {metrics_path}")

    # Save grid
    make_grid(grid_imgs, out_dir / "grid.png")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Prompt slug':<22}  {'SSIM vs simple':>14}  {'Time':>7}")
    print("=" * 70)
    for r in results:
        print(f"  {r['slug']:<20}  {r['ssim_vs_simple']:>14.3f}  {r['time_s']:>6.1f}s")
    print("=" * 70)
    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()
