"""
1.5.4 — Few-shot reference conditioning with IP-Adapter + ControlNet.

IP-Adapter injects a reference image's style via CLIP image embeddings,
while ControlNet-depth enforces geometry from the depth map.
This tests Q2: can a few photorealistic references act as style anchors
without full fine-tuning?

Reference images used: Luma Ray frame_0001.png (proxy for path-traced GT;
real Mitsuba GT references come in Phase 3 and should replace these).

Tests IP-Adapter scale in [0.3, 0.5, 0.7, 0.9, 1.0] with fixed seed=42.
Higher scale = more style weight, less text/depth fidelity.

Usage:
    python scripts/run_ip_adapter.py --volume CT95
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from ip_adapter import IPAdapter
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


SCALES = [0.3, 0.5, 0.7, 0.9, 1.0]
NEGATIVE = "blurry, low quality, distorted, deformed, painting, cartoon"


def pick_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def download_ip_adapter_weights() -> tuple[Path, Path]:
    """
    Download IP-Adapter weights and image encoder from HuggingFace.
    Returns (image_encoder_dir, ip_ckpt_path).
    Cached after first run — no re-download.
    """
    print("Downloading IP-Adapter weights (cached after first run) …")

    # IP-Adapter checkpoint
    ip_ckpt = hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="models/ip-adapter_sd15.bin",
    )

    # CLIP image encoder (ViT-H used by IP-Adapter)
    # IP-Adapter uses the image_encoder subfolder from the repo
    from huggingface_hub import snapshot_download
    encoder_dir = snapshot_download(
        repo_id="h94/IP-Adapter",
        allow_patterns=["models/image_encoder/**"],
    )
    image_encoder_path = Path(encoder_dir) / "models" / "image_encoder"

    print(f"  ip_ckpt          → {ip_ckpt}")
    print(f"  image_encoder    → {image_encoder_path}")
    return image_encoder_path, Path(ip_ckpt)


def load_pipeline(device: torch.device):
    dtype = torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe


def load_image(path: Path, size: int = 512) -> Image.Image:
    arr = cv2.imread(str(path))
    arr = cv2.resize(arr, (size, size))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def compute_ssim(color_path: Path, out: Image.Image, size: int = 512) -> float:
    ref = cv2.imread(str(color_path), cv2.IMREAD_GRAYSCALE)
    og  = np.array(out.convert("L").resize((size, size)))
    return float(ssim(ref, og, data_range=255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True, help="e.g. CT95")
    parser.add_argument("--steps",  type=int, default=20)
    parser.add_argument("--size",   type=int, default=512)
    args = parser.parse_args()

    base = Path("data/renders/simple")
    vol_dir = next(base.glob(f"*{args.volume}*"))
    out_dir = vol_dir / "ip_adapter"
    out_dir.mkdir(exist_ok=True)

    depth_path  = vol_dir / "depth.png"
    color_path  = vol_dir / "color.png"
    luma_frame  = vol_dir / "frames" / "frame_0001.png"

    if not luma_frame.exists():
        raise FileNotFoundError(
            f"Luma frame not found: {luma_frame}\n"
            "Run Luma Ray and extract frames first (see 1.5.1)."
        )

    device = pick_device()
    print(f"Device: {device}")

    image_encoder_path, ip_ckpt = download_ip_adapter_weights()

    print("\nLoading ControlNet pipeline …")
    pipe = load_pipeline(device)

    print("Wrapping with IP-Adapter …")
    ip_model = IPAdapter(
        sd_pipe=pipe,
        image_encoder_path=str(image_encoder_path),
        ip_ckpt=str(ip_ckpt),
        device=str(device),
    )

    # Style reference: Luma frame_0001 (photorealistic proxy for Mitsuba GT)
    style_ref  = load_image(luma_frame, args.size)
    depth_cond = load_image(depth_path,  args.size)

    # Convert depth to RGB if greyscale (ControlNet expects 3-channel)
    depth_arr = np.array(depth_cond)
    if depth_arr.ndim == 2 or depth_arr.shape[2] == 1:
        depth_arr = np.stack([depth_arr.squeeze()] * 3, axis=-1)
    depth_cond = Image.fromarray(depth_arr.astype(np.uint8))

    print(f"\nRunning IP-Adapter at scales {SCALES} (seed=42, steps={args.steps})\n")
    print(f"  Style reference : {luma_frame.name}")
    print(f"  Depth conditioning: {depth_path.name}\n")

    results   = []
    grid_imgs = []

    for scale in SCALES:
        print(f"  scale={scale}")
        t0 = time.perf_counter()

        outputs = ip_model.generate(
            pil_image=style_ref,
            prompt="",
            negative_prompt=NEGATIVE,
            scale=scale,
            num_samples=1,
            seed=42,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            # ControlNet kwargs passed through to the pipeline:
            image=depth_cond,
            controlnet_conditioning_scale=1.0,
        )
        out = outputs[0]
        elapsed = time.perf_counter() - t0

        geo_ssim = compute_ssim(color_path, out, args.size)
        slug = f"scale_{scale:.1f}".replace(".", "p")
        out.save(out_dir / f"{slug}.png")

        print(f"    → {elapsed:.1f}s  |  SSIM(vs color)={geo_ssim:.3f}")
        results.append({
            "scale":       scale,
            "ssim_vs_color": round(geo_ssim, 4),
            "time_s":      round(elapsed, 1),
            "output":      str(out_dir / f"{slug}.png"),
        })
        grid_imgs.append((f"scale={scale}", out))

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "volume":        args.volume,
            "steps":         args.steps,
            "seed":          42,
            "device":        str(device),
            "style_reference": str(luma_frame),
            "note_reference": "Luma Ray frame_0001 used as proxy for Mitsuba GT (Phase 3)",
            "note_lpips":    "LPIPS vs GT deferred — Mitsuba GT not yet rendered",
            "model":         "SD1.5 + ControlNet-depth + IP-Adapter (ip-adapter_sd15.bin)",
            "results":       results,
        }, f, indent=2)

    # ── Output grid ───────────────────────────────────────────────────────────
    n = len(grid_imgs) + 2  # +style ref +depth
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4))
    axes[0].imshow(style_ref);  axes[0].set_title("Style ref\n(Luma frame_0001)", fontsize=8); axes[0].axis("off")
    axes[1].imshow(depth_cond); axes[1].set_title("Depth conditioning", fontsize=8); axes[1].axis("off")
    for ax, (label, img) in zip(axes[2:], grid_imgs):
        ax.imshow(img); ax.set_title(label, fontsize=9); ax.axis("off")
    plt.suptitle(f"1.5.4 IP-Adapter few-shot — {args.volume}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Scale vs SSIM chart ───────────────────────────────────────────────────
    scales = [r["scale"] for r in results]
    scores = [r["ssim_vs_color"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scales, scores, "o-", color="#2d7ab5", linewidth=2, markersize=8)
    for x, y in zip(scales, scores):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("IP-Adapter scale (style weight)")
    ax.set_ylabel("SSIM vs color render (geometry proxy)")
    ax.set_title(f"IP-Adapter scale sweep — {args.volume}\n"
                 "Style reference: Luma Ray frame_0001 (proxy for Mitsuba GT)")
    ax.set_ylim(0, 1); ax.set_xticks(scales)
    plt.tight_layout()
    plt.savefig(out_dir / "scale_ssim.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"{'Scale':<8}  {'SSIM vs color':>13}  {'Time':>7}")
    print("=" * 45)
    for r in results:
        print(f"  {r['scale']:<6}  {r['ssim_vs_color']:>13.3f}  {r['time_s']:>6.1f}s")
    print("=" * 45)
    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
