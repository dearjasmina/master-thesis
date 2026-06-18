"""
config.py — Configuration for the deterministic G-buffer neural renderer.

Architecture follows `pipeline research/3rd.md`: a deterministic, geometry-buffer
conditioned pix2pixHD-style generator (L1 + VGG/LPIPS + feature-matching + small
multi-scale PatchGAN) that learns the fixed-lighting Cycles forward-rendering
function. Cross-view consistency is "free" because a deterministic function of
view-consistent inputs (depth/normals/seg-ID/flat-RGB) yields view-consistent output.

All knobs live here. `train.py` / `evaluate.py` load a named preset and allow
per-flag CLI overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ── Input conditioning buffers ────────────────────────────────────────────────
# Each entry maps to a loader in dataset.py. Channels are concatenated in the
# order listed below to form the generator input tensor.
#   seg_rgb : flat EEVEE semantic render (seg.png)        — the "simple input"   (3 ch)
#   depth   : metric Z from render.exr, normalised        — geometry            (1 ch)
#   normals : surface normals from render.exr             — geometry            (3 ch)
#   segid   : per-organ pass_index from render.exr        — organ identity      (1 ch)
ALL_INPUT_BUFFERS = ["seg_rgb", "depth", "normals", "segid"]

INPUT_CHANNELS = {"seg_rgb": 3, "depth": 1, "normals": 3, "segid": 1}


@dataclass
class DataConfig:
    # Root produced by scripts/training_dataset/generate_training_dataset.py.
    # Layout: {root}/{subject}/v{NN}_az.._el../{render.exr, seg.png, rgb_preview.png, meta.json}
    root: str = "data/training_dataset"

    # Which conditioning buffers feed the generator (order = channel order).
    # Default = full G-buffer stack (3rd.md PRIMARY). Set to ["seg_rgb"] for an
    # RGB-only Stage-0 ablation.
    input_buffers: List[str] = field(default_factory=lambda: ["seg_rgb", "depth", "normals", "segid"])

    # Ground-truth target. See README "Target choice".
    #   preview_png : rgb_preview.png — AgX + fog-glow + chromatic aberration baked in (EXACT, default).
    #   exr_agx     : render.exr Image pass, AgX-tonemapped at load (APPROXIMATE AgX, no glare/CA).
    #   exr_linear  : render.exr Image pass, raw scene-linear radiance (losses in log space).
    target: str = "preview_png"

    # Surface-normal frame: "world" (identical per surface point across views →
    # maximally consistency-promoting, default) or "camera" (view-space, lets the
    # net key view-dependent specular/Fresnel — 3rd.md primary; requires meta pose).
    normals_space: str = "world"

    # Image size fed to the network (square). Buffers are resized to this.
    size: int = 1024

    # Optional larger load size for random-crop augmentation (load_size >= size).
    # If equal to size, no crop. Kept minimal per 3rd.md ("over-augmenting hurts fidelity").
    load_size: int = 1024

    # Minimal augmentation only. Horizontal flip is OPT-IN and only valid with
    # normals_space="camera" (negates camera-space X); leave False otherwise.
    hflip: bool = False

    num_workers: int = 8


@dataclass
class ModelConfig:
    # "global" = pix2pixHD GlobalGenerator (good to ~512); "local" = +LocalEnhancer
    # for native 1024 (3rd.md trains natively at 1024²).
    netG: str = "local"
    ngf: int = 32              # 3rd.md: ngf 32
    n_downsample_global: int = 4
    n_blocks_global: int = 9
    n_local_enhancers: int = 1
    n_blocks_local: int = 3

    # Multi-scale PatchGAN (70×70 receptive field, 3rd.md).
    ndf: int = 64
    n_layers_D: int = 3
    num_D: int = 2             # number of discriminator scales
    norm: str = "instance"
    output_nc: int = 3         # RGB target


@dataclass
class LossConfig:
    # 3rd.md weights: L1 anchors fidelity (no hallucination); VGG/LPIPS restores
    # texture; FM stabilises; small PatchGAN adds sharpness only.
    w_l1: float = 10.0
    w_vgg: float = 1.0         # VGG19 perceptual
    w_lpips: float = 0.0       # optional LPIPS (lpips pkg). 0 = off; set >0 to add.
    w_fm: float = 10.0         # discriminator feature matching
    w_gan: float = 1.0         # adversarial (kept small)

    gan_mode: str = "lsgan"    # least-squares GAN (stable)

    # Stage-2 ONLY (see 3rd.md). Known-pose reprojection consistency. 0 = disabled.
    # Implemented as a documented placeholder in losses.py.
    w_reproj: float = 0.0


@dataclass
class TrainConfig:
    output_dir: str = "results/training_runs/run"
    epochs: int = 150
    batch_size: int = 1                 # per-GPU (1024² fits 1–2 in 24 GB)
    grad_accum: int = 4                 # effective batch = batch_size * grad_accum * n_gpu
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    ttur: bool = False                  # two-timescale: D lr = 4x G lr if True

    amp_dtype: str = "bf16"             # Ada/L4 supports bf16; avoids fp16 overflow
    bucket_cap_mb: int = 50             # DDP all-reduce bucket (3rd.md: raise for PCIe)

    save_every: int = 5                 # epochs
    sample_every: int = 1               # epochs — dump val/train sample grids
    log_every: int = 50                 # steps
    n_sample_images: int = 4
    seed: int = 1337
    resume: str = ""                    # checkpoint path to resume from


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def input_nc(self) -> int:
        return sum(INPUT_CHANNELS[b] for b in self.data.input_buffers)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def summary(self) -> str:
        return (f"netG={self.model.netG} ngf={self.model.ngf} | "
                f"in={self.data.input_buffers}({self.input_nc()}ch) → out={self.model.output_nc}ch | "
                f"target={self.data.target} normals={self.data.normals_space} | "
                f"size={self.data.size} | losses L1={self.loss.w_l1} VGG={self.loss.w_vgg} "
                f"LPIPS={self.loss.w_lpips} FM={self.loss.w_fm} GAN={self.loss.w_gan} "
                f"reproj={self.loss.w_reproj}")


# ── Presets ───────────────────────────────────────────────────────────────────
def preset(name: str) -> Config:
    """Named starting points. Override individual fields from the CLI in train.py."""
    if name == "proto512":
        # Stage 0 (3rd.md week 1): fast 512² prototype, global generator only.
        c = Config()
        c.data.size = 512
        c.data.load_size = 512
        c.model.netG = "global"
        c.train.epochs = 40
        c.train.batch_size = 2
        c.train.output_dir = "results/training_runs/proto512"
        return c
    if name == "full1024":
        # Stage 1 (the deliverable): native 1024², local enhancer, full G-buffer stack.
        c = Config()
        c.data.size = 1024
        c.data.load_size = 1024
        c.model.netG = "local"
        c.train.epochs = 150
        c.train.batch_size = 1
        c.train.grad_accum = 4
        c.train.output_dir = "results/training_runs/full1024"
        return c
    if name == "rgb_only":
        # Stage-0 ablation: flat RGB only (no G-buffers) — RGB-only baseline.
        c = preset("proto512")
        c.data.input_buffers = ["seg_rgb"]
        c.train.output_dir = "results/training_runs/rgb_only"
        return c
    raise ValueError(f"unknown preset '{name}' (choose: proto512, full1024, rgb_only)")
