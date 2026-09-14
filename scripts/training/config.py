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
#   position: object-space XYZ, unprojected from depth    — texture coordinate  (3 + 6*bands ch)
#
# Why `position` exists. v25 samples every organ texture through
# ShaderNodeTexCoord's *Object* output in BOX projection, so in the ground truth the
# vessel/mottle pattern at a pixel is a deterministic function of the 3D object-space
# point. seg_rgb and segid give identity, normals give orientation, depth gives
# distance — but recovering the 3D point additionally needs the camera pose, which was
# never an input. The network was therefore asked to predict a function of a variable
# it cannot observe, and its best answer is the average over all consistent positions:
# a smooth, texture-free surface. That is exactly the artefact seen in the renders.
# `position` closes the gap, and needs no re-rendering — meta.json already stores K and
# cam_to_world. See dataset.py::_load_position.
#
#   texture : the v25 albedo / normal-map / roughness tiles, BOX-projected at the
#             CORRECTED object-space position (texture_lookup.py)                (7 ch)
#
# `texture` supersedes `position` as the way to deliver organ texture. `position` (as
# trained in gbuf_pos1024) uses the legacy unprojection, which ignores the camera's lens
# shift and the emission-pass gain, so it was 7-30 mm off the surface; and even a correct
# coordinate would leave the network to memorise 18 texture sets. `texture` hands it the
# looked-up texture directly. See camera_model.py and thesis_context.md §4b.
#
#   glint   : the vessel glints v25 renders, shaded analytically (glint_model.py):
#             [vessel glints, plain geometric highlights]                       (2 ch)
#
# Why `glint`: the vessels exist only in the normal-map tiles and appear in the GT as
# specular glints. Given `texture` alone (the raw tile colours), the network drew none —
# not even on its training views (preflight_tex: M3 0.008 vs GT 0.051). The glint term
# depends on the tangent frame, camera and lights, all recovered and validated:
# texture_learnability.py `glint` found the Cycles-exact tangent frame locks to the GT
# streaks (partial r 0.149, half-tile null 0.030; 2.8x the raw-ridge proxy).
ALL_INPUT_BUFFERS = ["seg_rgb", "depth", "normals", "segid", "position", "texture", "glint"]

INPUT_CHANNELS = {"seg_rgb": 3, "depth": 1, "normals": 3, "segid": 1, "position": 3,
                  "texture": 7, "glint": 2}


def position_channels(bands: int) -> int:
    """Raw XYZ plus sin/cos Fourier bands (2 fns * 3 axes per band)."""
    return 3 + 6 * int(bands)


def buffer_channels(name: str, position_bands: int = 0) -> int:
    return position_channels(position_bands) if name == "position" else INPUT_CHANNELS[name]


@dataclass
class DataConfig:
    # Root produced by scripts/training_dataset/generate_training_dataset.py.
    # Layout: {root}/{subject}/v{NN}_az.._el../{render.exr, seg.png, rgb_preview.png, meta.json}
    root: str = "data/training_dataset"

    # Which conditioning buffers feed the generator (order = channel order).
    # Default = full G-buffer stack (3rd.md PRIMARY). Set to ["seg_rgb"] for an
    # RGB-only Stage-0 ablation.
    input_buffers: List[str] = field(default_factory=lambda: ["seg_rgb", "depth", "normals", "segid"])

    # Fourier bands added to the `position` buffer (ignored unless it is used).
    # A CNN has a strong low-frequency bias, so a raw coordinate ramp is a poor basis
    # for a fine, near-periodic texture; sin/cos at several wavelengths is the standard
    # remedy (positional encoding) and suits BOX-projected *tileable* textures, which
    # are literally periodic in object space. Bands run POSITION_BASE_LAMBDA_MM / 4**k,
    # i.e. 256 / 64 / 16 mm. 0 = raw XYZ only (cheapest, 3 ch).
    #
    # Why 3 and not 4: depth.exr was written as HALF float (Blender ignored
    # color_depth='32'), so Z is quantised to ~1.95 mm at a ~2 m camera distance and the
    # unprojected coordinate inherits that step. A 4 mm band would then get ~2 samples
    # per cycle — at Nyquist, encoding quantisation noise rather than position. 16 mm
    # gives ~8 samples per cycle and is safe. dataset.py measures the real step and
    # refuses an aliased setting. Same reasoning as v25's _safe_detail octave cap.
    position_bands: int = 3

    # `texture` buffer: MUST be the tiles the dataset was rendered with (synthesis is not
    # reproducible — see texture_lookup.py); verify against MD5SUMS.txt beside them.
    tex_dir: str = "data/renders/textures/tissue"
    tex_mm: float = 80.0                # generate_training_dataset.py --tex_mm

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

    # ── Split (see splits.py) ────────────────────────────────────────────────
    # Deterministic hash-based, volume-level, STABLE under dataset growth: each
    # subject's bucket depends only on its own id, so new subjects slot in without
    # reshuffling existing train/val/test (the dataset is still being generated).
    val_frac: float = 0.075
    test_frac: float = 0.075
    split_seed: int = 1337
    # 0 = keep all subjects. Low-tissue subjects were confirmed to be legitimate
    # limited-FOV CT scans (coherent upper-chest / pelvis clusters), not broken
    # extractions — so we keep them. Raise only to exclude truly-degenerate cases.
    min_tissues: int = 0

    # Debug/overfit knobs (do not use for real runs):
    #   subjects     — train on EXACTLY these subject ids, ignoring the split
    #                  (for the overfit sanity test, e.g. ["s0000", "s0001"]).
    #   max_subjects — cap the (post-split) subject list to the first N (0 = all).
    subjects: Optional[List[str]] = None
    max_subjects: int = 0

    # Per-view exclude list: path to a JSON array of "subject/view" strings
    # (e.g. results/framing/exclude.json from score_framing.py) — badly-framed
    # views are dropped from every split. None = use all views.
    exclude_file: Optional[str] = None


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

    # Generator 2x upsampling: "transpose" (original pix2pixHD, every checkpoint so far)
    # or "resize" (nearest + conv; no checkerboard lattice). See networks.up_layers.
    upsample: str = "transpose"


@dataclass
class LossConfig:
    # 3rd.md weights: L1 anchors fidelity (no hallucination); VGG/LPIPS restores
    # texture; FM stabilises; small PatchGAN adds sharpness only.
    #
    # NOTE on texture: for detail the network cannot predict from its inputs, L1 is
    # minimised by emitting the conditional mean — i.e. a smooth surface — and only the
    # GAN term pushes back, here at 1 against L1's 10. pix2pixHD as published uses NO
    # L1 at all and weights VGG at 10. If organ texture comes out smooth, this 10:1
    # tilt toward smoothing is the first thing to change (--w-l1 1 --w-vgg 10).
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
        return sum(buffer_channels(b, self.data.position_bands)
                   for b in self.data.input_buffers)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def summary(self) -> str:
        pos = (f" pos_bands={self.data.position_bands}"
               if "position" in self.data.input_buffers else "")
        if "texture" in self.data.input_buffers:
            pos += f" tex_dir={self.data.tex_dir} tex_mm={self.data.tex_mm:g}"
        return (f"netG={self.model.netG} ngf={self.model.ngf} up={self.model.upsample} | "
                f"in={self.data.input_buffers}({self.input_nc()}ch){pos} → out={self.model.output_nc}ch | "
                f"target={self.data.target} normals={self.data.normals_space} | "
                f"size={self.data.size} | losses L1={self.loss.w_l1} VGG={self.loss.w_vgg} "
                f"LPIPS={self.loss.w_lpips} FM={self.loss.w_fm} GAN={self.loss.w_gan} "
                f"reproj={self.loss.w_reproj}")


# ── Presets ───────────────────────────────────────────────────────────────────
# Every preset name, in one place. The argparse `choices=` lists in train.py,
# evaluate.py, infer.py and eval_common.py used to hardcode this separately, so adding
# a preset made config.preset() accept it while the CLIs still rejected it — which is
# exactly how rgb_1024 failed after being added correctly.
PRESET_NAMES = ("proto512", "full1024", "rgb_1024", "gbuf_pos1024", "full1024_glint",
                "preflight_tex", "preflight_ctrl", "preflight_glint", "rgb_only", "overfit")


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
    if name == "rgb_1024":
        # The RGB-only ablation AT FULL RESOLUTION — this is what results/training_runs/
        # rgb_baseline was actually trained with, and no preset described it.
        #
        # It matters that this derives from full1024 and not from rgb_only: rgb_only is
        # 512 px, so evaluating the 1024 px rgb_baseline checkpoint under it would
        # compare a different resolution AND a different input stack at once, which
        # confounds the ablation. Deriving from full1024 changes exactly one thing —
        # the input buffers — so a paired comparison against full1024 isolates the
        # contribution of the G-buffers.
        #
        # Without this, loading rgb_baseline under PRESET=full1024 fails with
        #   size mismatch for model.1.weight: copying a param with shape [64, 3, 7, 7]
        #   from checkpoint, the shape in current model is [64, 8, 7, 7]
        # because full1024 builds an 8-channel input (seg_rgb 3 + depth 1 + normals 3
        # + segid 1) while the checkpoint has 3.
        c = preset("full1024")
        c.data.input_buffers = ["seg_rgb"]
        c.train.output_dir = "results/training_runs/rgb_baseline"
        return c
    if name == "gbuf_pos1024":
        # full1024 PLUS the object-space position buffer — one change against full1024,
        # so a paired comparison isolates what the texture coordinate is worth.
        # Needs no dataset regeneration (position is unprojected from depth.exr using
        # meta.json's K / cam_to_world). See ALL_INPUT_BUFFERS above for the rationale.
        c = preset("full1024")
        c.data.input_buffers = ["seg_rgb", "depth", "normals", "segid", "position"]
        c.train.output_dir = "results/training_runs/gbuf_pos1024"
        return c
    if name == "full1024_glint":
        # The full retrain chosen from pre-flight #3 (thesis_context.md §4b): the full1024
        # training schedule on the full dataset, with the recipe that registered the most
        # vessel detail on held-out views —
        #   inputs   full1024's 8 ch + `texture` (7) + `glint` (2) = 17 ch
        #   upsample resize (removes the ConvTranspose2d checkerboard)
        #   losses   pix2pixHD objective: no L1, VGG 10, FM 10, GAN 1
        # Same batch 1 x grad_accum 4 x 4 GPUs and 61 epochs as full1024, so the two runs
        # differ only in this recipe.
        c = preset("full1024")
        c.model.upsample = "resize"
        c.data.input_buffers = ["seg_rgb", "depth", "normals", "segid", "texture", "glint"]
        c.loss.w_l1 = 0.0
        c.loss.w_vgg = 10.0
        c.data.num_workers = 8
        c.train.epochs = 61
        c.train.output_dir = "results/training_runs/gbuf_glint"
        return c
    if name in ("preflight_tex", "preflight_ctrl"):
        # Pre-flight for the last full retrain (scripts/analysis/preflight.py). Two runs on
        # the SAME 2 subjects with the SAME held-out views, differing in exactly one thing:
        #   preflight_tex  = full1024 inputs + `texture`   (15 ch), resize upsampling
        #   preflight_ctrl = full1024 inputs               ( 8 ch), resize upsampling
        # Both use resize upsampling so the control isolates the texture input, and the
        # lattice check covers both. Pass --subjects and --exclude-file from the plan.
        # grad_accum=1: 30-ish training views need optimizer steps, not a large batch.
        c = preset("full1024")
        c.model.upsample = "resize"
        if name == "preflight_tex":
            c.data.input_buffers = ["seg_rgb", "depth", "normals", "segid", "texture"]
        c.data.num_workers = 4
        c.train.epochs = 150
        c.train.grad_accum = 1
        c.train.save_every = 25
        c.train.sample_every = 10
        c.train.log_every = 10
        c.train.output_dir = f"results/training_runs/{name}"
        return c
    if name == "preflight_glint":
        # Pre-flight #2: preflight_tex plus the analytic glint buffer (17 ch). Identical in
        # everything else — subjects, held-out views, losses, epochs, resize upsampling —
        # so the existing preflight_ctrl run stays a valid control and preflight_tex shows
        # what the glint channels add on top of the raw tiles. Glint shading costs ~1 s per
        # view on top of the texture lookup, hence more loader workers.
        c = preset("preflight_tex")
        c.data.input_buffers = ["seg_rgb", "depth", "normals", "segid", "texture", "glint"]
        c.data.num_workers = 8
        c.train.output_dir = f"results/training_runs/{name}"
        return c
    if name == "overfit":
        # Step-1 sanity: memorise a handful of subjects (pass --subjects / --max-subjects).
        # Small + fast + frequent samples; if this won't drive L1→~0 and fake→GT,
        # there is a data/loss/normalisation bug — fix before scaling up.
        c = preset("rgb_only")
        c.data.size = 256
        c.data.load_size = 256
        c.train.epochs = 400
        c.train.batch_size = 2
        c.train.grad_accum = 1
        c.train.log_every = 5
        c.train.sample_every = 5
        c.train.save_every = 100
        c.train.output_dir = "results/training_runs/overfit"
        return c
    raise ValueError(f"unknown preset '{name}' "
                     f"(choose: {', '.join(PRESET_NAMES)})")
