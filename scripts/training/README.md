# Training pipeline — deterministic G-buffer neural renderer

Implements the architecture from [`pipeline research/3rd.md`](../../pipeline%20research/3rd.md):
a **deterministic, geometry-buffer-conditioned pix2pixHD-style generator** that learns
the fixed-lighting Cycles forward-rendering function. It is trained with
**L1 + VGG (+optional LPIPS) + feature-matching + a small multi-scale PatchGAN**.

Why this and not diffusion / multi-view attention (the short version):
- Your lighting rig and geometry are **fixed and known**, inputs/outputs are **perfectly
  registered**, and the metric is **fidelity to the Cycles GT** (PSNR/SSIM/LPIPS).
- A deterministic function of view-consistent inputs (depth/normals/seg-ID/flat-RGB)
  is **view-consistent by construction** — no cross-view attention needed.
- 4× L4 (24 GB, no NVLink) trains pix2pixHD at 1024² comfortably; full SDXL/MV-Adapter
  does not. See 3rd.md §"Trainability".

## Data it expects

Produced by [`scripts/training_dataset/generate_training_dataset.py`](../training_dataset/generate_training_dataset.py):

```
data/training_dataset/
  {subject}/                         e.g. s0000
    v{NN}_az{az}_el{el}/             20 views per subject
      render.exr        multilayer scene-linear: Image, Depth, Normal, IndexOB, ...
      seg.png           flat EEVEE semantic render  → the "simple input"
      rgb_preview.png   AgX + fog-glow + chromatic aberration (default GT target)
      meta.json         K, cam_to_world, depth_metric_ref, scene_scale, ...
    tissue_ids.json
    summary.json
```

## Inputs and target

**Input** = configurable stack (channel order = list order), each normalised to `[-1,1]`:

| buffer    | ch | source                        | notes |
|-----------|----|-------------------------------|-------|
| `seg_rgb` | 3  | `seg.png`                     | the flat "simple input" |
| `depth`   | 1  | `render.exr` Depth            | `/depth_metric_ref`, clipped |
| `normals` | 3  | `render.exr` Normal           | world-space default; `--normals-space camera` rotates via pose |
| `segid`   | 1  | `render.exr` IndexOB          | per-organ id, normalised |

Default stack = all four (3rd.md PRIMARY). For the RGB-only Stage-0 ablation use the
`rgb_only` preset or `--input-buffers seg_rgb`.

**Target** (`--target`, see `config.DataConfig.target`):
- `preview_png` *(default)* — `rgb_preview.png`, **exact** AgX + glare + CA. Network
  reproduces the full cinematic look. No tonemapper approximation.
- `exr_agx` — `render.exr` Image pass, **approximate** AgX at load (no glare/CA).
- `exr_linear` — raw scene-linear radiance, crude display clamp (not recommended; LPIPS/
  PSNR/SSIM expect display-referred images).

**"Can I use both?"** Yes — `--target` is a one-line switch, so you can train a
`preview_png` model and an `exr_agx`/`exr_linear` variant and compare PSNR/SSIM/LPIPS.
A combined linear-grounding + perceptual loss (predict linear, supervise in both linear
and tonemapped space) is a clean Stage-2 extension; left out of v1 to avoid baking in an
inexact differentiable AgX.

## Install

```bash
venv/bin/pip install -r scripts/training/requirements-training.txt
# (use the CUDA torch build on the L4 server, not the CPU wheel)
```

## Run

```bash
# Stage 0 — 512² prototype, fast sanity (3rd.md week 1)
bash scripts/training/run_train_4gpu.sh proto512

# Stage 1 — the deliverable: native 1024², full G-buffer stack, DDP on 4× L4
bash scripts/training/run_train_4gpu.sh full1024

# single-GPU / CPU debug
venv/bin/python scripts/training/train.py --preset proto512 --epochs 1 --num-workers 0
```

Evaluate (use `--split train` until the split is implemented — see below):

```bash
venv/bin/python scripts/training/evaluate.py \
  --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
  --preset full1024 --split train
```

## Files

| file | role |
|------|------|
| `config.py`   | all hyperparameters + presets (`proto512`, `full1024`, `rgb_only`) |
| `exr.py`      | robust multilayer-EXR reader (runtime channel discovery) |
| `dataset.py`  | paired G-buffer→photoreal dataset, minimal paired augmentation, AgX tonemap |
| `networks.py` | pix2pixHD `GlobalGenerator` / `LocalEnhancer` + multi-scale PatchGAN |
| `losses.py`   | GAN / VGG / feature-matching / LPIPS + reprojection placeholder |
| `train.py`    | DDP training loop (bf16, grad-accum, checkpoint/resume, sample dumps) |
| `evaluate.py` | PSNR/SSIM/LPIPS + cross-view reprojection gate (placeholder) |

## Placeholders (intentionally not done yet)

1. **Train/val/test split** — `splits.py` puts **every subject in `train`**. Implement a
   deterministic **volume-level** 85/7.5/7.5 split later (never split by view). A reference
   exists in `scripts/training_dataset/split_training_dataset.py` / `data/splits.json`.
2. **Reprojection consistency loss** (`losses.ReprojectionConsistencyLoss`, `w_reproj`) —
   Stage-2 only, needs paired-view batching. Add **only if** the Stage-1 cross-view gate
   fails (3rd.md: should pass by construction with fixed lighting).
3. **Cross-view reprojection eval gate** (`evaluate.cross_view_reprojection_gate`) — the
   advisor's consistency check; same paired-view machinery.

## Staged plan (3rd.md)

- **Stage 0** — `proto512` on a small subset; confirm loss decreases, no anatomy distortion.
- **Stage 1** — `full1024` on all data; report held-out PSNR/SSIM/LPIPS + cross-view gate.
  If the gate passes (it should), **stop — do not add complexity.**
- **Stage 2** — only if the gate fails: enable the reprojection loss / world-space normals.
- **Stage 3** — only if reviewers demand more texture realism: an SD1.5-ControlNet
  comparison with a fixed per-volume seed (documented realism-vs-fidelity aside).
