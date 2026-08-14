"""
train.py — DDP training for the deterministic G-buffer neural renderer (3rd.md Stage 1).

Launch on 4× L4 with torchrun:

    torchrun --standalone --nproc_per_node=4 scripts/training/train.py \
        --preset full1024 --data-root data/training_dataset

Single-GPU / CPU debug:

    python scripts/training/train.py --preset proto512 --epochs 1

Implements: pix2pixHD generator + multi-scale PatchGAN, L1 + VGG (+optional LPIPS)
+ feature-matching + small LSGAN, bf16 autocast, gradient accumulation, DDP with
raised all-reduce bucket size, checkpoint/resume, periodic sample dumps.
"""
from __future__ import annotations

import os
import sys
import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make local modules importable

from config import preset, ALL_INPUT_BUFFERS
from dataset import PairedRenderDataset
from networks import define_G, define_D
from losses import GANLoss, VGGLoss, feature_matching_loss, build_lpips


# ── distributed helpers ───────────────────────────────────────────────────────
def setup_dist():
    # If torchrun set RANK/WORLD_SIZE env vars, we're in multi-GPU mode.
    # NCCL is the fast GPU-to-GPU backend; gloo is the CPU fallback.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()          # global index of this process (0..world-1)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index on this machine
        world = dist.get_world_size()   # total number of GPUs
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)  # pin this process to its GPU
        return rank, local_rank, world, True
    # Single-GPU or CPU: fake the distributed state so the rest of the code is identical
    return 0, 0, 1, False


def is_main(rank):
    return rank == 0


def set_requires_grad(model, flag: bool):
    # Freeze or unfreeze all parameters in a model.
    # Used to stop D's weights updating when we compute the generator loss,
    # and to stop G's weights updating when we train the discriminator.
    for p in model.parameters():
        p.requires_grad = flag


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full1024",
                   choices=["proto512", "full1024", "rgb_only", "overfit"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--netG", default=None, choices=["global", "local"])
    p.add_argument("--target", default=None, choices=["preview_png", "exr_agx", "exr_linear"])
    p.add_argument("--normals-space", default=None, choices=["world", "camera"])
    p.add_argument("--input-buffers", default=None,
                   help="comma list from: " + ",".join(ALL_INPUT_BUFFERS))
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--subjects", default=None,
                   help="comma list of subject ids to train on, ignoring the split (overfit test)")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="cap the (post-split) subject list to the first N (0 = all)")
    p.add_argument("--exclude-file", default=None,
                   help="JSON list of 'subject/view' to drop (e.g. results/framing/exclude.json)")
    # Loss-weight overrides (handy for a pure-regression overfit test: --w-gan 0 --w-fm 0).
    p.add_argument("--w-l1", type=float, default=None)
    p.add_argument("--w-vgg", type=float, default=None)
    p.add_argument("--w-gan", type=float, default=None)
    p.add_argument("--w-fm", type=float, default=None)
    p.add_argument("--w-lpips", type=float, default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def build_config(args):
    c = preset(args.preset)
    if args.data_root:     c.data.root = args.data_root
    if args.output_dir:    c.train.output_dir = args.output_dir
    if args.epochs:        c.train.epochs = args.epochs
    if args.batch_size:    c.train.batch_size = args.batch_size
    if args.grad_accum:    c.train.grad_accum = args.grad_accum
    if args.size:          c.data.size = args.size; c.data.load_size = max(c.data.load_size, args.size)
    if args.lr:            c.train.lr = args.lr
    if args.netG:          c.model.netG = args.netG
    if args.target:        c.data.target = args.target
    if args.normals_space: c.data.normals_space = args.normals_space
    if args.input_buffers: c.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]
    if args.num_workers is not None: c.data.num_workers = args.num_workers
    if args.subjects:      c.data.subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if args.max_subjects is not None: c.data.max_subjects = args.max_subjects
    if args.exclude_file: c.data.exclude_file = args.exclude_file
    if args.w_l1 is not None:    c.loss.w_l1 = args.w_l1
    if args.w_vgg is not None:   c.loss.w_vgg = args.w_vgg
    if args.w_gan is not None:   c.loss.w_gan = args.w_gan
    if args.w_fm is not None:    c.loss.w_fm = args.w_fm
    if args.w_lpips is not None: c.loss.w_lpips = args.w_lpips
    if args.resume:        c.train.resume = args.resume
    return c


# ── sample dump ───────────────────────────────────────────────────────────────
def save_samples(inp, fake, real, path, n=4):
    # Saves a contact sheet every epoch so you can visually track training progress.
    # Each row = one image in the batch: [input seg_rgb | network output | GT target]
    import cv2
    def to_img(t):  # CxHxW [-1,1] → HxWx3 uint8 RGB
        a = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
        a = a.transpose(1, 2, 0)
        if a.shape[2] == 1: a = np.repeat(a, 3, axis=2)
        return a[:, :, :3]
    rows = []
    for i in range(min(n, inp.shape[0])):
        seg = to_img(inp[i, :3])             # first 3 input ch = seg_rgb (usually)
        rows.append(np.concatenate([seg, to_img(fake[i]), to_img(real[i])], axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


# ── train ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    rank, local_rank, world, distributed = setup_dist()
    cfg = build_config(args)

    # Each GPU gets a different seed so they don't sample the same augmentations.
    torch.manual_seed(cfg.train.seed + rank)
    np.random.seed(cfg.train.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.train.output_dir)
    # Only rank 0 creates directories and saves config — avoids race conditions.
    if is_main(rank):
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        cfg.to_json(str(out_dir / "config.json"))
    log(rank, f"[config] {cfg.summary()}")
    log(rank, f"[dist] world={world} device={device}")

    # data
    ds = PairedRenderDataset(cfg, split="train")
    log(rank, f"[data] {len(ds.subjects)} subjects, {len(ds)} views, num_classes={ds.num_classes}")
    # DistributedSampler ensures each GPU sees a different subset of the data each epoch.
    # drop_last=True avoids uneven batch sizes at the end of an epoch across GPUs.
    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=cfg.data.num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True, persistent_workers=cfg.data.num_workers > 0)

    # models
    G = define_G(cfg).to(device)  # Generator: G-buffer stack → RGB image
    D = define_D(cfg).to(device)  # Discriminator: (input | image) → patch realism scores
    if distributed:
        # DDP wraps each model so gradients are averaged across GPUs after each backward pass.
        # bucket_cap_mb controls how many gradients are batched per all-reduce — larger = fewer
        # communication rounds, important when GPUs are connected via PCIe (not NVLink).
        ddp_kw = dict(device_ids=[local_rank] if torch.cuda.is_available() else None,
                      bucket_cap_mb=cfg.train.bucket_cap_mb)
        G = DDP(G, **ddp_kw)
        D = DDP(D, **ddp_kw)

    lr = cfg.train.lr
    # TTUR (two-timescale update rule): running D at 4× G's lr can stabilise GAN training.
    d_lr = lr * 4 if cfg.train.ttur else lr
    # Adam with beta1=0.5 is standard for GANs (lower momentum = less oscillation).
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(cfg.train.beta1, cfg.train.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(cfg.train.beta1, cfg.train.beta2))

    criterion_gan = GANLoss(cfg.loss.gan_mode).to(device)  # LSGAN patch loss
    vgg = VGGLoss(device) if cfg.loss.w_vgg > 0 else None  # perceptual loss (frozen VGG19)
    lpips_net = build_lpips(device) if cfg.loss.w_lpips > 0 else None  # optional extra perceptual

    amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
    # Mixed precision: forward/backward in bf16 (faster, less memory), optimizer step in fp32.
    # bf16 preferred over fp16 on L4/Ada — wider dynamic range, no loss scaling needed.
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp else contextlib.nullcontext

    start_epoch = 0
    if cfg.train.resume and Path(cfg.train.resume).exists():
        # Resume from checkpoint: restores model weights, optimizer state (momentum buffers,
        # lr schedules), and epoch counter so training continues seamlessly.
        ck = torch.load(cfg.train.resume, map_location=device)
        (G.module if distributed else G).load_state_dict(ck["G"])
        (D.module if distributed else D).load_state_dict(ck["D"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        start_epoch = ck.get("epoch", 0) + 1
        log(rank, f"[resume] from {cfg.train.resume} @ epoch {start_epoch}")

    # Gradient accumulation: instead of updating every step, accumulate gradients over
    # `accum` steps then update once. Effective batch = batch_size × accum × n_gpus.
    accum = max(cfg.train.grad_accum, 1)
    n_layers_D, num_D = cfg.model.n_layers_D, cfg.model.num_D

    def maybe_no_sync(model, is_boundary):
        # In DDP, skip the expensive all-reduce on non-boundary steps (gradient accumulation).
        # Only sync gradients across GPUs on the last accumulation step.
        return model.no_sync() if (distributed and not is_boundary) else contextlib.nullcontext()

    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            # Must call set_epoch so each epoch shuffles differently across GPUs.
            sampler.set_epoch(epoch)
        G.train(); D.train()
        opt_G.zero_grad(set_to_none=True); opt_D.zero_grad(set_to_none=True)
        running = {}  # accumulates loss values for logging

        for step, batch in enumerate(loader):
            inp = batch["input"].to(device, non_blocking=True)   # G-buffer stack (8ch)
            real = batch["target"].to(device, non_blocking=True)  # Cycles GT render (3ch)
            is_boundary = ((step + 1) % accum == 0)  # True on the step where we actually update

            # ── Step 1: Train Discriminator ───────────────────────────────────
            # D learns to tell real renders from G's fakes.
            # fake.detach() stops gradients flowing back into G during this step.
            set_requires_grad(D, True)
            with maybe_no_sync(D, is_boundary):
                with autocast():
                    fake = G(inp)
                    # D sees input+image concatenated — it knows what the G-buffer was,
                    # so it can't be fooled by a plausible-looking but wrong-anatomy render.
                    real_pair = torch.cat([inp, real], dim=1)
                    fake_pair_d = torch.cat([inp, fake.detach()], dim=1)
                    pred_real = D(real_pair)
                    pred_fake_d = D(fake_pair_d)
                    # LSGAN: D is trained to output 1 for real, 0 for fake. Averaged 50/50.
                    loss_D = 0.5 * (criterion_gan(pred_real, True) + criterion_gan(pred_fake_d, False))
                (loss_D / accum).backward()

            # ── Step 2: Train Generator ───────────────────────────────────────
            # G tries to fool D (adversarial) AND match the GT pixel/perceptually.
            # D weights are frozen so only G's weights are updated here,
            # but gradients still flow THROUGH D's forward pass back into G.
            set_requires_grad(D, False)
            with maybe_no_sync(G, is_boundary):
                with autocast():
                    fake_pair_g = torch.cat([inp, fake], dim=1)
                    pred_fake_g = D(fake_pair_g)
                    # GAN loss: G wants D to think its output is real (target=True)
                    l_gan = criterion_gan(pred_fake_g, True) * cfg.loss.w_gan
                    # Feature matching: G's fake should activate D's layers similarly to real.
                    # This stabilises training and helps G learn structure, not just pixel values.
                    l_fm = feature_matching_loss(pred_fake_g, pred_real, num_D, n_layers_D) * cfg.loss.w_fm
                    # L1: direct pixel-level supervision — the main fidelity anchor.
                    l_l1 = torch.nn.functional.l1_loss(fake, real) * cfg.loss.w_l1
                    # VGG perceptual: compares deep features, restores texture sharpness.
                    l_vgg = vgg(fake, real) * cfg.loss.w_vgg if vgg is not None else torch.zeros((), device=device)
                    # LPIPS: optional extra perceptual loss (off by default, w_lpips=0).
                    l_lpips = (lpips_net(fake, real).mean() * cfg.loss.w_lpips
                               if lpips_net is not None else torch.zeros((), device=device))
                    loss_G = l_gan + l_fm + l_l1 + l_vgg + l_lpips
                (loss_G / accum).backward()

            # NaN/Inf guard — abort immediately rather than silently corrupt the checkpoint.
            if not (torch.isfinite(loss_G) and torch.isfinite(loss_D)):
                log(rank, f"[FATAL] non-finite loss at e{epoch} step {step} "
                          f"(G={float(loss_G)}, D={float(loss_D)}) — aborting. "
                          f"Check LR / inputs / target range.")
                if distributed:
                    dist.destroy_process_group()
                sys.exit(1)

            # Only update weights once we've accumulated enough gradients.
            if is_boundary:
                opt_D.step(); opt_G.step()
                opt_D.zero_grad(set_to_none=True); opt_G.zero_grad(set_to_none=True)

            for k, v in dict(D=loss_D, G=loss_G, gan=l_gan, fm=l_fm, l1=l_l1, vgg=l_vgg, lpips=l_lpips).items():
                running[k] = running.get(k, 0.0) + float(v.detach())

            if is_main(rank) and (step % cfg.train.log_every == 0):
                msg = " ".join(f"{k}={running[k]/(step+1):.3f}" for k in ["D", "G", "l1", "vgg", "fm", "gan"])
                log(rank, f"[e{epoch:03d} {step:05d}/{len(loader)}] {msg}")

        # ── End of epoch: save samples and checkpoint (rank 0 only) ─────────
        if is_main(rank) and (epoch % cfg.train.sample_every == 0):
            G.eval()
            with torch.no_grad(), autocast():
                fake = G(inp)  # run on last batch — quick visual sanity check
            save_samples(inp, fake, real, out_dir / "samples" / f"e{epoch:03d}.png", cfg.train.n_sample_images)
        if is_main(rank) and (epoch % cfg.train.save_every == 0 or epoch == cfg.train.epochs - 1):
            # Save both G and D weights + optimizer states so training can be resumed exactly.
            # latest.pt is always overwritten — use epoch_NNN.pt for rollback points.
            ck = {"G": (G.module if distributed else G).state_dict(),
                  "D": (D.module if distributed else D).state_dict(),
                  "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                  "epoch": epoch, "cfg": cfg.summary()}
            torch.save(ck, out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            torch.save(ck, out_dir / "checkpoints" / "latest.pt")
            log(rank, f"[ckpt] saved epoch {epoch}")

    if distributed:
        dist.barrier(); dist.destroy_process_group()
    log(rank, "[done]")


if __name__ == "__main__":
    main()
