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
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world, True
    return 0, 0, 1, False


def is_main(rank):
    return rank == 0


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full1024", choices=["proto512", "full1024", "rgb_only"])
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
    if args.resume:        c.train.resume = args.resume
    return c


# ── sample dump ───────────────────────────────────────────────────────────────
def save_samples(inp, fake, real, path, n=4):
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

    torch.manual_seed(cfg.train.seed + rank)
    np.random.seed(cfg.train.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.train.output_dir)
    if is_main(rank):
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        cfg.to_json(str(out_dir / "config.json"))
    log(rank, f"[config] {cfg.summary()}")
    log(rank, f"[dist] world={world} device={device}")

    # data
    ds = PairedRenderDataset(cfg, split="train")
    log(rank, f"[data] {len(ds.subjects)} subjects, {len(ds)} views, num_classes={ds.num_classes}")
    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=cfg.data.num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True, persistent_workers=cfg.data.num_workers > 0)

    # models
    G = define_G(cfg).to(device)
    D = define_D(cfg).to(device)
    if distributed:
        ddp_kw = dict(device_ids=[local_rank] if torch.cuda.is_available() else None,
                      bucket_cap_mb=cfg.train.bucket_cap_mb)
        G = DDP(G, **ddp_kw)
        D = DDP(D, **ddp_kw)

    lr = cfg.train.lr
    d_lr = lr * 4 if cfg.train.ttur else lr
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(cfg.train.beta1, cfg.train.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(cfg.train.beta1, cfg.train.beta2))

    criterion_gan = GANLoss(cfg.loss.gan_mode).to(device)
    vgg = VGGLoss(device) if cfg.loss.w_vgg > 0 else None
    lpips_net = build_lpips(device) if cfg.loss.w_lpips > 0 else None

    amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp else contextlib.nullcontext

    start_epoch = 0
    if cfg.train.resume and Path(cfg.train.resume).exists():
        ck = torch.load(cfg.train.resume, map_location=device)
        (G.module if distributed else G).load_state_dict(ck["G"])
        (D.module if distributed else D).load_state_dict(ck["D"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        start_epoch = ck.get("epoch", 0) + 1
        log(rank, f"[resume] from {cfg.train.resume} @ epoch {start_epoch}")

    accum = max(cfg.train.grad_accum, 1)
    n_layers_D, num_D = cfg.model.n_layers_D, cfg.model.num_D

    def maybe_no_sync(model, is_boundary):
        return model.no_sync() if (distributed and not is_boundary) else contextlib.nullcontext()

    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        G.train(); D.train()
        opt_G.zero_grad(set_to_none=True); opt_D.zero_grad(set_to_none=True)
        running = {}

        for step, batch in enumerate(loader):
            inp = batch["input"].to(device, non_blocking=True)
            real = batch["target"].to(device, non_blocking=True)
            is_boundary = ((step + 1) % accum == 0)

            # ---- Discriminator (D grads on; fake detached so G isn't updated) ----
            set_requires_grad(D, True)
            with maybe_no_sync(D, is_boundary):
                with autocast():
                    fake = G(inp)
                    real_pair = torch.cat([inp, real], dim=1)
                    fake_pair_d = torch.cat([inp, fake.detach()], dim=1)
                    pred_real = D(real_pair)
                    pred_fake_d = D(fake_pair_d)
                    loss_D = 0.5 * (criterion_gan(pred_real, True) + criterion_gan(pred_fake_d, False))
                (loss_D / accum).backward()

            # ---- Generator (freeze D so its params get no grad from loss_G,
            #      but gradients still flow THROUGH D back to G) ----
            set_requires_grad(D, False)
            with maybe_no_sync(G, is_boundary):
                with autocast():
                    fake_pair_g = torch.cat([inp, fake], dim=1)
                    pred_fake_g = D(fake_pair_g)
                    l_gan = criterion_gan(pred_fake_g, True) * cfg.loss.w_gan
                    l_fm = feature_matching_loss(pred_fake_g, pred_real, num_D, n_layers_D) * cfg.loss.w_fm
                    l_l1 = torch.nn.functional.l1_loss(fake, real) * cfg.loss.w_l1
                    l_vgg = vgg(fake, real) * cfg.loss.w_vgg if vgg is not None else torch.zeros((), device=device)
                    l_lpips = (lpips_net(fake, real).mean() * cfg.loss.w_lpips
                               if lpips_net is not None else torch.zeros((), device=device))
                    loss_G = l_gan + l_fm + l_l1 + l_vgg + l_lpips
                (loss_G / accum).backward()

            if is_boundary:
                opt_D.step(); opt_G.step()
                opt_D.zero_grad(set_to_none=True); opt_G.zero_grad(set_to_none=True)

            for k, v in dict(D=loss_D, G=loss_G, gan=l_gan, fm=l_fm, l1=l_l1, vgg=l_vgg, lpips=l_lpips).items():
                running[k] = running.get(k, 0.0) + float(v.detach())

            if is_main(rank) and (step % cfg.train.log_every == 0):
                msg = " ".join(f"{k}={running[k]/(step+1):.3f}" for k in ["D", "G", "l1", "vgg", "fm", "gan"])
                log(rank, f"[e{epoch:03d} {step:05d}/{len(loader)}] {msg}")

        # ---- end epoch: samples + checkpoint (rank 0) ----
        if is_main(rank) and (epoch % cfg.train.sample_every == 0):
            G.eval()
            with torch.no_grad(), autocast():
                fake = G(inp)
            save_samples(inp, fake, real, out_dir / "samples" / f"e{epoch:03d}.png", cfg.train.n_sample_images)
        if is_main(rank) and (epoch % cfg.train.save_every == 0 or epoch == cfg.train.epochs - 1):
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
