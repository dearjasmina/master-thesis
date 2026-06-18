"""
losses.py — Loss terms for the deterministic G-buffer renderer (3rd.md).

Total G objective (weights in config.LossConfig):
    w_l1   * L1(fake, real)              fidelity anchor — prevents hallucination
    w_vgg  * VGG19_perceptual(fake,real) texture / structure
    w_lpips* LPIPS(fake, real)           optional perceptual (lpips pkg)
    w_fm   * discriminator feature match stabiliser
    w_gan  * LSGAN(fake)                 sharpness only (kept small)
    w_reproj * reprojection consistency  Stage-2 ONLY (placeholder below)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GANLoss(nn.Module):
    """LSGAN / vanilla GAN loss over a list (per scale) of patch-logit maps."""
    def __init__(self, gan_mode="lsgan"):
        super().__init__()
        self.gan_mode = gan_mode
        self.loss = nn.MSELoss() if gan_mode == "lsgan" else nn.BCEWithLogitsLoss()

    def __call__(self, preds_per_scale, target_is_real: bool):
        total = 0.0
        for scale in preds_per_scale:
            logits = scale[-1] if isinstance(scale, (list, tuple)) else scale
            tgt = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
            total = total + self.loss(logits, tgt)
        return total / max(len(preds_per_scale), 1)


class VGGLoss(nn.Module):
    """Weighted L1 over VGG19 feature maps (pix2pixHD perceptual loss)."""
    def __init__(self, device):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        slices, idxs = [], [2, 7, 12, 21, 30]
        prev = 0
        self.blocks = nn.ModuleList()
        for i in idxs:
            self.blocks.append(nn.Sequential(*[vgg[j] for j in range(prev, i)]))
            prev = i
        for p in self.parameters():
            p.requires_grad = False
        self.weights = [1/32, 1/16, 1/8, 1/4, 1.0]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
        self.eval()

    def _prep(self, x):  # [-1,1] → ImageNet-normalised
        x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        f, r = self._prep(fake), self._prep(real)
        loss = 0.0
        for block, w in zip(self.blocks, self.weights):
            f, r = block(f), block(r)
            loss = loss + w * F.l1_loss(f, r.detach())
        return loss


def feature_matching_loss(fake_feats, real_feats, num_D: int, n_layers_D: int):
    """L1 between discriminator intermediate features (per scale, per layer)."""
    loss = 0.0
    w = 4.0 / (n_layers_D + 1)
    for f_scale, r_scale in zip(fake_feats, real_feats):
        for f_map, r_map in zip(f_scale[:-1], r_scale[:-1]):  # exclude final logits
            loss = loss + w * F.l1_loss(f_map, r_map.detach())
    return loss / max(num_D, 1)


class ReprojectionConsistencyLoss(nn.Module):
    """
    >>> Stage-2 PLACEHOLDER (3rd.md) — disabled unless cfg.loss.w_reproj > 0. <<<

    Known-pose cross-view consistency: warp generated view i into view j's frame
    using the *known* relative camera pose + depth_i (all in meta.json: K,
    cam_to_world, depth_metric_ref), mask occlusions from depth, and penalise the
    photometric / LPIPS difference on co-visible pixels. Because poses are exact,
    no epipolar estimation is needed — a fixed spatial-transformer warp suffices.

    Add ONLY if the Stage-1 cross-view reprojection gate fails (it should pass by
    construction with fixed lighting). Requires the dataloader to yield view PAIRS
    from the same subject, which is not wired up yet — hence this stub.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ReprojectionConsistencyLoss is a Stage-2 placeholder. Wire up paired-view "
            "batching and the known-pose warp before enabling cfg.loss.w_reproj.")


def build_lpips(device):
    """Optional LPIPS metric/loss (lpips pkg). Returns None if unavailable."""
    try:
        import lpips
        net = lpips.LPIPS(net="alex").to(device)
        for p in net.parameters():
            p.requires_grad = False
        return net.eval()
    except Exception as e:
        print(f"[losses] LPIPS unavailable ({e}); set w_lpips=0 or install lpips.")
        return None
