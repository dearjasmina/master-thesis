"""
networks.py — pix2pixHD generator + multi-scale PatchGAN discriminator.

Faithful, trimmed re-implementation of NVIDIA pix2pixHD (Wang et al., 2018), the
3rd.md PRIMARY backbone: a coarse-to-fine encoder–ResNet–decoder generator and a
multi-scale 70×70 PatchGAN with intermediate features exposed for feature matching.
Input channel count is arbitrary (G-buffer stack), output is 3-channel RGB.
"""
from __future__ import annotations

import functools
import torch
import torch.nn as nn


def get_norm_layer(norm: str):
    if norm == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    raise ValueError(f"unknown norm '{norm}'")


def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in cls and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ── Generator ─────────────────────────────────────────────────────────────────
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True)):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm_layer(dim), activation,
                  nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm_layer(dim)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        act = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), norm_layer(ngf), act]
        for i in range(n_downsampling):
            m = 2 ** i
            model += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1),
                      norm_layer(ngf * m * 2), act]
        m = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * m, norm_layer, act)]
        for i in range(n_downsampling):
            m = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(ngf * m // 2), act]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LocalEnhancer(nn.Module):
    """Full-res enhancer wrapping a half-res GlobalGenerator (pix2pixHD, for 1024²)."""
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=4,
                 n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.n_local_enhancers = n_local_enhancers
        act = nn.ReLU(True)

        # Coarse global generator runs at 1/2 resolution; we reuse its feature trunk.
        ngf_global = ngf * (2 ** n_local_enhancers)
        global_net = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global,
                                     n_blocks_global, norm_layer).model
        # Drop the final (pad, conv, tanh) so we keep its last feature map.
        self.model = nn.Sequential(*list(global_net)[:-3])

        for n in range(1, n_local_enhancers + 1):
            ngf_l = ngf * (2 ** (n_local_enhancers - n))
            down = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_l, 7), norm_layer(ngf_l), act,
                    nn.Conv2d(ngf_l, ngf_l * 2, 3, stride=2, padding=1), norm_layer(ngf_l * 2), act]
            up = []
            for _ in range(n_blocks_local):
                up += [ResnetBlock(ngf_l * 2, norm_layer, act)]
            up += [nn.ConvTranspose2d(ngf_l * 2, ngf_l, 3, stride=2, padding=1, output_padding=1),
                   norm_layer(ngf_l), act]
            if n == n_local_enhancers:
                up += [nn.ReflectionPad2d(3), nn.Conv2d(ngf_l, output_nc, 7), nn.Tanh()]
            setattr(self, f"model{n}_1", nn.Sequential(*down))
            setattr(self, f"model{n}_2", nn.Sequential(*up))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        pyramid = [x]
        for _ in range(self.n_local_enhancers):
            pyramid.append(self.downsample(pyramid[-1]))
        out = self.model(pyramid[-1])
        for n in range(1, self.n_local_enhancers + 1):
            down = getattr(self, f"model{n}_1")
            up = getattr(self, f"model{n}_2")
            inp = pyramid[self.n_local_enhancers - n]
            out = up(down(inp) + out)
        return out


def define_G(cfg):
    norm_layer = get_norm_layer(cfg.model.norm)
    m = cfg.model
    if m.netG == "global":
        net = GlobalGenerator(cfg.input_nc(), m.output_nc, m.ngf,
                              m.n_downsample_global, m.n_blocks_global, norm_layer)
    elif m.netG == "local":
        net = LocalEnhancer(cfg.input_nc(), m.output_nc, m.ngf, m.n_downsample_global,
                            m.n_blocks_global, m.n_local_enhancers, m.n_blocks_local, norm_layer)
    else:
        raise ValueError(f"unknown netG '{m.netG}'")
    net.apply(weights_init)
    return net


# ── Discriminator ─────────────────────────────────────────────────────────────
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw, pad = 4, 2
        seq = [[nn.Conv2d(input_nc, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            seq += [[nn.Conv2d(nf_prev, nf, kw, 2, pad), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev, nf = nf, min(nf * 2, 512)
        seq += [[nn.Conv2d(nf_prev, nf, kw, 1, pad), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        seq += [[nn.Conv2d(nf, 1, kw, 1, pad)]]
        # Register each scale block so intermediate features can be returned.
        self.n_layers = len(seq)
        for i, block in enumerate(seq):
            setattr(self, f"model{i}", nn.Sequential(*block))

    def forward(self, x):
        feats = [x]
        for i in range(self.n_layers):
            feats.append(getattr(self, f"model{i}")(feats[-1]))
        return feats[1:]  # list of intermediate maps; last is the patch logits


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=2, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.num_D = num_D
        for i in range(num_D):
            setattr(self, f"disc{i}", NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        result = []
        for i in range(self.num_D):
            result.append(getattr(self, f"disc{i}")(x))
            if i != self.num_D - 1:
                x = self.downsample(x)
        return result  # list (per scale) of list (per layer) of feature maps


def define_D(cfg):
    norm_layer = get_norm_layer(cfg.model.norm)
    m = cfg.model
    # Discriminator sees the conditioning input concatenated with the image.
    in_nc = cfg.input_nc() + m.output_nc
    net = MultiscaleDiscriminator(in_nc, m.ndf, m.n_layers_D, m.num_D, norm_layer)
    net.apply(weights_init)
    return net
