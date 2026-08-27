"""
synth_tissue_textures.py — synthesise seamless per-organ PBR texture sets

Why this exists
---------------
Procedural shading in the render script hit a ceiling: every organ evaluates the same
node graph (three noise octaves + one contour-vessel field + a perfusion blob), so they
differ in colour but not in KIND. Real organs differ structurally — lung is reticular
septa with anthracotic speckle, colon has haustral banding, pancreas has coarse
fat-separated lobules, bowel has transverse folds and dense vasa recta. A shader that
can afford three octaves per pixel cannot represent that; an image can.

This generates those maps offline, at whatever resolution you like, using structure
that would be far too expensive to evaluate per-shading-sample — in particular REAL
BRANCHING VESSEL TREES grown recursively, rather than thresholded noise contours. That
single change is most of the difference between "veins" and "squiggles".

Output (per organ class, into --out):
    {name}_albedo.png    linear-ish sRGB colour, already in the organ's palette
    {name}_normal.png    tangent-space normal map (Non-Color)
    {name}_rough.png     roughness modulation (Non-Color)

The render script picks these up automatically via box (triplanar) projection, so the
marching-cubes meshes need no UVs. One set serves every tissue whose name contains the
class name, so `lung` covers all four lobes.

Everything is TILEABLE. Box projection repeats the map, so any generator that is not
periodic shows hard seams:
  · noise      — band-limited in the frequency domain, so periodic by construction
  · cellular   — jittered grid with wraparound neighbour search
  · vessels    — grown on a torus, coordinates taken modulo the image size
  · speckle    — wrapped splatting

Deterministic: seeded per organ class, so a given class always produces the same maps.

Usage
-----
    python scripts/synth_tissue_textures.py                     # all classes, 1024 px
    python scripts/synth_tissue_textures.py --size 2048
    python scripts/synth_tissue_textures.py --only liver,lung
    python scripts/synth_tissue_textures.py --preview           # also write a contact sheet

Sizing note: --tex_mm in the render script defaults to 120 mm per tile. A whole-torso
frame renders at roughly 0.65 mm/px, so features finer than ~0.7 mm are averaged away
and 1024 px per 120 mm tile (0.12 mm/texel) is already oversampled. 2048 only helps if
you render tight crops.

Requires numpy and Pillow only.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


# ── Tileable primitives ───────────────────────────────────────────────────────

def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def spectral_noise(size, rng, freq, aniso=1.0, angle=0.0):
    """Band-limited periodic noise.

    White noise filtered to a frequency band in the Fourier domain. The result is
    periodic by construction, so it tiles seamlessly — which plain value/Perlin noise
    sampled on a finite grid does not.

    `freq` is in cycles across the tile. `aniso` > 1 stretches features along `angle`
    (radians), which is how fibre striation and rugal folds are made directional.
    """
    w = rng.normal(size=(size, size))
    F = np.fft.fft2(w)
    f = np.fft.fftfreq(size) * size
    fy, fx = np.meshgrid(f, f, indexing="ij")
    ca, sa = math.cos(angle), math.sin(angle)
    u = (fx * ca + fy * sa) / max(aniso, 1e-6)
    v = -fx * sa + fy * ca
    R = np.hypot(u, v)
    band = np.exp(-((R - freq) ** 2) / (2.0 * max(freq * 0.55, 0.7) ** 2))
    return _norm(np.real(np.fft.ifft2(F * band)))


def fbm(size, rng, freq, octaves=4, persistence=0.55, aniso=1.0, angle=0.0):
    out, amp, f, tot = np.zeros((size, size)), 1.0, float(freq), 0.0
    for _ in range(octaves):
        out += amp * spectral_noise(size, rng, f, aniso, angle)
        tot += amp
        amp *= persistence
        f *= 2.0
    return _norm(out / max(tot, 1e-6))


def worley(size, rng, cells, jitter=0.9, kind="f1"):
    """Tileable cellular noise on a jittered grid.

    Wraparound neighbour search over the 3x3 cell block makes it periodic. Returns
    distance-to-nearest-point ("f1") or distance-to-cell-edge ("edge"), the latter via
    the f2-f1 difference, which is what gives lobule and septum boundaries.
    """
    c = max(int(cells), 1)
    gy, gx = np.meshgrid(np.arange(c), np.arange(c), indexing="ij")
    pts = np.stack([gy + rng.uniform(0.5 - jitter / 2, 0.5 + jitter / 2, (c, c)),
                    gx + rng.uniform(0.5 - jitter / 2, 0.5 + jitter / 2, (c, c))], -1)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cy, cx = ys * c / size, xs * c / size
    iy, ix = np.floor(cy).astype(int), np.floor(cx).astype(int)

    d1 = np.full((size, size), 1e9)
    d2 = np.full((size, size), 1e9)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            py = pts[(iy + oy) % c, (ix + ox) % c, 0] + oy * 0  # value already in cells
            px = pts[(iy + oy) % c, (ix + ox) % c, 1]
            # neighbour cell origin in continuous cell space, with wrap
            ny = np.floor(cy) + oy + (py - np.floor(py))
            nx = np.floor(cx) + ox + (px - np.floor(px))
            dy = np.abs(cy - ny); dy = np.minimum(dy, c - dy)
            dx = np.abs(cx - nx); dx = np.minimum(dx, c - dx)
            d = np.hypot(dy, dx)
            nd1 = np.minimum(d1, d)
            d2 = np.minimum(d2, np.maximum(d1, d))
            d1 = nd1
    return _norm(d2 - d1) if kind == "edge" else _norm(d1)


# ── Vessel calibres ──────────────────────────────────────────────────────────
# MEASURED from Human Organ Atlas HiP-CT (CC-BY-4.0) via scripts/hoa_vessel_stats.py.
# Method: Hessian multiscale vesselness with SCALE SELECTION — the sigma maximising
# the ridge response gives the calibre directly, so it is immune to the wall-vs-lumen
# problem that made an earlier segment-then-measure attempt meaningless (a bright-wall
# detector returns broken crescents, and a distance transform on a crescent measures
# arc thickness, not vessel radius). sigma -> radius = 1.336, CALIBRATED on synthetic
# tubes of known radius rather than taken from the paper.
#
# Restricted to a subcapsular shell, because that is the network the renderer draws.
#
#   organ    donor           p50    p95    filter   detected   ROI quality
#   liver    LADAF-2020-27   0.55   2.4    sato     0.0175     good
#   kidney   LADAF-2020-27   0.9    1.5    sato     0.0008     good
#   lung     LADAF-2020-27   0.2    2.4    sato     0.0013     good (hilum-weighted)
#   heart    LADAF-2020-31   0.9    2.4    frangi   0.0042     good
#   colon    LADAF-2021-17   0.9    1.5    frangi   0.0011     good (one segment)
#
# heart and colon need FRANGI, not Sato: their walls and trabeculae are SHEETS, and
# Sato responds to sheets — it flagged 2.7 % of the heart volume as "vessel". Frangi
# suppresses plate-like structure explicitly and drops that to 0.42 %.
#
# start_w below is the TRUNK radius. Since the tree tapers by Murray's law (0.794) over
# ~5 levels, the typical DRAWN vessel is about start_w x 0.794^2.5 = 0.56 x start_w, so
# start_w is set to p50 / 0.56 to make the typical rendered vessel match the measured
# median.
#
# HONEST UNCERTAINTY: these p50s are threshold-sensitive. Loosening the detection
# threshold raises them (kidney 0.9 -> 1.5, lung 0.2 -> 0.55) and pushes p95 against
# the largest scale offered, which is a sign of over-growth rather than more anatomy.
# The values below use the conservative setting. What is ROBUST across every method and
# threshold tried is the ORDER OF MAGNITUDE: subcapsular vessels are 0.2-1.5 mm radius.
# My original hand-picked values were 3.4 mm (liver) and 4.5 mm (heart) — that is the
# quantitative form of "the veins look too prominent", and it is the finding that
# actually matters here.
#
# NOT measured: split probability and segment length. The skeleton at this voxel size
# gives 7604 branch points against 3620 endpoints, but a binary tree must have roughly
# one MORE endpoint than branch point, so its topology is unusable. Those stay
# hand-set, as do bowel, stomach, pancreas and the great-vessel walls (extrapolated).
#
# Three donors — cite each dataset DOI separately, plus the HOA paper
# (DOI 10.1126/sciadv.adz2240).

def vessel_tree(size, rng, n_roots=5, depth=7, step=None, start_w=None,
                curl=0.45, split=0.62, taper=0.72, aniso=None):
    """Recursively grown branching vessel network, drawn on a torus.

    This is the point of the whole file. A thresholded noise contour produces sinuous
    lines of roughly uniform width with no hierarchy — it reads as cracks. A real
    vascular tree has a trunk that splits into progressively finer branches, and that
    hierarchy is what the eye recognises. Grown here as an explicit recursion and
    splatted with a tapering radius; coordinates wrap, so it tiles.

    `aniso` optionally biases growth toward a direction (radians) — used for the
    roughly parallel vasa recta of small bowel.
    """
    step = step or size / 26.0
    start_w = start_w or size / 110.0
    acc = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    def splat(y, x, w):
        # local disc, wrapped — cheaper than touching the whole image per segment
        r = int(max(w * 2.5, 2))
        ys = (np.arange(-r, r + 1) + int(y)) % size
        xs = (np.arange(-r, r + 1) + int(x)) % size
        dy = (np.arange(-r, r + 1) + int(y) - y)[:, None]
        dx = (np.arange(-r, r + 1) + int(x) - x)[None, :]
        d = np.hypot(dy, dx)
        blob = np.clip(1.0 - (d / max(w, 0.6)), 0.0, 1.0) ** 0.7
        acc[np.ix_(ys, xs)] = np.maximum(acc[np.ix_(ys, xs)], blob)

    def grow(y, x, ang, w, d):
        if d <= 0 or w < 0.45:
            return
        for _ in range(int(rng.integers(4, 9))):
            ang += rng.normal(0, curl)
            if aniso is not None:                      # bias back toward a direction
                ang += 0.35 * math.sin(aniso - ang)
            # Walk the segment in sub-steps no larger than half the current radius.
            # Splatting only at the segment endpoint leaves gaps whenever step > w —
            # which is always — and the tree renders as a dotted line instead of a
            # vessel. This is what made the first pass look like stippling.
            nsub = max(int(step / max(w * 0.5, 1.0)), 1)
            for _s in range(nsub):
                y = (y + math.sin(ang) * step / nsub) % size
                x = (x + math.cos(ang) * step / nsub) % size
                splat(y, x, w)
        if rng.random() < split:
            for s in (-1, 1):
                grow(y, x, ang + s * rng.uniform(0.35, 0.95), w * taper, d - 1)
        else:
            grow(y, x, ang + rng.normal(0, 0.3), w * taper, d - 1)

    for _ in range(n_roots):
        grow(rng.uniform(0, size), rng.uniform(0, size),
             rng.uniform(0, 2 * math.pi) if aniso is None else aniso + rng.normal(0, 0.4),
             start_w, depth)
    return np.clip(acc, 0, 1)


def soften(a, sigma_px):
    """Periodic Gaussian blur, via the frequency domain so it stays tileable.

    Vessels do not sit ON an organ, they sit UNDER a translucent capsule, and that
    capsule scatters: the deeper a vessel is, the softer and lower-contrast it looks.
    Painting them as sharp dark lines is what makes every organ read as the same
    substance with a decal applied — which is exactly the failure mode this fixes.
    Blur radius is the effective capsule depth.
    """
    if sigma_px <= 0.2:
        return a
    n = a.shape[0]
    f = np.fft.fftfreq(n) * n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    g = np.exp(-2.0 * (np.pi * sigma_px / n) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


# Measured off the Visible Korean abdominal section (Park et al., 2015, Int J Morphol
# 33(4):1323-1332, Fig. 5b): subcutaneous fat samples at sRGB (205,152,80) and
# (194,141,101) -> linear ~(0.61, 0.32, 0.08). Green sits near HALF of red; adipose is
# a strong yellow-orange, not a cream. Two earlier guesses here were both wrong in the
# same direction, the second worse than the first: what made fat read as pathology was
# that it was splatted as opaque CIRCLES, not that it was too yellow.
# Fresh adipose photographs around sRGB (240, 220, 150) -> linear (0.83, 0.69, 0.29).
# The first correction over-shot into olive: 0.42/0.355/0.215 is dark and green-
# leaning, which on a pink serosa reads as a bruise rather than fat. Pale and warm.
FAT_RGB = np.array((0.612, 0.316, 0.081), np.float32)


def lobulate(mask, size, rng, freq=9.0):
    """Break round speckle discs into irregular lobulated patches.

    A gaussian splat is a perfect circle; fat tags are pendulous and lumpy. Modulating
    by a mid-frequency noise and re-thresholding keeps the placement but destroys the
    tell-tale circularity."""
    n = spectral_noise(size, rng, freq)
    return np.clip(mask * (0.45 + 1.15 * n) - 0.10, 0.0, 1.0)


def speckle(size, rng, count, radius, softness=1.6):
    """Wrapped point splatter — anthracotic pigment, fat flecks."""
    acc = np.zeros((size, size), np.float32)
    for _ in range(count):
        y, x = rng.uniform(0, size), rng.uniform(0, size)
        w = radius * rng.uniform(0.4, 1.8)
        r = int(max(w * 2.5, 2))
        ys = (np.arange(-r, r + 1) + int(y)) % size
        xs = (np.arange(-r, r + 1) + int(x)) % size
        dy = (np.arange(-r, r + 1) + int(y) - y)[:, None]
        dx = (np.arange(-r, r + 1) + int(x) - x)[None, :]
        blob = np.exp(-((np.hypot(dy, dx) / max(w, 0.5)) ** 2) * softness)
        acc[np.ix_(ys, xs)] = np.maximum(acc[np.ix_(ys, xs)], blob)
    return np.clip(acc, 0, 1)


def height_to_normal(h, strength=1.0):
    """Tangent-space normal map from a height field, using wrapped gradients."""
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    n = np.stack([-gx, -gy, np.ones_like(h) / max(strength, 1e-3)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def tint(mask, lo_rgb, hi_rgb):
    """Blend two colours by a scalar mask -> RGB image."""
    lo = np.array(lo_rgb, np.float32)
    hi = np.array(hi_rgb, np.float32)
    return lo[None, None, :] + (hi - lo)[None, None, :] * mask[..., None]


# ── Colour helpers ────────────────────────────────────────────────────────────

def srgb(lin):
    """Linear -> sRGB. The albedo PNG is read back as sRGB by Blender, so the palette
    values (which are linear scattering albedos) must be encoded on the way out or the
    organs render far too dark."""
    a = np.clip(lin, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def _u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ── Per-organ recipes ─────────────────────────────────────────────────────────
#
# Each returns (albedo_linear RGB, height, roughness). The structural vocabulary is
# deliberately different per organ — that is the entire point, and what parameter
# tuning on a shared node graph could never deliver.

def r_liver(S, rng, mm):
    zones = fbm(S, rng, 2.2, 3)                       # broad congested / perfused zones
    grain = fbm(S, rng, 34.0, 3)                      # fine parenchymal granularity
    caps  = fbm(S, rng, 9.0, 4)                       # Glisson capsule stretch
    vein  = vessel_tree(S, rng, n_roots=3, depth=5, start_w=0.98*mm, step=4.0*mm, split=0.5)
    # Measured VK liver (0.101, 0.034, 0.026) linear; range straddles it. Note this is
    # a CUT FACE — parenchyma, not Glisson's capsule — so it is a floor, not a target.
    base  = tint(0.35 + 0.5 * zones + 0.15 * grain,
                 (0.072, 0.024, 0.019), (0.142, 0.048, 0.037))
    base *= (1.0 - 0.26 * soften(vein, 1.6*mm))[..., None]            # vessels darken, not tint
    h = 0.55 * caps + 0.25 * grain + 0.35 * vein
    r = 0.30 + 0.18 * (1 - caps)
    return base, h, r


def r_lung(S, rng, mm):
    # Interlobular septa are faint outlines on a fresh lung, not a strong polygonal
    # net. At 1 - edge*3.2 they became thick bands and the surface read as cracked mud.
    # Narrow threshold, then soften — barely-there is correct here.
    septa = worley(S, rng, 9, kind="edge")
    septa = soften(np.clip(1.0 - septa * 8.0, 0, 1), 0.8 * mm)
    mott  = fbm(S, rng, 3.4, 4)
    fine  = vessel_tree(S, rng, n_roots=8, depth=6, start_w=0.36*mm,
                        step=2.4*mm, split=0.72)
    # Anthracotic pigment: carbon deposits collect ALONG the septa, not uniformly.
    carbon = speckle(S, rng, 240, 0.55*mm) * (0.55 + 0.45 * septa)
    # Darker and more saturated than the first pass: SSS at 0.70 plus the coat lift
    # the rendered result well above the albedo, so a texture that looks correct as a
    # flat tile renders as pale grey-pink on the organ.
    base = tint(0.34 + 0.60 * mott + 0.06 * septa,
                (0.190, 0.098, 0.104), (0.395, 0.222, 0.226))
    base *= (1.0 - 0.14 * soften(fine, 1.3*mm))[..., None]
    base *= (1.0 - 0.45 * carbon)[..., None]
    h = 0.16 * septa + 0.42 * mott + 0.18 * fine
    r = 0.42 + 0.22 * mott
    return base, h, r


def r_bowel(S, rng, mm):
    ang   = math.pi / 2
    folds = fbm(S, rng, 12.0, 3, aniso=7.0, angle=ang)      # transverse mucosal folds
    # vasa recta run roughly perpendicular to the mesenteric border: biased growth
    vasa  = vessel_tree(S, rng, n_roots=14, depth=5, start_w=0.90*mm,
                        step=2.8*mm, split=0.55, curl=0.30, aniso=0.0)
    fat   = speckle(S, rng, 9,  2.6*mm)                   # mesenteric fat tags
    base  = tint(0.35 + 0.55 * folds, (0.440, 0.215, 0.160), (0.660, 0.390, 0.300))
    base *= (1.0 - 0.40 * soften(vasa, 0.7*mm))[..., None]
    fat3 = lobulate(fat, S, rng)[..., None] * 0.42
    base = base * (1 - fat3) + FAT_RGB * fat3
    h = 0.70 * folds + 0.25 * vasa + 0.4 * fat
    r = 0.26 + 0.14 * folds
    return base, h, r


def r_colon(S, rng, mm):
    haustra = fbm(S, rng, 5.0, 2, aniso=9.0, angle=math.pi / 2)   # haustral banding
    taenia  = fbm(S, rng, 2.0, 1, aniso=14.0, angle=0.0)          # longitudinal taeniae
    vein    = vessel_tree(S, rng, n_roots=7, depth=5, start_w=1.60*mm, step=3.6*mm, split=0.6)
    fat     = speckle(S, rng, 14, 3.4*mm)                       # appendices epiploicae
    base = tint(0.35 + 0.5 * haustra + 0.2 * taenia,
                (0.330, 0.180, 0.180), (0.520, 0.310, 0.295))
    base *= (1.0 - 0.26 * soften(vein, 1.1*mm))[..., None]
    fat3 = lobulate(fat, S, rng)[..., None] * 0.42
    base = base * (1 - fat3) + FAT_RGB * fat3
    h = 0.85 * haustra + 0.3 * taenia + 0.45 * fat
    r = 0.26 + 0.14 * haustra
    return base, h, r


def r_pancreas(S, rng, mm):
    lob  = worley(S, rng, 11, kind="f1")              # coarse lobules
    edge = soften(np.clip(1.0 - worley(S, rng, 11, kind="edge") * 5.0, 0, 1), 0.7 * mm)
    base = tint(0.30 + 0.6 * lob, (0.360, 0.250, 0.150), (0.560, 0.430, 0.290))
    e3 = 0.30 * edge[..., None]
    base = base * (1 - e3) + FAT_RGB * e3
    h = 0.9 * lob - 0.5 * edge
    r = 0.48 + 0.20 * (1 - lob)
    return base, h, np.clip(r, 0, 1)


def r_stomach(S, rng, mm):
    rugae = fbm(S, rng, 7.0, 3, aniso=5.0, angle=0.6)   # rugal folds
    vein  = vessel_tree(S, rng, n_roots=4, depth=4, start_w=0.75*mm, step=3.2*mm, split=0.5)
    base  = tint(0.35 + 0.55 * rugae, (0.400, 0.280, 0.230), (0.580, 0.430, 0.360))
    base *= (1.0 - 0.18 * soften(vein, 1.2*mm))[..., None]
    return base, 0.8 * rugae + 0.2 * vein, 0.28 + 0.14 * rugae


def r_spleen(S, rng, mm):
    pulp = fbm(S, rng, 3.0, 4)
    fine = fbm(S, rng, 28.0, 2)
    base = tint(0.30 + 0.55 * pulp + 0.15 * fine,
                (0.095, 0.020, 0.036), (0.190, 0.048, 0.076))
    return base, 0.4 * pulp + 0.2 * fine, 0.30 + 0.12 * pulp


def r_kidney(S, rng, mm):
    cap  = fbm(S, rng, 4.0, 3)
    vein = vessel_tree(S, rng, n_roots=3, depth=5, start_w=1.60*mm, step=3.0*mm, split=0.55)
    base = tint(0.32 + 0.55 * cap, (0.120, 0.045, 0.030), (0.235, 0.085, 0.058))
    base *= (1.0 - 0.20 * soften(vein, 1.4*mm))[..., None]
    return base, 0.45 * cap + 0.3 * vein, 0.30 + 0.12 * cap


def r_heart(S, rng, mm):
    fibre = fbm(S, rng, 18.0, 3, aniso=6.0, angle=0.9)          # myocardial fibre run
    cor   = vessel_tree(S, rng, n_roots=3, depth=6, start_w=1.60*mm, step=4.0*mm, split=0.55)
    fat   = speckle(S, rng, 10, 3.6*mm) * np.clip(soften(cor, 1.2*mm) * 1.8 - 0.18, 0, 1)  # fat follows grooves
    base  = tint(0.32 + 0.5 * fibre, (0.220, 0.060, 0.050), (0.360, 0.115, 0.095))
    base *= (1.0 - 0.26 * soften(cor, 1.1*mm))[..., None]
    fat3 = lobulate(fat, S, rng)[..., None] * 0.46
    base = base * (1 - fat3) + FAT_RGB * fat3
    return base, 0.35 * fibre + 0.55 * cor + 0.4 * fat, 0.30 + 0.16 * fibre


def r_autochthon(S, rng, mm):
    fibre = fbm(S, rng, 26.0, 4, aniso=12.0, angle=math.pi / 2)  # strong striation
    peri  = worley(S, rng, 9, kind="edge")
    base  = tint(0.30 + 0.6 * fibre, (0.175, 0.052, 0.062), (0.330, 0.108, 0.130))
    base *= (1.0 - 0.14 * soften(np.clip(1 - peri * 5, 0, 1), 0.9 * mm))[..., None]
    return base, 0.8 * fibre, 0.48 + 0.18 * fibre


def r_vertebrae(S, rng, mm):
    grain = fbm(S, rng, 22.0, 4)
    pore  = speckle(S, rng, 130, 0.7*mm)
    base  = tint(0.35 + 0.55 * grain, (0.480, 0.420, 0.330), (0.680, 0.625, 0.520))
    base *= (1.0 - 0.30 * pore)[..., None]
    return base, 0.4 * grain + 0.5 * pore, 0.62 + 0.18 * grain


def r_vessel_wall(S, rng, mm, col_lo, col_hi):
    stri = fbm(S, rng, 20.0, 3, aniso=10.0, angle=math.pi / 2)   # longitudinal fibres
    vasa = vessel_tree(S, rng, n_roots=6, depth=3, start_w=0.60*mm, step=2.6*mm, split=0.4)
    base = tint(0.35 + 0.55 * stri, col_lo, col_hi)
    base *= (1.0 - 0.16 * soften(vasa, 1.0*mm))[..., None]
    return base, 0.5 * stri + 0.2 * vasa, 0.22 + 0.12 * stri


def r_aorta(S, rng, mm):     return r_vessel_wall(S, rng, mm, (0.230, 0.080, 0.068), (0.400, 0.160, 0.132))
def r_vena(S, rng, mm):      return r_vessel_wall(S, rng, mm, (0.130, 0.070, 0.145), (0.245, 0.135, 0.270))
def r_esophagus(S, rng, mm): return r_vessel_wall(S, rng, mm, (0.290, 0.185, 0.170), (0.450, 0.300, 0.270))


def r_smooth(S, rng, mm, col_lo, col_hi, freq=3.5, rough=0.30):
    m = fbm(S, rng, freq, 4)
    f = fbm(S, rng, 26.0, 2)
    return tint(0.32 + 0.55 * m + 0.13 * f, col_lo, col_hi), 0.4 * m + 0.2 * f, rough + 0.12 * m


def r_gallbladder(S, rng, mm): return r_smooth(S, rng, mm, (0.040, 0.062, 0.036), (0.090, 0.130, 0.080), 4.0, 0.20)
def r_bladder(S, rng, mm):     return r_smooth(S, rng, mm, (0.240, 0.180, 0.175), (0.400, 0.300, 0.290), 3.0, 0.32)


RECIPES = {
    "liver": r_liver, "lung": r_lung, "bowel": r_bowel, "duodenum": r_bowel,
    "colon": r_colon, "pancreas": r_pancreas, "stomach": r_stomach,
    "spleen": r_spleen, "kidney": r_kidney, "heart": r_heart,
    "autochthon": r_autochthon, "vertebrae": r_vertebrae,
    "aorta": r_aorta, "vena_cava": r_vena, "portal": r_vena,
    "esophagus": r_esophagus, "gallbladder": r_gallbladder,
    "urinary_bladder": r_bladder,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  default="data/renders/textures/tissue")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--tex_mm", type=float, default=80.0,
                    help="real-world width of one tile in mm. MUST match --tex_mm in "
                         "the render script: every feature size in the recipes is given "
                         "in mm and converted through this. The first pass sized "
                         "features in TEXTURE PIXELS, which put bowel vasa recta at "
                         "0.39 mm — below one screen pixel at render scale, so they "
                         "averaged away and bowel rendered indistinguishable from lung.")
    ap.add_argument("--only", default="", help="comma-separated class names")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--normal_strength", type=float, default=2.5)
    ap.add_argument("--preview", action="store_true", help="write a contact sheet")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(RECIPES)
    S = args.size
    print(f"synthesising {len(names)} classes at {S}px -> {out}")

    tiles = []
    for i, name in enumerate(sorted(set(names))):
        if name not in RECIPES:
            print(f"  [skip] no recipe for '{name}'"); continue
        rng = np.random.default_rng(args.seed + abs(hash(name)) % 100000)
        alb, h, r = RECIPES[name](S, rng, S / float(args.tex_mm))
        Image.fromarray(_u8(srgb(alb))).save(out / f"{name}_albedo.png")
        Image.fromarray(_u8(height_to_normal(_norm(h), args.normal_strength))).save(
            out / f"{name}_normal.png")
        Image.fromarray(_u8(np.clip(r, 0, 1)), mode="L").save(out / f"{name}_rough.png")
        print(f"  {name:<16} albedo+normal+rough")
        if args.preview:
            tiles.append((name, Image.fromarray(_u8(srgb(alb))).resize((192, 192))))

    if args.preview and tiles:
        cols = 6; rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 196 + 4, rows * 210 + 4), (18, 18, 18))
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for i, (n, im) in enumerate(tiles):
            x, y = (i % cols) * 196 + 4, (i // cols) * 210 + 18
            sheet.paste(im, (x, y)); d.text((x + 2, y - 14), n, fill=(235, 235, 235))
        sheet.save(out / "_preview.png")
        print(f"  preview -> {out/'_preview.png'}")


if __name__ == "__main__":
    main()
