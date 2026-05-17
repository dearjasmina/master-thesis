"""
Calibration Part 2 — tissue rendering with ACES, multi-pass composite, and masking.

Problem identified in Part 1 (calibrate_mitsuba_gt.py):
  ANY tissue medium turns the image grey because tissue scatter converts the
  directional key light into diffuse ambient fill, collapsing contrast (std: 33→11).

Approaches tested here (one change at a time):
  S11  ACES tonemapping (exposure=3×) on tissue render          — does ACES recover contrast?
  S12  ACES tonemapping (exposure=6×) on tissue render          — more aggressive
  S13  Multi-pass: bone-only + 0.2×tissue-only (Reinhard)       — composite background=0 + glow
  S14  Multi-pass: bone-only + 0.5×tissue-only (Reinhard)       — more tissue
  S15  Multi-pass: bone-only + 1.0×tissue-only (Reinhard)       — full tissue
  S16  S13 composite + ACES (exposure=2×)                       — best combo?
  S17  S14 composite + skull mask (force background to 0)       — aggressive BG kill
  S18  Tissue only rendered, skull mask applied, ACES            — mask approach alone

All steps start from the same base scene. Notes document WHY each setting was chosen.

Usage: python scripts/calibrate_tissue_rendering.py
"""

import json, time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ROOT    = Path(".")
DATA    = ROOT / "data/renders/gt/CQ500CT95"
CAL2    = DATA / "calibration2"
EXP_DIR = DATA / "experiments"
SKULL   = DATA / "skull_surface.obj"
SIMPLE  = ROOT / "data/renders/simple/CQ500CT95 CQ500CT95.nii/color.png"
CAL2.mkdir(parents=True, exist_ok=True)

with open(DATA / "CQ500CT95 CQ500CT95_meta.json") as f:
    meta = json.load(f)
cam  = meta["camera"]
aabb = meta["aabb_mm"]
hx   = (aabb[3]-aabb[0])/2; hy = (aabb[4]-aabb[1])/2; hz = (aabb[5]-aabb[2])/2
cx   = aabb[0]+hx; cy = aabb[1]+hy; cz = aabb[2]+hz
VOL_TF = (mi.ScalarTransform4f.translate([cx,cy,cz]) @
          mi.ScalarTransform4f.scale([hx,hy,hz]))
CAM_TF = mi.ScalarTransform4f.look_at(
    origin=cam["position"], target=cam["focal_point"], up=cam["up"])
SK = [117.0, 125.0, 72.0]
SPP = 64
RES = 512

SIMPLE_GRAY = np.array(Image.open(SIMPLE).convert("L"))
SIMPLE_RGB  = np.array(Image.open(SIMPLE))


# ── Helpers ───────────────────────────────────────────────────────────────────

def bone_bsdf():
    return {
        "type": "roughplastic", "distribution": "ggx", "alpha": 0.25,
        "int_ior": 1.49,
        "diffuse_reflectance": {"type": "rgb", "value": [0.93, 0.89, 0.82]},
    }

def sensor():
    return {
        "type": "perspective", "fov": cam["fov_deg"], "fov_axis": "y",
        "to_world": CAM_TF,
        "film": {"type": "hdrfilm", "width": RES, "height": RES,
                 "rfilter": {"type": "tent"}, "pixel_format": "rgb"},
        "sampler": {"type": "independent", "sample_count": SPP},
    }

def kv_light(r=[2.5,2.1,1.7]):
    """Warm key light upper-right-front (300,-80,260)."""
    return {
        "type": "rectangle",
        "to_world": (mi.ScalarTransform4f.look_at([300,-80,260], SK, [0,0,1]) @
                     mi.ScalarTransform4f.scale([70,70,1])),
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": r}},
    }

def fill_light(r=[0.30,0.42,0.75]):
    """Cool fill light. Repositioned to z=300 so angle=35° > camera half-FOV 15°."""
    return {
        "type": "rectangle",
        "to_world": (mi.ScalarTransform4f.look_at([-120,310,300], SK, [0,0,1]) @
                     mi.ScalarTransform4f.scale([100,100,1])),
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": r}},
    }

def tissue_vol(sigma_name="tissue_sigma.vol", albedo_name="tissue_albedo.vol", scale=0.3):
    """Soft-tissue heterogeneous medium filling the bounding cube."""
    return {
        "type": "cube", "to_world": VOL_TF,
        "interior": {
            "type": "heterogeneous", "scale": scale,
            "sigma_t": {"type": "gridvolume",
                        "filename": str(EXP_DIR/sigma_name), "to_world": VOL_TF},
            "albedo":  {"type": "gridvolume",
                        "filename": str(EXP_DIR/albedo_name), "to_world": VOL_TF},
            "phase":   {"type": "hg", "g": 0.70},
        },
    }

def aces_filmic(x: np.ndarray) -> np.ndarray:
    """
    ACES filmic tonemapping (Hill 2015 approximation).
    More aggressive than Reinhard: lifts darks less and compresses highlights more.
    Good for high-dynamic-range scenes with large key/shadow ratio.
    Formula: (x*(2.51x+0.03)) / (x*(2.43x+0.59)+0.14)
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)

def reinhard(x: np.ndarray, exposure=1.0) -> np.ndarray:
    x = x * exposure
    return np.clip(x / (1.0 + x), 0.0, 1.0)

def to_uint8(ldr: np.ndarray) -> np.ndarray:
    """Apply gamma 2.2 and convert to uint8."""
    return (np.clip(ldr, 0, 1) ** (1/2.2) * 255).astype(np.uint8)

def skull_mask(threshold=30, sigma=4.0) -> np.ndarray:
    """
    Soft skull mask from PyVista simple render.
    Hard threshold at pixel > threshold, then Gaussian blur for soft edges.
    Returns float32 [0,1] per pixel, broadcastable to RGB.
    """
    hard = (SIMPLE_GRAY > threshold).astype(np.float32)
    soft = gaussian_filter(hard, sigma=sigma)
    return np.clip(soft, 0, 1)[..., np.newaxis]   # shape (H,W,1)

def stats(srgb: np.ndarray) -> str:
    return f"min={srgb.min():3d} max={srgb.max():3d} std={srgb.std():5.1f}"

def save_and_report(srgb: np.ndarray, name: str, note: str):
    out = CAL2 / f"{name}.png"
    Image.fromarray(srgb).save(str(out))
    verdict = "OK" if srgb.max() > 140 and srgb.min() < 20 else "GREY"
    print(f"  [{verdict}]  {name:<45}  {stats(srgb)}")
    print(f"          note: {note}")
    return srgb, verdict


# ── Base renders (HDR, before tonemapping) ────────────────────────────────────

def render_hdr(scene_dict: dict) -> np.ndarray:
    s = mi.load_dict(scene_dict)
    return np.array(mi.render(s, spp=SPP))


print("Pre-rendering base HDR images (reused across steps to avoid JIT cache) …\n")

# Bone-only scene (working baseline from calibration part 1)
# key[2.5,2.1,1.7] fill[0.30,0.42,0.75] — original intensities, fill repositioned z=300
_bone_scene = {
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":   kv_light([2.5, 2.1, 1.7]),         # original baseline key
    "fill":  fill_light([0.30, 0.42, 0.75]),     # fill repositioned out of FOV
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}
t0 = time.time()
bone_hdr = render_hdr(_bone_scene)
print(f"  Bone-only HDR rendered in {time.time()-t0:.1f}s  "
      f"hdr_max={bone_hdr.max():.3f}  hdr_mean={bone_hdr.mean():.4f}")

# Tissue-only scene (same lights, tissue medium, NO bone mesh)
# Purpose: capture warm scatter glow from tissue interior only
_tissue_scene = {
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":    kv_light([2.5, 2.1, 1.7]),
    "fill":   fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_vol(scale=0.3),              # scale=0.3 for visible glow
}
t0 = time.time()
tissue_hdr = render_hdr(_tissue_scene)
print(f"  Tissue-only HDR rendered in {time.time()-t0:.1f}s  "
      f"hdr_max={tissue_hdr.max():.3f}  hdr_mean={tissue_hdr.mean():.4f}")

# Full scene (bone + tissue together — the grey wash we're trying to fix)
_full_scene = {
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":    kv_light([2.5, 2.1, 1.7]),
    "fill":   fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_vol(scale=0.3),
    "skull":  {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}
t0 = time.time()
full_hdr = render_hdr(_full_scene)
print(f"  Full (bone+tissue) HDR rendered in {time.time()-t0:.1f}s  "
      f"hdr_max={full_hdr.max():.3f}  hdr_mean={full_hdr.mean():.4f}")

SKULL_MASK = skull_mask(threshold=30, sigma=4.0)
print(f"\nSkull mask: coverage={SKULL_MASK.mean()*100:.1f}%  (threshold=30, sigma=4)\n")
print("─"*75)


# ── Experiments ───────────────────────────────────────────────────────────────

renders  = []
labels   = []
verdicts = []

def record(srgb, lbl, verdict):
    renders.append(srgb); labels.append(lbl); verdicts.append(verdict)

# Reference: bone-only Reinhard (the working S0 from calibration part 1)
ref_srgb = to_uint8(reinhard(bone_hdr, exposure=1.0))
srgb, v = save_and_report(ref_srgb, "S_REF_bone_reinhard",
    "Baseline bone-only Reinhard (reference). min should be 0, max ~150.")
record(srgb, "REF\nbone+Reinhard", v)

# Reference: full (bone+tissue) Reinhard (the grey problem we're solving)
full_reinhard = to_uint8(reinhard(full_hdr, exposure=1.0))
srgb, v = save_and_report(full_reinhard, "S_GREY_full_reinhard",
    "Full (bone+tissue) Reinhard — shows the grey problem. Target: fix this.")
record(srgb, "GREY\nfull+Reinhard", v)

print()

# ── S11: ACES exposure=3× on full render ─────────────────────────────────────
# ACES has a steeper curve than Reinhard: highlights compress more, shadows lift less.
# exposure=3× pre-multiplies HDR values to push skull surface into the ACES highlight zone.
# Expected: skull brighter than Reinhard, but background may still be lifted.
srgb, v = save_and_report(
    to_uint8(aces_filmic(full_hdr * 3.0)),
    "S11_aces_exp3_full",
    "ACES (exposure=3) on full bone+tissue HDR. Does ACES reduce grey vs Reinhard?"
)
record(srgb, "S11\nACES×3\nfull", v)

# ── S12: ACES exposure=6× on full render ─────────────────────────────────────
# More aggressive: pulls skull highlights toward saturation while shadows stay dark.
# Risk: skull will overexpose (blow out specular highlights).
srgb, v = save_and_report(
    to_uint8(aces_filmic(full_hdr * 6.0)),
    "S12_aces_exp6_full",
    "ACES (exposure=6) on full bone+tissue. Higher exposure = more contrast but risk of overexposure."
)
record(srgb, "S12\nACES×6\nfull", v)

print()

# ── S13: Multi-pass composite α=0.2 (Reinhard) ───────────────────────────────
# Physically-motivated: bone-only (direct illumination, dark background) +
# small fraction of tissue glow (warm interior scatter, no bone interaction).
# α=0.2: subtle tissue tint, mostly bone appearance.
# Expected: min=0 from bone pass, max similar to bone, slight warm tint in skull interior.
alpha = 0.2
composite = bone_hdr + alpha * tissue_hdr
srgb, v = save_and_report(
    to_uint8(reinhard(composite, exposure=1.0)),
    f"S13_multipass_a{alpha:.1f}_reinhard",
    f"Multi-pass: bone_hdr + {alpha}×tissue_hdr (Reinhard). Background from bone=0, tissue adds warm glow."
)
record(srgb, f"S13\nbone+{alpha}×tissue\nReinhard", v)

# ── S14: Multi-pass composite α=0.5 (Reinhard) ───────────────────────────────
alpha = 0.5
composite = bone_hdr + alpha * tissue_hdr
srgb, v = save_and_report(
    to_uint8(reinhard(composite, exposure=1.0)),
    f"S14_multipass_a{alpha:.1f}_reinhard",
    f"Multi-pass: bone_hdr + {alpha}×tissue_hdr. More tissue glow, slightly warmer."
)
record(srgb, f"S14\nbone+{alpha}×tissue\nReinhard", v)

# ── S15: Multi-pass composite α=1.0 (Reinhard) ───────────────────────────────
alpha = 1.0
composite = bone_hdr + alpha * tissue_hdr
srgb, v = save_and_report(
    to_uint8(reinhard(composite, exposure=1.0)),
    f"S15_multipass_a{alpha:.1f}_reinhard",
    f"Multi-pass: bone_hdr + {alpha}×tissue_hdr. Full tissue glow (equal weight to bone)."
)
record(srgb, f"S15\nbone+{alpha}×tissue\nReinhard", v)

print()

# ── S16: Best composite + ACES ────────────────────────────────────────────────
# S14 (α=0.5) composite → apply ACES with exposure=2 to boost contrast.
# ACES compresses highlights (prevents skull overexposure) while lifting contrast.
alpha = 0.5
composite = bone_hdr + alpha * tissue_hdr
srgb, v = save_and_report(
    to_uint8(aces_filmic(composite * 2.0)),
    "S16_multipass_a0p5_aces_exp2",
    "S14 composite (bone+0.5×tissue) + ACES exposure=2. Best combination?"
)
record(srgb, "S16\nbone+0.5×tissue\nACES×2", v)

# ── S17: Skull mask on tissue, composite ─────────────────────────────────────
# Apply skull mask to tissue HDR BEFORE compositing — forces tissue glow to skull only.
# Mask from PyVista simple render (white = skull, black = background).
# Gaussian blur (sigma=4) gives soft transition at skull boundary.
# Then composite with bone HDR: bone_hdr + α × (tissue_hdr × mask)
alpha = 0.5
tissue_masked_hdr = tissue_hdr * SKULL_MASK
composite = bone_hdr + alpha * tissue_masked_hdr
srgb, v = save_and_report(
    to_uint8(reinhard(composite, exposure=1.0)),
    "S17_multipass_masked_a0p5_reinhard",
    "Skull-masked tissue: tissue_hdr × soft_mask, then bone+0.5×masked_tissue. Forces glow to skull only."
)
record(srgb, "S17\nbone+masked×0.5\nReinhard", v)

# ── S18: Skull mask on tissue + ACES ─────────────────────────────────────────
composite = bone_hdr + alpha * tissue_masked_hdr
srgb, v = save_and_report(
    to_uint8(aces_filmic(composite * 2.0)),
    "S18_multipass_masked_a0p5_aces",
    "S17 (masked composite) + ACES exposure=2. Hopefully: bone structure + warm glow + dark BG."
)
record(srgb, "S18\nbone+masked×0.5\nACES×2", v)

# ── S19: Bright key + tissue masked composite ────────────────────────────────
# Use the brighter key [15,12.6,10.2] for the bone pass (like Ideal GT),
# but add the tissue glow from a separate tissue pass on top.
# This is the combination of the best bone lighting + tissue as additive overlay.
print("\nRendering bone-only HDR with BRIGHT key [15,12.6,10.2] for S19-S20 …")
_bone_bright_scene = {
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":   kv_light([15.0, 12.6, 10.2]),
    "fill":  fill_light([0.45, 0.63, 1.26]),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}
t0 = time.time()
bone_bright_hdr = render_hdr(_bone_bright_scene)
print(f"  Bone-bright HDR in {time.time()-t0:.1f}s  hdr_max={bone_bright_hdr.max():.3f}")

alpha = 0.3
composite19 = bone_bright_hdr + alpha * tissue_masked_hdr
srgb, v = save_and_report(
    to_uint8(reinhard(composite19, exposure=1.0)),
    "S19_bright_bone_plus_masked_tissue",
    "Bright key [15,12.6,10.2] bone + 0.3×skull-masked tissue (Reinhard). Best of both worlds?"
)
record(srgb, "S19\nbrightbone+0.3×tissue\nReinhard", v)

# ── S20: S19 + ACES ──────────────────────────────────────────────────────────
srgb, v = save_and_report(
    to_uint8(aces_filmic(composite19 * 1.5)),
    "S20_bright_bone_plus_masked_tissue_aces",
    "S19 composite + ACES exposure=1.5. Filmic contrast on bright-bone + tissue-glow."
)
record(srgb, "S20\nbrightbone+0.3×tissue\nACES×1.5", v)


# ── Comparison grid ───────────────────────────────────────────────────────────
print("\nBuilding grid …")
n  = len(renders)
TH = 220
LH = 62
pad = 5
canvas = Image.new("RGB", (n*(TH+pad)-pad, TH+LH), (10,10,10))
draw   = ImageDraw.Draw(canvas)

for i, (img_arr, lbl, v) in enumerate(zip(renders, labels, verdicts)):
    thumb = Image.fromarray(img_arr).resize((TH, TH))
    x = i*(TH+pad)
    canvas.paste(thumb, (x, LH))
    col = (140,220,140) if v == "OK" else (220,120,120)
    draw.text((x+2,  2), lbl,                         fill=(210,210,210))
    draw.text((x+2, 34), f"min={img_arr.min()} max={img_arr.max()}", fill=col)
    draw.text((x+2, 48), f"std={img_arr.std():.1f}",                  fill=(150,150,170))

out = ROOT / "results/calibration2_grid.png"
canvas.save(str(out))
print(f"Grid → {out}  ({out.stat().st_size//1024} KB)")
print("\nAll done. Renders in data/renders/gt/CQ500CT95/calibration2/")
