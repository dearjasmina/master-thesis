"""
Systematic one-at-a-time calibration of Mitsuba GT rendering settings.
Starts from the working E0 baseline (pure surface + area lights, min=0 max=154)
and changes ONE setting per step to identify which settings cause grey washout.

Output:
  data/renders/gt/CQ500CT95/calibration/  — per-step renders
  results/calibration_grid.png            — annotated comparison grid

Usage: python scripts/calibrate_mitsuba_gt.py
"""

import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import numpy as np
from PIL import Image, ImageDraw

ROOT     = Path(".")
DATA     = ROOT / "data/renders/gt/CQ500CT95"
CAL_DIR  = DATA / "calibration"
EXP_DIR  = DATA / "experiments"
SKULL    = DATA / "skull_surface.obj"
CAL_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA / "CQ500CT95 CQ500CT95_meta.json") as f:
    meta = json.load(f)
cam  = meta["camera"]
aabb = meta["aabb_mm"]
hx = (aabb[3]-aabb[0])/2; hy = (aabb[4]-aabb[1])/2; hz = (aabb[5]-aabb[2])/2
cx = aabb[0]+hx; cy = aabb[1]+hy; cz = aabb[2]+hz
VOL_TF = (mi.ScalarTransform4f.translate([cx,cy,cz]) @
          mi.ScalarTransform4f.scale([hx,hy,hz]))
CAM_TF = mi.ScalarTransform4f.look_at(
    origin=cam["position"], target=cam["focal_point"], up=cam["up"])

SK  = [117.0, 125.0, 72.0]
SPP = 64
RES = 512


# ── Shared pieces ─────────────────────────────────────────────────────────────

def bone_bsdf():
    return {
        "type": "roughplastic",
        "distribution": "ggx",
        "alpha": 0.25,
        "int_ior": 1.49,
        "diffuse_reflectance": {"type": "rgb", "value": [0.93, 0.89, 0.82]},
    }


def sensor():
    return {
        "type": "perspective",
        "fov": cam["fov_deg"],
        "fov_axis": "y",
        "to_world": CAM_TF,
        "film": {"type": "hdrfilm", "width": RES, "height": RES,
                 "rfilter": {"type": "tent"}, "pixel_format": "rgb"},
        "sampler": {"type": "independent", "sample_count": SPP},
    }


def kv_light(radiance):
    """Key light upper-right-front."""
    return {
        "type": "rectangle",
        "to_world": (mi.ScalarTransform4f.look_at([300,-80,260], SK, [0,0,1]) @
                     mi.ScalarTransform4f.scale([70,70,1])),
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}},
    }


def fill_light(radiance):
    """Fill light lower-left-back."""
    return {
        "type": "rectangle",
        "to_world": (mi.ScalarTransform4f.look_at([-120,310,40], SK, [0,0,1]) @
                     mi.ScalarTransform4f.scale([90,90,1])),
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}},
    }


def tissue_medium(sigma_vol: str, albedo_vol: str, scale: float):
    return {
        "type": "cube",
        "to_world": VOL_TF,
        "interior": {
            "type": "heterogeneous",
            "scale": scale,
            "sigma_t": {"type": "gridvolume",
                        "filename": str(EXP_DIR / sigma_vol), "to_world": VOL_TF},
            "albedo":  {"type": "gridvolume",
                        "filename": str(EXP_DIR / albedo_vol), "to_world": VOL_TF},
            "phase":   {"type": "hg", "g": 0.70},
        },
    }


# ── Render + stats ────────────────────────────────────────────────────────────

def render(scene_dict, out_name, note, exposure=1.0):
    scene = mi.load_dict(scene_dict)
    t0    = time.time()
    img   = mi.render(scene, spp=SPP)
    dt    = time.time() - t0
    hdr   = np.array(img) * exposure
    srgb  = (np.clip(hdr/(1+hdr), 0, 1)**(1/2.2)*255).astype(np.uint8)
    out   = CAL_DIR / f"{out_name}.png"
    Image.fromarray(srgb).save(str(out))
    verdict = "OK" if srgb.max() > 140 and srgb.min() < 30 else "GREY/DARK"
    print(f"  [{verdict}]  {out_name:40s}  min={srgb.min():3d} max={srgb.max():3d} "
          f"std={srgb.std():.1f}  {dt:.1f}s")
    print(f"          note: {note}")
    return srgb, verdict


# ── Calibration steps ─────────────────────────────────────────────────────────

steps = []   # (scene_dict, name, note, exposure)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — Baseline (E0 style)
# Pure bone surface + original area lights. This is the only known-working config.
# key=[2.5,2.1,1.7]  fill=[0.30,0.42,0.75]  NO tissue medium
# Expected: min=0, max≈154, std≈28  → dark background, bright skull, good contrast
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key": kv_light([2.5, 2.1, 1.7]),   # warm white key
    "fill": fill_light([0.30, 0.42, 0.75]),  # cool blue fill
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S0_baseline", "Baseline: bone surface + area lights, no tissue", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Add tissue medium, scale=0.3 (original bad setting)
# Adds soft-tissue heterogeneous medium inside the bounding cube.
# scale=0.3 → MFP≈56mm → tissue absorbs ~70% of light from key before it reaches bone
# Expected problem: skull dims from max≈154 → ≈110, background non-zero (scattered glow)
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key": kv_light([2.5, 2.1, 1.7]),
    "fill": fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.3),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S1_tissue_scale0p3",
   "Added tissue medium, scale=0.3 (MFP≈56mm). Key light highly attenuated before reaching bone.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Reduce tissue to scale=0.1
# MFP=125mm, key attenuated ≈40% over 50mm tissue path (exp(-0.1*0.08*50)=0.67)
# Better than scale=0.3, skull should be brighter
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key": kv_light([2.5, 2.1, 1.7]),
    "fill": fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.1),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S2_tissue_scale0p1",
   "Tissue scale=0.1 (MFP≈125mm). Less attenuation: ~33% of key attenuated over 50mm.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Reduce tissue to scale=0.05
# MFP=250mm (>> skull diameter ≈230mm), key attenuated only 18% over 50mm tissue
# Tissue nearly transparent — very subtle scatter, bone should approach E0 brightness
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key": kv_light([2.5, 2.1, 1.7]),
    "fill": fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.05),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S3_tissue_scale0p05",
   "Tissue scale=0.05 (MFP≈250mm). Nearly transparent. ~18% key attenuation over 50mm.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Reduce tissue to scale=0.01 (almost no tissue)
# MFP=1250mm (10× skull size), key barely attenuated (2% over 50mm tissue)
# Should be near-identical to baseline, with very faint tissue colour
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key": kv_light([2.5, 2.1, 1.7]),
    "fill": fill_light([0.30, 0.42, 0.75]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.01),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S4_tissue_scale0p01",
   "Tissue scale=0.01 (MFP≈1250mm). Barely any scatter. Should look close to baseline.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Baseline + strong key (×6), same fill ratio
# No tissue. Brighter key light to simulate what we'd need with tissue added later.
# key=[15,12.6,10.2] = 6× brighter, fill=[1.8,2.52,4.5] = 6× brighter (same ratio)
# Expected: much brighter skull, same contrast ratio as baseline
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),   # 6× baseline key
    "fill": fill_light([1.8, 2.52, 4.5]),    # 6× baseline fill (same ratio)
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S5_bright_key_no_tissue",
   "6× brighter key + fill (no tissue). Tests: does bright fill raise background floor?", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Bright key + dim fill (no tissue)
# Strong key [15,12.6,10.2] but fill reduced to [0.05,0.07,0.14] (1/6 of baseline)
# Fill light at (-120,310,40) is ~11° from camera view direction — close to FOV edge.
# With bright key and dim fill, background should stay dark.
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),
    "fill": fill_light([0.05, 0.07, 0.14]),  # 1/6 of baseline fill
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S6_bright_key_dim_fill_no_tissue",
   "Strong key [15,12.6,10.2], dim fill [0.05,0.07,0.14]. Fill is ~11° from camera FOV edge.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — S6 + tissue scale=0.05
# Does adding nearly-transparent tissue to the bright-key setup create grey?
# This is the ideal setting if it works.
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),
    "fill": fill_light([0.05, 0.07, 0.14]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.05),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S7_bright_key_dim_fill_tissue0p05",
   "S6 + tissue scale=0.05. KEY TEST: does near-transparent tissue still create grey background?", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — S6 + tissue scale=0.01
# Even smaller tissue scale to check if 0.05 was still too much scatter
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),
    "fill": fill_light([0.05, 0.07, 0.14]),
    "tissue": tissue_medium("tissue_sigma.vol", "tissue_albedo.vol", scale=0.01),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S8_bright_key_dim_fill_tissue0p01",
   "S6 + tissue scale=0.01. Barely-there tissue. Background should be near-baseline.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — Best working config + cinematic RGB albedo TF
# Use scale from whichever of S7/S8 works, add cinematic Dappa 2016 colours
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),
    "fill": fill_light([0.05, 0.07, 0.14]),
    "tissue": tissue_medium("tissue_albedo.vol", "tissue_albedo.vol", scale=0.01),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S9_cinematic_rgb_tf_scale0p01",
   "S8 but with cinematic RGB albedo (rose tissue, ivory bone, amber fat). Dappa 2016 colours.", 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — Best config + gradient-magnitude blend (Siemens approx)
# GradMag suppresses tissue sigma at bone boundaries (Siemens hybrid phase approx)
# ──────────────────────────────────────────────────────────────────────────────
steps.append(({
    "type": "scene",
    "integrator": {"type": "volpath", "max_depth": 48, "rr_depth": 5},
    "sensor": sensor(),
    "key":  kv_light([15.0, 12.6, 10.2]),
    "fill": fill_light([0.05, 0.07, 0.14]),
    "tissue": tissue_medium("blended_sigma.vol", "blended_albedo.vol", scale=0.01),
    "skull": {"type": "obj", "filename": str(SKULL), "bsdf": bone_bsdf()},
}, "S10_gradmag_blend_scale0p01",
   "Full Siemens approx: gradmag blend sigma + cinematic RGB albedo, scale=0.01.", 1.0))


# ── Run all steps ─────────────────────────────────────────────────────────────

print(f"Running {len(steps)} calibration steps at {SPP} SPP ({RES}×{RES}) …\n")
print(f"{'Step':<45}  {'min':>4} {'max':>4} {'std':>5}  verdict")
print("─" * 70)

renders = []
labels  = []
verdicts = []

for i, (scene_dict, name, note, exposure) in enumerate(steps):
    print(f"\nStep {i}: {name}")
    srgb, verdict = render(scene_dict, name, note, exposure)
    renders.append(srgb)
    labels.append(f"S{i}\n{name[3:].split('_')[0]}")
    verdicts.append(verdict)

# ── Build comparison grid ─────────────────────────────────────────────────────

print("\nBuilding grid …")
n  = len(renders)
W  = H = RES
TH = 256   # thumbnail size for grid (smaller so it fits)
LH = 58
pad = 6
canvas = Image.new("RGB", (n*(TH+pad)-pad, TH+LH), (10,10,10))
draw   = ImageDraw.Draw(canvas)

for i, (img_arr, lbl, v) in enumerate(zip(renders, labels, verdicts)):
    thumb = Image.fromarray(img_arr).resize((TH, TH))
    x     = i*(TH+pad)
    canvas.paste(thumb, (x, LH))
    colour = (140,200,140) if v == "OK" else (200,120,120)
    draw.text((x+2, 2),  lbl, fill=(210,210,210))
    draw.text((x+2, 18), f"min={renders[i].min()} max={renders[i].max()}", fill=colour)
    draw.text((x+2, 34), f"std={renders[i].std():.1f}", fill=(150,150,170))
    draw.text((x+2, 46), v, fill=colour)

out = ROOT / "results/calibration_grid.png"
canvas.save(str(out))
print(f"Grid saved → {out}  ({out.stat().st_size//1024} KB)")
print("\nAll done. Calibration outputs: data/renders/gt/CQ500CT95/calibration/")
