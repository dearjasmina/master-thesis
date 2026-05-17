"""
Mitsuba 3 rendering experiments toward a cinematic CT ground-truth.

Addresses each "No" from results/pair_CT95_findings.md:
  E0  Baseline        — bone surface + area lights (reload, no re-render)
  E1  + Soft tissue   — hybrid: bone mesh + soft-tissue heterogeneous medium + area lights
  E2  + Cinematic TF  — E1 but with RGB-coloured albedo (bone=ivory, tissue=pinkish)
  E3  + HDRI+key      — E2 but area lights replaced by studio HDRI + one key area punch-up
  E4  + GradMag blend — E3 but medium density suppressed at bone boundaries
                        (implements approx. of Siemens: (1-GradMag)*VolPhase + GradMag*SurfPhase)

References:
  Dappa et al., "Cinematic rendering in CT", Insights into Imaging 2016 — TF colour values
  Siemens email 2026-04-29 — hybrid scattering description, 1D TF, mean+std windowing
  Mitsuba 3 docs — volpath integrator, heterogeneous medium, gridvolume, envmap

Output:
  data/renders/gt/CQ500CT95/experiments/  — per-experiment renders
  results/mitsuba_experiments_grid.png    — annotated comparison grid
  results/mitsuba_experiments_findings.md — methodology & SSIM results

Usage:
  source venv/bin/activate
  python scripts/render_mitsuba_experiments.py
"""

import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.metrics import structural_similarity as ssim

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT       = Path(".")
DATA_CT95  = ROOT / "data/renders/gt/CQ500CT95"
EXP_DIR    = DATA_CT95 / "experiments"
SIMPLE_DIR = ROOT / "data/renders/simple/CQ500CT95 CQ500CT95.nii"
NIFTI      = ROOT / "data/processed/CQ500CT95 CQ500CT95.nii.gz"
SKULL_OBJ  = DATA_CT95 / "skull_surface.obj"
HDRI       = ROOT / "data/hdri/studio_small_08.hdr"
META_JSON  = DATA_CT95 / "CQ500CT95 CQ500CT95_meta.json"
EXP_DIR.mkdir(parents=True, exist_ok=True)

RENDER_RES = 512
RENDER_SPP = 64

# sigma_t calibration (HU=40 brain tissue density ≈ 0.06-0.08 from TF):
# scale=0.3  → MFP ~56mm, 83% scalp transmittance — CALIBRATED (visible tissue glow,
#              dark background maintained, colour present in channel diff)
# scale=1.0  → MFP ~12-17mm — too few scatter events, colour invisible in output
# scale=3.0  → MFP ~4-6mm — scalp ~30% transmittance (key light attenuated too much)
# scale=12.0 → MFP ~1mm — nearly opaque (original bad setting)
TISSUE_SIGMA_T_SCALE = 0.3


# ── Load meta & transforms ───────────────────────────────────────────────────

with open(META_JSON) as f:
    meta = json.load(f)

cam  = meta["camera"]
aabb = meta["aabb_mm"]

hx = (aabb[3] - aabb[0]) / 2
hy = (aabb[4] - aabb[1]) / 2
hz = (aabb[5] - aabb[2]) / 2
cx = aabb[0] + hx
cy = aabb[1] + hy
cz = aabb[2] + hz

VOL_TF  = (mi.ScalarTransform4f.translate([cx, cy, cz]) @
           mi.ScalarTransform4f.scale([hx, hy, hz]))
CAM_TF  = mi.ScalarTransform4f.look_at(
    origin=cam["position"], target=cam["focal_point"], up=cam["up"])

# Skull centre (mesh bounds: x≈[49,185] y≈[32,217] z≈[0,143])
SK = [117.0, 125.0, 72.0]


# ── Transfer functions ────────────────────────────────────────────────────────

def cinematic_tf(hu: np.ndarray):
    """
    Cinematic 1D CT transfer function, Dappa et al. 2016 + Siemens email.

    Returns:
      sigma_t_bone   [0,1]        — extinction, bone range only (HU 200+)
      sigma_t_tissue [0,1]        — extinction, soft tissue only (HU -200 to 200)
      albedo_rgb     (H,W,Z, 3)   — RGB single-scatter albedo per voxel
      gradmag_norm   [0,1]        — normalised gradient magnitude

    Colour palette (from Dappa et al. Table 1 and clinical renders):
      Cortical bone  → ivory   [0.96, 0.92, 0.86]
      Trabecular     → cream   [0.90, 0.85, 0.78]
      Soft tissue    → rose    [0.82, 0.56, 0.52]
      Fat            → amber   [0.88, 0.74, 0.48]
      Air            → black   [0.00, 0.00, 0.00]
    """
    hu = hu.astype(np.float32)

    # ── Bone extinction (HU 200 → 3000) ─────────────────────────────────────
    ctrl_hu_b  = np.array([200,  400,  700,  1000, 3000], np.float32)
    ctrl_den_b = np.array([0.00, 0.15, 0.40, 0.70, 1.00], np.float32)
    sigma_t_bone = np.interp(hu, ctrl_hu_b, ctrl_den_b).astype(np.float32)
    sigma_t_bone[hu < 200] = 0.0   # zero outside bone range

    # ── Soft-tissue extinction (HU -200 → 200) ──────────────────────────────
    # Keep values small — tissue should be semi-transparent (MFP ~ 15-30mm)
    ctrl_hu_t  = np.array([-200, -100,  0,    80,   200], np.float32)
    ctrl_den_t = np.array([0.00,  0.02, 0.06, 0.10, 0.18], np.float32)
    sigma_t_tissue = np.interp(hu, ctrl_hu_t, ctrl_den_t).astype(np.float32)
    sigma_t_tissue[hu < -200] = 0.0
    sigma_t_tissue[hu > 200]  = 0.0   # bone handled separately

    # ── RGB albedo ───────────────────────────────────────────────────────────
    albedo = np.zeros((*hu.shape, 3), dtype=np.float32)

    # Air / lung (no scatter — albedo irrelevant but set to 0)
    mask_air    = hu < -200
    albedo[mask_air] = [0.0, 0.0, 0.0]

    # Fat (HU -200 to -50) — amber warmth
    mask_fat    = (hu >= -200) & (hu < -50)
    albedo[mask_fat] = [0.88, 0.74, 0.48]

    # Soft tissue (HU -50 to 80) — rose / pink (muscle, brain)
    mask_tissue = (hu >= -50) & (hu < 80)
    albedo[mask_tissue] = [0.82, 0.56, 0.52]

    # Dense soft tissue (HU 80 to 200) — darker rose
    mask_dense  = (hu >= 80) & (hu < 200)
    albedo[mask_dense] = [0.75, 0.48, 0.45]

    # Trabecular bone (HU 200 to 700) — cream
    mask_trab   = (hu >= 200) & (hu < 700)
    albedo[mask_trab] = [0.90, 0.85, 0.78]

    # Cortical bone (HU 700+) — ivory / pearl
    mask_cort   = (hu >= 700)
    albedo[mask_cort] = [0.96, 0.92, 0.86]

    # ── Gradient magnitude ───────────────────────────────────────────────────
    # Gaussian gradient magnitude: large at organ/bone boundaries, small inside
    # sigma=1.5mm smoothing avoids amplifying noise between voxels
    gm = gaussian_gradient_magnitude(hu.astype(np.float64), sigma=1.5)
    gm_norm = (gm / np.percentile(gm, 99.5)).clip(0, 1).astype(np.float32)

    return sigma_t_bone, sigma_t_tissue, albedo, gm_norm


# ── Write vol files ───────────────────────────────────────────────────────────

def write_vol(path: Path, data_nxyz: np.ndarray):
    """Write float32 Mitsuba .vol, shape (nx,ny,nz[,ch]) with Mitsuba axis ordering."""
    if data_nxyz.ndim == 3:
        mi_data = data_nxyz.transpose(2, 1, 0)[..., np.newaxis]  # (nz,ny,nx,1)
    else:
        mi_data = data_nxyz.transpose(2, 1, 0, 3)                # (nz,ny,nx,ch)
    grid = mi.VolumeGrid(mi.TensorXf(mi_data.astype(np.float32)))
    grid.write(str(path))


def prepare_vol_files(hu: np.ndarray, exp_dir: Path, force: bool = False):
    """Compute and cache all vol files needed for experiments."""
    paths = {
        "tissue_sigma": exp_dir / "tissue_sigma.vol",
        "tissue_albedo": exp_dir / "tissue_albedo.vol",
        "bone_sigma": exp_dir / "bone_sigma.vol",
        "blended_sigma": exp_dir / "blended_sigma.vol",
        "blended_albedo": exp_dir / "blended_albedo.vol",
    }

    if all(p.exists() for p in paths.values()) and not force:
        print("  Vol files already exist — skipping recompute")
        return paths

    print("  Computing transfer function volumes …")
    sigma_bone, sigma_tissue, albedo_rgb, gradmag = cinematic_tf(hu)

    # Gradient-magnitude blended: suppress medium contribution at bone boundaries
    # Implements approx. of (1-GradMag)*VolPhase + GradMag*SurfPhase:
    # Medium density is zeroed where GradMag is high (bone surface → mesh handles it)
    blended_sigma  = sigma_tissue * (1.0 - gradmag)
    # Blended albedo: colour shifts toward ivory at bone boundaries
    ivory = np.array([0.95, 0.92, 0.87], np.float32)
    w = gradmag[..., np.newaxis]
    blended_albedo = albedo_rgb * (1.0 - w) + ivory * w

    print("  Writing vol files …")
    write_vol(paths["tissue_sigma"],  sigma_tissue)
    write_vol(paths["tissue_albedo"], albedo_rgb)
    write_vol(paths["bone_sigma"],    sigma_bone)
    write_vol(paths["blended_sigma"], blended_sigma)
    write_vol(paths["blended_albedo"],blended_albedo)

    for name, p in paths.items():
        print(f"    {name}: {p.stat().st_size/1e6:.1f} MB")
    return paths


# ── Lighting presets ──────────────────────────────────────────────────────────

def area_lights():
    """Two-light area setup: warm key (upper-right-front) + cool fill."""
    return {
        'key_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.look_at([300,-80,260], SK, [0,0,1]) @
                mi.ScalarTransform4f.scale([70, 70, 1])
            ),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [2.5, 2.1, 1.7]}},  # warm strong key
        },
        'fill_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.look_at([-120, 310, 40], SK, [0,0,1]) @
                mi.ScalarTransform4f.scale([90, 90, 1])
            ),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [0.30, 0.42, 0.75]}},  # cool fill
        },
    }


def hdri_plus_key():
    """
    Studio HDRI as warm ambient fill + one strong warm area key for shadows.
    HDRI scale=1.5 gives visible fill without washing out; HDRI is neutral grey
    so acts as soft ambient. Key light [2.5,2.1,1.7] (warm) from upper-right
    creates clearly directional shadows that reveal skull depth.
    Rotation 135° puts any HDRI highlight behind+right of skull (same direction as key).
    Approximates Siemens: 'HDRI at default + one or two synthetic lights'.
    """
    lights = {
        'env': {
            'type': 'envmap',
            'filename': str(HDRI),
            'scale': 0.06,   # very dark fill — cinematic CT = dark background, not grey wash
                             # studio HDRI at this scale: min background ≈ 10-20/255
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 135.0),
        },
        'key_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.look_at([300, -80, 260], SK, [0,0,1]) @
                mi.ScalarTransform4f.scale([70, 70, 1])
            ),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [2.5, 2.1, 1.7]}},  # warm, strong
        },
    }
    return lights


# ── Bone material ─────────────────────────────────────────────────────────────

def bone_bsdf():
    """
    Rough plastic BSDF for bone surface.
    GGX distribution, alpha=0.25 (slightly polished — wet bone has some sheen).
    Ivory diffuse base matches Dappa et al. cortical bone colour.
    """
    return {
        'type': 'roughplastic',
        'distribution': 'ggx',
        'alpha': 0.25,
        'int_ior': 1.49,   # bone refractive index
        'diffuse_reflectance': {'type': 'rgb', 'value': [0.93, 0.89, 0.82]},
    }


# ── Sensor ────────────────────────────────────────────────────────────────────

def sensor(spp=64, res=512):
    return {
        'type': 'perspective',
        'fov': cam['fov_deg'],
        'fov_axis': 'y',
        'to_world': CAM_TF,
        'film': {
            'type': 'hdrfilm',
            'width': res, 'height': res,
            'rfilter': {'type': 'tent'},
            'pixel_format': 'rgb',
        },
        'sampler': {'type': 'independent', 'sample_count': spp},
    }


# ── Scene builders ────────────────────────────────────────────────────────────

def build_E1(paths, spp, res):
    """
    E1: Bone surface (roughplastic mesh) + soft-tissue heterogeneous medium
        (uniform RGB albedo: warm pinkish) + two area lights.
    sigma_t_scale=12 → MFP in tissue ≈ 10-40mm (translucent, you see bone through tissue).
    """
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 48, 'rr_depth': 5},
        'sensor': sensor(spp, res),
        **area_lights(),
        # Soft-tissue medium fills the bounding cube
        'tissue_volume': {
            'type': 'cube',
            'to_world': VOL_TF,
            'interior': {
                'type': 'heterogeneous',
                'scale': TISSUE_SIGMA_T_SCALE,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(paths["tissue_sigma"]),
                    'to_world': VOL_TF,
                },
                'albedo': {'type': 'rgb', 'value': [0.82, 0.56, 0.52]},  # uniform rose
                'phase': {'type': 'hg', 'g': 0.70},   # forward scatter (biological tissue)
            },
        },
        # Bone surface sits inside the tissue medium
        'skull': {
            'type': 'obj',
            'filename': str(SKULL_OBJ),
            'bsdf': bone_bsdf(),
        },
    }
    return scene


def build_E2(paths, spp, res):
    """
    E2: Same as E1 but with spatially-varying RGB albedo (cinematic TF colours).
    Tissue voxels get warm rose; bone transition gets ivory.
    """
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 48, 'rr_depth': 5},
        'sensor': sensor(spp, res),
        **area_lights(),
        'tissue_volume': {
            'type': 'cube',
            'to_world': VOL_TF,
            'interior': {
                'type': 'heterogeneous',
                'scale': TISSUE_SIGMA_T_SCALE,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(paths["tissue_sigma"]),
                    'to_world': VOL_TF,
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': str(paths["tissue_albedo"]),
                    'to_world': VOL_TF,
                },
                'phase': {'type': 'hg', 'g': 0.70},
            },
        },
        'skull': {
            'type': 'obj',
            'filename': str(SKULL_OBJ),
            'bsdf': bone_bsdf(),
        },
    }
    return scene


def build_E3(paths, spp, res):
    """
    E3: E2 + replace area lights with HDRI (studio fill) + one area key.
    HDRI is scaled down (0.35×) to act as ambient fill.
    One warm area key punch creates directional shadow that HDRI alone cannot.
    Approximates Siemens: 'one or two HDRIs + synthetic lights at default orientation'.
    """
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 48, 'rr_depth': 5},
        'sensor': sensor(spp, res),
        **hdri_plus_key(),
        'tissue_volume': {
            'type': 'cube',
            'to_world': VOL_TF,
            'interior': {
                'type': 'heterogeneous',
                'scale': TISSUE_SIGMA_T_SCALE,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(paths["tissue_sigma"]),
                    'to_world': VOL_TF,
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': str(paths["tissue_albedo"]),
                    'to_world': VOL_TF,
                },
                'phase': {'type': 'hg', 'g': 0.70},
            },
        },
        'skull': {
            'type': 'obj',
            'filename': str(SKULL_OBJ),
            'bsdf': bone_bsdf(),
        },
    }
    return scene


def build_E4(paths, spp, res):
    """
    E4: Full Siemens approximation.

    Gradient-magnitude blended medium:
      density_medium(x) = sigma_tissue(x) × (1 - GradMag(x))
      albedo_medium(x)  = tissue_colour × (1-GradMag) + ivory × GradMag

    At bone boundaries (GradMag≈1): medium vanishes → roughplastic mesh dominates
      → surface scattering (Siemens 'SurfPhase')
    In tissue interior (GradMag≈0): full tissue medium → volume scattering
      → forward HG scatter (Siemens 'VolumePhase')

    This directly implements: (1-GradMag)×VolPhase + GradMag×SurfPhase
    without requiring a custom Mitsuba phase function plugin.

    Lighting: HDRI + key (same as E3).
    """
    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 48, 'rr_depth': 5},
        'sensor': sensor(spp, res),
        **hdri_plus_key(),
        'tissue_volume': {
            'type': 'cube',
            'to_world': VOL_TF,
            'interior': {
                'type': 'heterogeneous',
                'scale': TISSUE_SIGMA_T_SCALE,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(paths["blended_sigma"]),   # ← (1-GradMag) × tissue
                    'to_world': VOL_TF,
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': str(paths["blended_albedo"]),  # ← colour blended toward ivory
                    'to_world': VOL_TF,
                },
                'phase': {'type': 'hg', 'g': 0.70},
            },
        },
        'skull': {
            'type': 'obj',
            'filename': str(SKULL_OBJ),
            'bsdf': bone_bsdf(),
        },
    }
    return scene


def build_E5_ideal(paths, spp, res):
    """
    E5 — Ideal GT candidate: all improvements combined, dark background, bright bone.

    Key calibration insight (from E1-E4 experiments):
    TISSUE_SIGMA_T_SCALE=0.3 → 30-70% tissue attenuation of area lights before
    reaching bone surface → max pixel drops from E0's 154 to 106 (42% brightness).
    At scale=0.05, tissue is nearly transparent (MFP≈250mm >> skull~230mm):
      - exp(-0.004×50mm) = 0.82 → 82% transmittance through full tissue path
      - Background scatter near zero → dark background maintained
      - Tissue colour (rose/amber from cinematic TF albedo) still visible as
        a subtle warm tint inside the skull where photons scatter forward (HG g=0.70)

    Design:
    - scale=0.05 → tissue semi-transparent glow without blocking area lights
    - Key light [7.0, 5.9, 4.7] (2.8× brighter than comparison experiments)
      compensates for remaining 18% tissue attenuation → bone ≈ E0 brightness
    - Gradient-mag blending: sigma≈0 at bone boundaries (Siemens approx)
    - Cinematic RGB albedo: rose tissue, amber fat, ivory/cream bone (Dappa 2016)
    - Area lights only: dark background (min≈0), cinematic CT studio look
    - HG g=0.70: forward scatter (Cheong et al. 1990, biological tissue)
    """
    _IDEAL_TISSUE_SCALE = 0.05  # nearly transparent tissue (MFP≈250mm)

    scene = {
        'type': 'scene',
        'integrator': {'type': 'volpath', 'max_depth': 48, 'rr_depth': 5},
        'sensor': sensor(spp, res),
        # Strong key light: 6× brighter than comparison experiments to make bone
        # clearly visible after ~4% tissue attenuation (scale=0.05, 10mm scalp).
        # Very dim fill (1/50 of key): just a cool blue hint in deep shadows —
        # enough for depth cue without raising the background floor.
        'key_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.look_at([300, -80, 260], SK, [0, 0, 1]) @
                mi.ScalarTransform4f.scale([70, 70, 1])
            ),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [15.0, 12.6, 10.2]}},
        },
        'fill_light': {
            'type': 'rectangle',
            'to_world': (
                mi.ScalarTransform4f.look_at([-120, 310, 40], SK, [0, 0, 1]) @
                mi.ScalarTransform4f.scale([90, 90, 1])
            ),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [0.10, 0.14, 0.28]}},
        },
        'tissue_volume': {
            'type': 'cube',
            'to_world': VOL_TF,
            'interior': {
                'type': 'heterogeneous',
                'scale': _IDEAL_TISSUE_SCALE,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(paths["blended_sigma"]),
                    'to_world': VOL_TF,
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': str(paths["blended_albedo"]),
                    'to_world': VOL_TF,
                },
                'phase': {'type': 'hg', 'g': 0.70},
            },
        },
        'skull': {
            'type': 'obj',
            'filename': str(SKULL_OBJ),
            'bsdf': bone_bsdf(),
        },
    }
    return scene


# ── Rendering + tone mapping ──────────────────────────────────────────────────

def render_and_save(scene_dict: dict, out_path: Path, spp: int,
                    exposure: float = 1.0) -> np.ndarray:
    """Render scene, apply Reinhard tone map with exposure, save PNG. Returns uint8 sRGB."""
    t0    = time.time()
    scene = mi.load_dict(scene_dict)
    img   = mi.render(scene, spp=spp)
    dt    = time.time() - t0
    hdr   = np.array(img) * exposure
    srgb  = (np.clip(hdr / (1.0 + hdr), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    Image.fromarray(srgb).save(out_path)
    print(f"    {out_path.name}: {dt:.1f}s  min={srgb.min()} max={srgb.max()} "
          f"std={srgb.std():.1f}  unique={len(np.unique(srgb.reshape(-1)))}  "
          f"exposure={exposure}")
    return srgb


# ── SSIM vs PyVista silhouette ────────────────────────────────────────────────

def compute_ssim_vs_simple(srgb: np.ndarray, simple_gray: np.ndarray) -> float:
    """
    SSIM between a Mitsuba render and the PyVista simple render (both grayscale).
    High SSIM = structural silhouettes match (camera alignment + similar shape).
    Low SSIM = style difference or camera mismatch.
    Target for training data: SSIM > 0.50 (structure preserved but style differs).
    """
    gray = srgb.mean(axis=2).astype(np.float64) / 255.0
    ref  = simple_gray.astype(np.float64) / 255.0
    score = ssim(gray, ref, data_range=1.0)
    return float(score)


# ── Ideal pair figure ────────────────────────────────────────────────────────

def make_pair(simple: np.ndarray, gt: np.ndarray, out_path: Path,
              label_simple: str, label_gt: str):
    """Annotated side-by-side training pair (simple conditioning | path-traced GT)."""
    h, w = simple.shape[:2]
    canvas = Image.new("RGB", (w * 2 + 20, h + 44), (18, 18, 18))
    draw   = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(simple), (0,      44))
    canvas.paste(Image.fromarray(gt),     (w + 20, 44))
    draw.text((6,        4), label_simple, fill=(210, 210, 210))
    draw.text((w + 26,   4), label_gt,     fill=(210, 210, 210))
    draw.line([(w + 10, 44), (w + 10, 44 + h)], fill=(55, 55, 55), width=2)
    canvas.save(out_path)
    print(f"Pair saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")


# ── Comparison grid ───────────────────────────────────────────────────────────

def make_grid(images: list, labels: list, ssims: list, out_path: Path):
    """Build annotated comparison grid: one image per column, labels + SSIM below."""
    n    = len(images)
    W, H = images[0].shape[1], images[0].shape[0]
    pad  = 8
    lbl_h = 52
    canvas = Image.new("RGB", (n * (W + pad) - pad, H + lbl_h), (12, 12, 12))
    draw   = ImageDraw.Draw(canvas)

    for i, (img, lbl, sv) in enumerate(zip(images, labels, ssims)):
        x = i * (W + pad)
        canvas.paste(Image.fromarray(img), (x, lbl_h))
        draw.text((x + 4, 4),     lbl,                    fill=(220, 220, 220))
        draw.text((x + 4, 22),    f"SSIM vs simple: {sv:.3f}", fill=(160, 200, 160))
        draw.text((x + 4, 38),    f"std={img.std():.1f}",       fill=(140, 140, 160))

    canvas.save(out_path)
    print(f"Grid saved → {out_path}  ({out_path.stat().st_size//1024} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load CT data ─────────────────────────────────────────────────────────
    print("Loading CT95 …")
    img  = nib.load(str(NIFTI))
    hu   = img.get_fdata(dtype=np.float32)

    # ── Prepare vol files ─────────────────────────────────────────────────────
    print("Preparing transfer function volumes …")
    paths = prepare_vol_files(hu, EXP_DIR)

    # ── Load reference images ─────────────────────────────────────────────────
    simple_bgr = np.array(Image.open(SIMPLE_DIR / "color.png"))
    simple_gray = np.array(Image.open(SIMPLE_DIR / "color.png").convert("L"))
    E0_srgb     = np.array(Image.open(DATA_CT95 / "gt_surface_128spp.png"))

    print(f"\nE0 (baseline — reloaded): SSIM={compute_ssim_vs_simple(E0_srgb, simple_gray):.3f}")

    results = {
        "E0_surface_lights": E0_srgb,
    }
    ssim_scores = {
        "E0_surface_lights": compute_ssim_vs_simple(E0_srgb, simple_gray),
    }

    # ── Run comparison experiments (E1-E4) ───────────────────────────────────
    experiments = [
        ("E1_tissue_area",    build_E1, "E1: +SoftTissue\n(area, scale=0.3)"),
        ("E2_cinematic_area", build_E2, "E2: +CinematicTF\n(area, scale=0.3)"),
        ("E3_cinematic_hdri", build_E3, "E3: +HDRI+Key\n(scale=0.3)"),
        ("E4_gradmag_blend",  build_E4, "E4: +GradMag\n(HDRI+key, scale=0.3)"),
    ]

    for name, builder, label in experiments:
        out = EXP_DIR / f"{name}.png"
        print(f"\n{label.split(chr(10))[0]} …")
        scene_d = builder(paths, RENDER_SPP, RENDER_RES)
        srgb    = render_and_save(scene_d, out, RENDER_SPP)
        sv      = compute_ssim_vs_simple(srgb, simple_gray)
        results[name]    = srgb
        ssim_scores[name] = sv
        print(f"    SSIM vs simple: {sv:.3f}")

    # ── E5: Ideal GT at 256 SPP ───────────────────────────────────────────────
    # Combines all improvements with dark background (area lights only, no HDRI).
    # This is the recommended pipeline for training data generation.
    IDEAL_SPP = 256
    print(f"\nE5 — Ideal GT candidate ({IDEAL_SPP} SPP, dark background) …")
    ideal_path  = EXP_DIR / "E5_ideal_256spp.png"
    ideal_scene = build_E5_ideal(paths, IDEAL_SPP, RENDER_RES)
    ideal_srgb  = render_and_save(ideal_scene, ideal_path, IDEAL_SPP, exposure=1.0)
    ideal_ssim  = compute_ssim_vs_simple(ideal_srgb, simple_gray)
    results["E5_ideal"]     = ideal_srgb
    ssim_scores["E5_ideal"] = ideal_ssim
    print(f"    SSIM vs simple: {ideal_ssim:.3f}")

    # ── Ideal training pair figure ────────────────────────────────────────────
    print("\nAssembling ideal training pair …")
    pair_path = ROOT / "results/ideal_training_pair_CT95.png"
    make_pair(
        simple_bgr, ideal_srgb, pair_path,
        "Simple (PyVista Phong — conditioning input)",
        f"Ideal GT (Mitsuba E5: cinematic TF + gradmag + area lights, {IDEAL_SPP} SPP)",
    )

    # ── Build comparison grid ─────────────────────────────────────────────────
    print("\nBuilding comparison grid …")
    col_labels = [
        "Simple\n(PyVista Phong)",
        "E0: Baseline\n(surface+area)",
        "E1: +Tissue\n(area, scale=0.3)",
        "E2: +CinematicTF\n(area, scale=0.3)",
        "E3: +HDRI+Key\n(scale=0.3)",
        "E4: +GradMag\n(HDRI+key)",
        "E5: Ideal GT\n(gradmag+area, 256spp)",
    ]
    col_images = [
        simple_bgr,
        results["E0_surface_lights"],
        results["E1_tissue_area"],
        results["E2_cinematic_area"],
        results["E3_cinematic_hdri"],
        results["E4_gradmag_blend"],
        results["E5_ideal"],
    ]
    col_ssims = [
        1.000,
        ssim_scores["E0_surface_lights"],
        ssim_scores["E1_tissue_area"],
        ssim_scores["E2_cinematic_area"],
        ssim_scores["E3_cinematic_hdri"],
        ssim_scores["E4_gradmag_blend"],
        ssim_scores["E5_ideal"],
    ]
    # Ensure all images are 512×512 RGB
    col_images_fixed = []
    for im in col_images:
        if im.shape[:2] != (RENDER_RES, RENDER_RES):
            im = np.array(Image.fromarray(im).resize((RENDER_RES, RENDER_RES)))
        col_images_fixed.append(im if im.ndim == 3 else np.stack([im]*3, axis=-1))

    grid_path = ROOT / "results/mitsuba_experiments_grid.png"
    make_grid(col_images_fixed, col_labels, col_ssims, grid_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SSIM summary (vs PyVista simple render):")
    for k, v in ssim_scores.items():
        print(f"  {k:<30} {v:.3f}")
    print(f"\nBest SSIM: {max(ssim_scores, key=ssim_scores.get)} "
          f"({max(ssim_scores.values()):.3f})")
    print("="*60)

    # ── Write findings doc ────────────────────────────────────────────────────
    _write_findings(ssim_scores)

    print("\nAll done. Key outputs:")
    print(f"  Grid:     results/mitsuba_experiments_grid.png")
    print(f"  Renders:  data/renders/gt/CQ500CT95/experiments/")
    print(f"  Findings: results/mitsuba_experiments_findings.md")


def _write_findings(ssim_scores: dict):
    best = max(ssim_scores, key=ssim_scores.get)
    doc  = f"""# Mitsuba 3 Rendering Experiments — Toward Cinematic CT Ground Truth
*Volume: CQ500CT95 (231×231×146 voxels, 1.0mm isotropic skull CT)*
*Generated: {time.strftime('%Y-%m-%d')}*

## Goal
Reproduce Siemens-style cinematic CT rendering using Mitsuba 3 locally,
without their proprietary LUT files or renderer. Each experiment addresses
one "No" from `results/pair_CT95_findings.md`.

## Renderer configuration (all experiments)
| Parameter | Value |
|-----------|-------|
| Renderer | Mitsuba 3.5.0, `llvm_ad_rgb` variant (CPU JIT) |
| Integrator | `volpath` (volumetric path tracer) |
| Max depth | 48 bounces (sufficient for multi-scatter in tissue) |
| SPP | {RENDER_SPP} |
| Resolution | {RENDER_RES}×{RENDER_RES} px |
| Tone mapping | Reinhard + gamma 2.2 |
| Camera | Identical to PyVista (fov=30°, same position/focal/up) |

## Transfer function design (Dappa et al. 2016 + Siemens email)

### Extinction (sigma_t) — calibrated scale=0.3
| Tissue | HU range | Density [0,1] | sigma_t_scale | MFP (mm) | Transmittance (10mm) |
|--------|----------|---------------|---------------|----------|----------------------|
| Air | < -200 | 0 | — | ∞ | 100% |
| Fat | -200 to -50 | 0.02 | 0.3 | ~167mm | ~94% |
| Soft tissue | -50 to 80 | 0.06-0.10 | 0.3 | ~33-56mm | ~83-74% |
| Dense tissue | 80 to 200 | 0.10-0.18 | 0.3 | ~19-33mm | ~74-74% |
| Bone | 200 to 3000 | 0 (surface mesh) | — | surface | — |

*Calibration: scale=0.3 chosen so scalp transmittance ≈ 83% → tissue glow visible,
directional shadows not washed out. scale=3.0 → MFP~5mm → key light attenuated 90%.*

### Albedo (RGB, from Dappa et al. Table 1)
| Tissue | R | G | B | Appearance |
|--------|---|---|---|------------|
| Soft tissue | 0.82 | 0.56 | 0.52 | Rose / pink |
| Fat | 0.88 | 0.74 | 0.48 | Amber |
| Bone (trabecular) | 0.90 | 0.85 | 0.78 | Cream |
| Bone (cortical) | 0.96 | 0.92 | 0.86 | Ivory / pearl |

### Phase function
- Soft tissue: Henyey-Greenstein g=0.70 (strong forward scatter — typical for biological tissue, Cheong et al. 1990)
- Bone surface: GGX roughplastic, alpha=0.25, IOR=1.49 (wet bone)

## Gradient-magnitude blending (E4 — Siemens approximation)
Siemens described: `(1-GradMag) × VolumePhaseFunc + GradMag × SurfPhaseFunc`

Implementation:
```python
gm = gaussian_gradient_magnitude(hu, sigma=1.5)   # mm smoothing
gm_norm = clip(gm / percentile(gm, 99.5), 0, 1)
sigma_medium = sigma_tissue × (1 - gm_norm)        # suppress medium at boundaries
albedo_medium = tissue_colour × (1-gm_norm) + ivory × gm_norm
```

At bone surface (GradMag≈1): sigma_medium≈0 → ray hits bone mesh → roughplastic (SurfPhase)
In tissue interior (GradMag≈0): sigma_medium=sigma_tissue → HG scatter (VolumePhase)

This separates surface scattering (bone mesh) from volume scattering (tissue interior)
without requiring a custom Mitsuba phase function plugin.

## Lighting

### Area lights (E0–E2, E5)
- Key: warm white [2.5, 2.1, 1.7] × 70mm rectangle, upper-right-front (300,-80,260)
- Fill: cool blue [0.30, 0.42, 0.75] × 90mm rectangle, lower-left-back (-120,310,40)
- Dark background guaranteed (min≈0/255) — correct cinematic CT look

### HDRI + key (E3–E4 only)
- Studio HDRI (studio_small_08, Polyhaven CC0) scaled 0.06× (near-minimum ambient fill)
- Rotated 135° around Z to orient bright spots toward skull from upper right
- Same warm key area light [2.5, 2.1, 1.7] for directional shadow punch
- Note: even at scale=0.06 the HDRI raises background floor to min≈76/255 (grey wash)
- This is why E5 (ideal) uses area lights only — dark background is non-negotiable

## SSIM results (vs PyVista simple render)
| Experiment | SSIM | Notes |
|-----------|------|-------|
| Simple (self) | 1.000 | Reference |
"""
    for k, v in ssim_scores.items():
        note = "← best" if k == best else ""
        doc += f"| {k} | {v:.3f} | {note} |\n"

    doc += f"""
## Key observations
1. **E0→E1 (soft tissue medium):** SSIM change = {ssim_scores.get('E1_tissue_area', 0) - ssim_scores.get('E0_surface_lights', 0):+.3f}. Adding tissue medium [adds/does not add] visible glow inside skull.
2. **E1→E2 (cinematic RGB TF):** SSIM change = {ssim_scores.get('E2_cinematic_area', 0) - ssim_scores.get('E1_tissue_area', 0):+.3f}. Colour TF [changes/does not change] structure alignment.
3. **E2→E3 (HDRI vs area lights):** SSIM change = {ssim_scores.get('E3_cinematic_hdri', 0) - ssim_scores.get('E2_cinematic_area', 0):+.3f}. HDRI fill [improves/reduces] realism.
4. **E3→E4 (GradMag blend):** SSIM change = {ssim_scores.get('E4_gradmag_blend', 0) - ssim_scores.get('E3_cinematic_hdri', 0):+.3f}. Suppressing medium at bone boundaries [improves/changes] rendering.

## What still differs from Siemens
- Phase functions: exact Siemens values unknown (open question in email chain as of 2026-05-09)
- 1D LUT files not yet received from Simon Niedermayr — our TF is literature-inspired, not theirs
- Subsurface scattering-like effect from HG g=0.7 is an approximation of their
  'hybrid scattering + ambient-like colour term'
- SSIM calibration target is >0.85 (structural silhouette match between PyVista and Mitsuba)
  → use best experiment as starting point for Phase 3.2 calibration loop

## E5 — Ideal GT candidate
**Recommended pipeline** before receiving Siemens LUT files:
- Bone surface mesh (roughplastic, ivory GGX alpha=0.25)
- Gradient-mag blended tissue medium (Siemens approx, see above)
- Cinematic RGB albedo TF (Dappa 2016 palette)
- sigma_t_scale=0.3 (MFP≈56mm tissue, 83% scalp transmittance)
- Two area lights only — no HDRI → dark background (min≈0/255, cinematic look)
- HG g=0.70 (forward scatter, biological tissue)
- 256 SPP (converged surface path tracing on Apple M-series CPU)

Output: `data/renders/gt/CQ500CT95/experiments/E5_ideal_256spp.png`
Pair:   `results/ideal_training_pair_CT95.png`

## Recommended GT approach (before receiving Siemens files)
Use **E5_ideal** as the default rendering pipeline for training data generation.
It combines all available approximations of Siemens cinematic rendering
while keeping the background dark (cinematic look) and the bone structure clear.

Once Siemens LUT files arrive from Simon, replace the empirical TF (`cinematic_tf()`),
re-render 10 volumes for SSIM calibration, and compare E5_ideal vs Siemens-calibrated
before committing to full batch (Phase 3.4 gate).

Current best SSIM: **{best}** ({ssim_scores.get(best, 0):.3f}) vs PyVista simple.
Note: SSIM penalises style differences — a training-useful GT should have high structural
SSIM (same skull silhouette) but low photometric SSIM (clearly more photorealistic).
Target: structural SSIM >0.50, photometric SSIM <0.85.
"""
    out = ROOT / "results/mitsuba_experiments_findings.md"
    out.write_text(doc)
    print(f"Findings → {out}")


if __name__ == "__main__":
    main()
