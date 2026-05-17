"""
Full heterogeneous volumetric path tracer — bone + tissue encoded together.

Previous attempts failed because bone was a SURFACE MESH and tissue was a SEPARATE
MEDIUM cube. The bone mesh is invisible to the medium — photons travel around it,
illuminating the shadow side (scatter haze). Fix: encode bone as a VOLUMETRIC absorber
inside the SAME medium. Dense bone blocks the key light → real volumetric shadows.

Physics:
  scale=1.3, cortical bone (sigma_t_density=0.60):
    sigma_t_physical = 1.3 × 0.60 = 0.78 mm^-1 → MFP = 1.28mm
    Through 3mm cortical bone: exp(-2.34) = 9.6% transmittance → nearly opaque ✓
  scale=1.3, brain tissue (sigma_t_density=0.04):
    sigma_t_physical = 1.3 × 0.04 = 0.052 mm^-1 → MFP = 19mm
    Through 10mm scalp: exp(-0.52) = 60% transmittance → semi-transparent ✓
    Through 100mm skull: exp(-5.2) = 0.5% → very little through full skull

TF colours: Dappa et al. 2016 (Insights into Imaging), Table 1

Usage:
  python scripts/render_volumetric_full.py --scale 1.3 --spp 64
  python scripts/render_volumetric_full.py --scale 1.3 --spp 256
"""

import argparse
import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import nibabel as nib
import numpy as np
from PIL import Image

ROOT     = Path(".")
DATA     = ROOT / "data/renders/gt/CQ500CT95"
VOL_DIR  = DATA / "volumetric"
VOL_DIR.mkdir(parents=True, exist_ok=True)
NIFTI    = ROOT / "data/processed/CQ500CT95 CQ500CT95.nii.gz"
META     = DATA / "CQ500CT95 CQ500CT95_meta.json"


# ── Transfer function ─────────────────────────────────────────────────────────

def full_volumetric_tf(hu: np.ndarray):
    """
    Combined sigma_t + RGB albedo for full volumetric skull CT.

    Key design: bone sigma_t is HIGH (opaque), tissue sigma_t is LOW (semi-transparent).
    Ratio ~15:1 between bone and tissue at HU 700 vs HU 40.
    With scale=1.3, cortical bone has MFP≈1.3mm (opaque over 3mm), tissue MFP≈19mm.

    Albedo: Dappa 2016 cinematic palette (Insights into Imaging, Table 1).
    Phase function: g=0.70 forward scatter (biological tissue, Cheong 1990).
    """
    hu = hu.astype(np.float32)

    # sigma_t: continuous piecewise linear, bone ~15-25× higher than tissue
    ctrl_hu  = np.array([-1000, -200, -50,   80,   200,  400,  700, 1500, 3000], np.float32)
    ctrl_den = np.array([ 0.00,  0.00, 0.030, 0.050, 0.07, 0.20, 0.60, 1.00, 1.00], np.float32)
    sigma_t = np.interp(hu, ctrl_hu, ctrl_den).astype(np.float32)

    # RGB albedo
    albedo = np.zeros((*hu.shape, 3), np.float32)
    albedo[hu <  -200]                    = [0.00, 0.00, 0.00]  # air     → black
    albedo[(hu >= -200) & (hu <  -50)]    = [0.88, 0.74, 0.48]  # fat     → amber
    albedo[(hu >=  -50) & (hu <   80)]    = [0.82, 0.56, 0.52]  # tissue  → rose/pink
    albedo[(hu >=   80) & (hu <  200)]    = [0.75, 0.48, 0.45]  # dense tissue → dark rose
    albedo[(hu >=  200) & (hu <  700)]    = [0.90, 0.85, 0.78]  # trabecular bone → cream
    albedo[ hu >= 700]                    = [0.96, 0.92, 0.86]  # cortical bone → ivory/pearl

    return sigma_t, albedo


# ── Volume writer ─────────────────────────────────────────────────────────────

def write_vol(path: Path, data_nxyz: np.ndarray):
    """Write float32 Mitsuba gridvolume (1 or 3 channels)."""
    if data_nxyz.ndim == 3:
        mi_data = data_nxyz.transpose(2, 1, 0)[..., np.newaxis]
    else:
        mi_data = data_nxyz.transpose(2, 1, 0, 3)
    grid = mi.VolumeGrid(mi.TensorXf(mi_data.astype(np.float32)))
    grid.write(str(path))
    sz = path.stat().st_size / 1e6
    print(f"  {path.name}: {sz:.0f} MB  VolumeGrid.size()={list(grid.size())}")


def ensure_volumes():
    """Compute and cache combined volumetric TF volumes."""
    sigma_path  = VOL_DIR / "full_sigma.vol"
    albedo_path = VOL_DIR / "full_albedo.vol"

    if sigma_path.exists() and albedo_path.exists():
        print("  Volumetric TF volumes already cached.")
        return sigma_path, albedo_path

    print("Loading CT data …")
    img = nib.load(str(NIFTI))
    hu  = img.get_fdata(dtype=np.float32)
    print(f"  Shape: {hu.shape}  HU: [{hu.min():.0f}, {hu.max():.0f}]")

    print("Computing volumetric TF …")
    sigma_t, albedo = full_volumetric_tf(hu)
    print(f"  sigma_t: [{sigma_t.min():.4f}, {sigma_t.max():.4f}]  mean={sigma_t.mean():.4f}")
    print(f"  albedo:  mean_R={albedo[...,0].mean():.3f}  mean_G={albedo[...,1].mean():.3f}  mean_B={albedo[...,2].mean():.3f}")

    print("Writing volume files …")
    write_vol(sigma_path,  sigma_t)
    write_vol(albedo_path, albedo)
    return sigma_path, albedo_path


# ── Tone mapping ──────────────────────────────────────────────────────────────

def aces_filmic(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    x = hdr * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0, 1)
    return (ldr ** (1/2.2) * 255).astype(np.uint8)

def reinhard(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    x = hdr * exposure
    ldr = np.clip(x / (1 + x), 0, 1)
    return (ldr ** (1/2.2) * 255).astype(np.uint8)


# ── Main render ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale",    type=float, default=1.3,
                        help="Extinction scale. 1.3 → cortical bone MFP≈1.3mm (opaque).")
    parser.add_argument("--spp",      type=int,   default=64)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tonemap",  default="aces", choices=["aces","reinhard"])
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    sigma_path, albedo_path = ensure_volumes()

    with open(META) as f:
        meta = json.load(f)
    cam  = meta["camera"]
    aabb = meta["aabb_mm"]
    hx=(aabb[3]-aabb[0])/2; hy=(aabb[4]-aabb[1])/2; hz=(aabb[5]-aabb[2])/2
    cx=aabb[0]+hx; cy=aabb[1]+hy; cz=aabb[2]+hz
    # Cube: local [-1,1]^3 → world [cx-hx, cx+hx] etc. via translate(center) @ scale(half)
    VOL_TF = (mi.ScalarTransform4f.translate([cx,cy,cz]) @
              mi.ScalarTransform4f.scale([hx,hy,hz]))
    # GridVolume: grid [0,1]^3 → world [aabb_min, aabb_max] (full extent, not half)
    GRID_TF = (mi.ScalarTransform4f.translate([aabb[0],aabb[1],aabb[2]]) @
               mi.ScalarTransform4f.scale([aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2]]))
    CAM_TF = mi.ScalarTransform4f.look_at(
        origin=cam["position"], target=cam["focal_point"], up=cam["up"])
    SK = [117.0, 125.0, 72.0]

    tag = f"scale{args.scale}_spp{args.spp}_{args.tonemap}"
    out_path = Path(args.out) if args.out else DATA / f"vol_{tag}.png"

    print(f"\nScene: scale={args.scale}  SPP={args.spp}  tonemap={args.tonemap}")
    print(f"Output: {out_path}")

    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": 64,   # enough for multiple scatter through bone+tissue
            "rr_depth": 5,
        },
        "sensor": {
            "type": "perspective",
            "fov": cam["fov_deg"], "fov_axis": "y",
            "to_world": CAM_TF,
            "film": {
                "type": "hdrfilm",
                "width": 512, "height": 512,
                "rfilter": {"type": "tent"},
                "pixel_format": "rgb",
            },
            "sampler": {"type": "independent", "sample_count": args.spp},
        },
        # Strong warm key from upper-right-front
        "key_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at([300,-80,260], SK, [0,0,1]) @
                         mi.ScalarTransform4f.scale([70,70,1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [15.0, 12.6, 10.2]}},
        },
        # Cool fill — repositioned to z=300 (angle=35° from view, outside 15° half-FOV)
        "fill_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at([-120,310,300], SK, [0,0,1]) @
                         mi.ScalarTransform4f.scale([100,100,1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [0.45, 0.63, 1.26]}},
        },
        # Single heterogeneous volume: bone + tissue + air encoded together.
        # Dense bone blocks key light → volumetric shadow on opposite side.
        # Tissue is semi-transparent → warm glow from inside.
        "ct_volume": {
            "type": "cube",
            "to_world": VOL_TF,
            "bsdf": {"type": "null"},  # transparent shell — rays pass through into medium
            "interior": {
                "type": "heterogeneous",
                "scale": args.scale,
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(sigma_path),
                    "to_world": GRID_TF,
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(albedo_path),
                    "to_world": GRID_TF,
                },
                "phase": {"type": "hg", "g": 0.0},
            },
        },
    }

    print("\nBuilding scene …")
    s = mi.load_dict(scene)

    print(f"Rendering {args.spp} SPP … (this may take several minutes)")
    t0 = time.time()
    img = mi.render(s, spp=args.spp)
    dt  = time.time() - t0
    print(f"Render complete in {dt:.1f}s ({dt/60:.1f} min)")

    hdr = np.array(img)
    print(f"HDR: min={hdr.min():.4f}  max={hdr.max():.4f}  mean={hdr.mean():.4f}")

    if args.tonemap == "aces":
        srgb = aces_filmic(hdr, exposure=args.exposure)
    else:
        srgb = reinhard(hdr, exposure=args.exposure)

    print(f"sRGB: min={srgb.min()}  max={srgb.max()}  std={srgb.std():.1f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(srgb).save(str(out_path))
    print(f"Saved → {out_path}")

    # Quick alignment check
    simple_p = ROOT/"data/renders/simple/CQ500CT95 CQ500CT95.nii/color.png"
    if simple_p.exists():
        simple_g = np.array(Image.open(simple_p).convert("L"))
        gt_g     = srgb.mean(axis=2).astype(np.uint8)
        def bbox(im):
            rows=np.any(im>10,axis=1); cols=np.any(im>10,axis=0)
            if not rows.any(): return None
            r0,r1=np.where(rows)[0][[0,-1]]; c0,c1=np.where(cols)[0][[0,-1]]
            return r0,c0,r1,c1
        sb=bbox(simple_g); gb=bbox(gt_g)
        if sb and gb:
            inter=(max(0,min(sb[2],gb[2])-max(sb[0],gb[0]))*
                   max(0,min(sb[3],gb[3])-max(sb[1],gb[1])))
            union=((sb[2]-sb[0])*(sb[3]-sb[1])+(gb[2]-gb[0])*(gb[3]-gb[1])-inter)
            print(f"BBox IoU vs simple: {inter/union:.3f}")
        else:
            print("(GT all black — try adjusting scale or exposure)")


if __name__ == "__main__":
    main()
