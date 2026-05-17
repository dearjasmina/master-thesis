"""
Render a path-traced ground-truth image from Mitsuba 3 .vol files.

Uses the volumetric path tracer (volpath) with a heterogeneous medium
to produce physically-based lighting, soft shadows, and subsurface-like
scattering — the photorealistic target that ControlNet learns to produce.

Camera parameters must match render_simple_preview.py exactly so that
the simple/GT pair is geometrically aligned.

Usage (preview — fast, noisy):
    python scripts/render_mitsuba_gt.py \
        --meta  data/renders/gt/CQ500CT95/CQ500CT95_CQ500CT95_meta.json \
        --hdri  data/hdri/studio_small_08.hdr \
        --out   data/renders/gt/CQ500CT95/gt_preview_32spp.png \
        --spp 32 --sigma_t_scale 80

Usage (final GT):
    python scripts/render_mitsuba_gt.py \
        --meta  data/renders/gt/CQ500CT95/CQ500CT95_CQ500CT95_meta.json \
        --hdri  data/hdri/studio_small_08.hdr \
        --out   data/renders/gt/CQ500CT95/gt_128spp.png \
        --spp 128 --sigma_t_scale <tuned_value>
"""

import argparse
import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import numpy as np
from PIL import Image


# ── Tone mapping ─────────────────────────────────────────────────────────────

def reinhard_tonemap(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Reinhard global tone mapping + gamma correction → uint8 sRGB."""
    hdr = hdr * exposure
    ldr = hdr / (1.0 + hdr)         # Reinhard: L_display = L / (1 + L)
    ldr = np.clip(ldr, 0.0, 1.0)
    srgb = ldr ** (1.0 / 2.2)       # gamma
    return (srgb * 255).astype(np.uint8)


# ── Scene builder ─────────────────────────────────────────────────────────────

def build_scene(meta: dict, density_path: Path, albedo_path: Path,
                hdri_path: Path, spp: int, sigma_t_scale: float,
                hg_g: float) -> dict:
    """
    Build a Mitsuba 3 scene dict for heterogeneous CT volume rendering.

    Coordinate system matches PyVista exactly:
      - Volume AABB: [0, nx*dx] × [0, ny*dy] × [0, nz*dz] mm
      - Camera position/focal/up taken from meta JSON (computed identically to
        set_skull_camera() in render_simple_preview.py)
      - Up vector: (0, 0, 1) — Z-up, same as VTK/PyVista default

    The gridvolume covers [-1,1]^3 by default; we scale it to match the AABB
    using a to_world transform: translate(center) @ scale(half_extents).
    """
    aabb    = meta["aabb_mm"]             # [xmin, ymin, zmin, xmax, ymax, zmax]
    cam     = meta["camera"]

    # Half-extents and centre of volume AABB
    hx = (aabb[3] - aabb[0]) / 2.0
    hy = (aabb[4] - aabb[1]) / 2.0
    hz = (aabb[5] - aabb[2]) / 2.0
    cx = aabb[0] + hx
    cy = aabb[1] + hy
    cz = aabb[2] + hz

    # Transform: maps Mitsuba's [-1,1]^3 cube → [aabb_min, aabb_max]
    vol_transform = (
        mi.ScalarTransform4f.translate([cx, cy, cz]) @
        mi.ScalarTransform4f.scale([hx, hy, hz])
    )

    # Camera: lookat identical to PyVista's set_skull_camera()
    pos   = cam["position"]
    focal = cam["focal_point"]
    up    = cam["up"]
    cam_transform = mi.ScalarTransform4f.look_at(
        origin=pos, target=focal, up=up
    )

    # Emitter: HDRI if file exists, else fall back to constant
    if hdri_path.exists():
        emitter = {
            'type': 'envmap',
            'filename': str(hdri_path),
        }
    else:
        print("  HDRI not found — using constant emitter (uniform white lighting)")
        emitter = {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': 1.0}
        }

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'volpath',
            'max_depth': 32,    # enough bounces for scattering inside skull
            'rr_depth': 5,      # start Russian roulette at depth 5
        },
        'sensor': {
            'type': 'perspective',
            'fov': cam["fov_deg"],
            'fov_axis': 'y',
            'to_world': cam_transform,
            'film': {
                'type': 'hdrfilm',
                'width':  512,
                'height': 512,
                'rfilter': {'type': 'tent'},   # anti-aliasing filter
                'pixel_format': 'rgb',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
        },
        # The cube shape encloses the medium — Mitsuba requires a shape to bound it
        'ct_volume': {
            'type': 'cube',
            'to_world': vol_transform,
            'interior': {
                'type': 'heterogeneous',
                # scale multiplies sigma_t values from the vol file [0,1] by this factor
                # to get physical extinction in scene units (mm^-1 equivalent)
                'scale': sigma_t_scale,
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': str(density_path),
                    'to_world': vol_transform,
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': str(albedo_path),
                    'to_world': vol_transform,
                },
                'phase': {
                    'type': 'hg',
                    'g': hg_g,   # forward scattering: 0=isotropic, 0.9=strong forward
                },
            },
        },
        'emitter': emitter,
    }
    return scene


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mitsuba 3 path-traced GT renderer")
    parser.add_argument("--meta",          required=True, help="JSON meta file from nifti_to_vol.py")
    parser.add_argument("--hdri",          default="data/hdri/studio_small_08.hdr",
                        help="HDR environment map (.hdr or .exr)")
    parser.add_argument("--out",           required=True, help="Output PNG path")
    parser.add_argument("--spp",           type=int,   default=64,
                        help="Samples per pixel (32=fast preview, 128=good, 512=final)")
    parser.add_argument("--sigma_t_scale", type=float, default=80.0,
                        help="Extinction scale factor. Increase if too transparent, decrease if too dense.")
    parser.add_argument("--hg_g",          type=float, default=0.3,
                        help="Henyey-Greenstein g parameter (0=isotropic, 0.9=forward)")
    parser.add_argument("--exposure",      type=float, default=1.0,
                        help="Tone mapping exposure multiplier")
    args = parser.parse_args()

    meta_path    = Path(args.meta)
    out_path     = Path(args.out)
    hdri_path    = Path(args.hdri)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate vol files next to the meta file
    vol_dir = meta_path.parent
    density_files = list(vol_dir.glob("*_density.vol"))
    albedo_files  = list(vol_dir.glob("*_albedo.vol"))
    if not density_files or not albedo_files:
        raise FileNotFoundError(
            f"Could not find *_density.vol and *_albedo.vol in {vol_dir}. "
            "Run nifti_to_vol.py first."
        )
    density_path = density_files[0]
    albedo_path  = albedo_files[0]

    print(f"Density vol : {density_path.name}")
    print(f"Albedo vol  : {albedo_path.name}")
    print(f"HDRI        : {hdri_path}")
    print(f"SPP         : {args.spp}")
    print(f"sigma_t_scale: {args.sigma_t_scale}")
    print(f"HG g        : {args.hg_g}")

    with open(meta_path) as f:
        meta = json.load(f)

    print("\nBuilding Mitsuba scene …")
    scene_dict = build_scene(
        meta=meta,
        density_path=density_path,
        albedo_path=albedo_path,
        hdri_path=hdri_path,
        spp=args.spp,
        sigma_t_scale=args.sigma_t_scale,
        hg_g=args.hg_g,
    )
    scene = mi.load_dict(scene_dict)

    print(f"Rendering at {args.spp} SPP (this may take several minutes on CPU) …")
    t0  = time.time()
    img = mi.render(scene, spp=args.spp)
    elapsed = time.time() - t0
    print(f"Render complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    hdr   = np.array(img)
    print(f"HDR image stats: min={hdr.min():.4f}  max={hdr.max():.4f}  mean={hdr.mean():.4f}")

    # Warn if image is suspiciously empty
    if hdr.max() < 0.01:
        print("WARNING: image is nearly all black — volume may be too transparent. "
              "Try increasing --sigma_t_scale (e.g. 200, 500).")
    elif hdr.mean() > 0.9:
        print("WARNING: image is nearly all white — volume may be too dense. "
              "Try decreasing --sigma_t_scale (e.g. 10, 5).")

    srgb = reinhard_tonemap(hdr, exposure=args.exposure)
    Image.fromarray(srgb).save(out_path)
    print(f"Saved → {out_path}")

    # Quick alignment metric vs simple render (if it exists)
    stem     = meta_path.stem.replace('_meta', '')
    # Try to find matching simple render
    simple_candidates = list(
        Path("data/renders/simple").glob(f"**/{stem.split('_')[0]}*/color.png")
    )
    if simple_candidates:
        simple_img = np.array(Image.open(simple_candidates[0]).convert("L"))
        gt_img     = np.array(Image.open(out_path).convert("L"))
        def bbox(im):
            rows = np.any(im > 10, axis=1); cols = np.any(im > 10, axis=0)
            if not rows.any():
                return None
            r0,r1 = np.where(rows)[0][[0,-1]]; c0,c1 = np.where(cols)[0][[0,-1]]
            return r0, c0, r1, c1
        sb = bbox(simple_img); gb = bbox(gt_img)
        if sb and gb:
            inter = (max(0, min(sb[2],gb[2])-max(sb[0],gb[0])) *
                     max(0, min(sb[3],gb[3])-max(sb[1],gb[1])))
            union = ((sb[2]-sb[0])*(sb[3]-sb[1]) +
                     (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
            iou = inter / union if union > 0 else 0.0
            print(f"\nAlignment check vs {simple_candidates[0].name}:")
            print(f"  Simple bbox : {sb}")
            print(f"  GT bbox     : {gb}")
            print(f"  BBox IoU    : {iou:.3f}  {'OK' if iou > 0.80 else 'LOW — check camera transform'}")
        else:
            print("\nAlignment check: GT image appears all black — check sigma_t_scale")
    else:
        print("\n(Skipping alignment check — no matching simple render found)")


if __name__ == "__main__":
    main()
