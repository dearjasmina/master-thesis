"""
General-purpose pipeline: simple (PyVista Phong) + GT (Mitsuba path-traced) pair.

Produces a side-by-side 1024×512 pair PNG for any preprocessed NIfTI CT volume.
Uses the working S6-type GT configuration (bone surface, roughplastic GGX, key+fill).

Usage:
  python scripts/render_pair.py --volume "data/processed/CQ500CT312 CQ500CT312.nii.gz"
  python scripts/render_pair.py --volume "data/processed/CQ500CT200 CQ500CT200.nii.gz" --spp 128
"""

import argparse
import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

ROOT = Path(".")


# ── Camera ────────────────────────────────────────────────────────────────────

def compute_camera(shape_nxyz, spacing_mm):
    """3/4 front-top view — same formula as render_simple_preview.py."""
    nx, ny, nz = shape_nxyz
    sx, sy, sz = spacing_mm
    cx = nx * sx / 2
    cy = ny * sy / 2
    cz = nz * sz / 2
    radius = max(nx * sx, ny * sy, nz * sz) * 0.9
    return {
        "focal_point": [cx, cy, cz],
        "position": [cx + radius * 0.6, cy - radius * 1.1, cz + radius * 0.5],
        "up": [0.0, 0.0, 1.0],
        "fov_deg": 30.0,
    }


# ── PyVista simple render ─────────────────────────────────────────────────────

def render_simple(nifti_path: Path, size=512):
    img = nib.load(str(nifti_path))
    hu  = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]

    grid = pv.ImageData()
    grid.dimensions = np.array(hu.shape) + 1
    grid.spacing    = zooms
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["HU"] = hu.flatten(order="F")

    surface = (grid.cell_data_to_point_data()
               .contour(isosurfaces=[300], scalars="HU")
               .connectivity(extraction_mode="largest"))

    cam = compute_camera(hu.shape, zooms)

    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="#e8dcc8", ambient=0.25, diffuse=0.75,
                specular=0.15, specular_power=12, smooth_shading=True)
    pl.add_light(pv.Light(position=(1, -1, 1), intensity=0.8))
    pl.camera.focal_point = cam["focal_point"]
    pl.camera.position    = cam["position"]
    pl.camera.up          = cam["up"]
    simple_rgb = pl.screenshot(return_img=True)
    pl.close()

    return simple_rgb, surface, hu, zooms, cam


# ── Skull mesh → OBJ for Mitsuba ─────────────────────────────────────────────

def save_skull_obj(surface: pv.PolyData, path: Path):
    triangulated = surface.triangulate()
    verts = triangulated.points
    faces = triangulated.faces.reshape(-1, 4)[:, 1:]

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ── ACES tonemapping ──────────────────────────────────────────────────────────

def aces_filmic(hdr, exposure=1.0):
    x = hdr * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0, 1)**(1/2.2) * 255).astype(np.uint8)


# ── Mitsuba GT render ─────────────────────────────────────────────────────────

def render_mitsuba_gt(skull_obj: Path, cam: dict, aabb: list, spp: int, size=512):
    """S6-type: bone roughplastic + bright key + repositioned cool fill."""
    hx = (aabb[3]-aabb[0])/2; hy = (aabb[4]-aabb[1])/2; hz = (aabb[5]-aabb[2])/2
    cx = aabb[0]+hx;           cy = aabb[1]+hy;           cz = aabb[2]+hz
    sk = [cx, cy, cz]  # skull centre for light targeting

    CAM_TF = mi.ScalarTransform4f.look_at(
        origin=cam["position"], target=cam["focal_point"], up=cam["up"])

    # Light positions relative to skull size (same proportions as CT95)
    scale = max(aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2])
    kp = [cx + scale*0.73, cy - scale*0.54, cz + scale*1.28]   # key: upper-right-front
    fp = [cx - scale*0.59, cy + scale*1.52, cz + scale*1.48]   # fill: upper-left-behind

    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 48},
        "sensor": {
            "type": "perspective",
            "fov": cam["fov_deg"], "fov_axis": "y",
            "to_world": CAM_TF,
            "film": {
                "type": "hdrfilm", "width": size, "height": size,
                "rfilter": {"type": "tent"}, "pixel_format": "rgb",
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "key_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(kp, sk, [0,0,1]) @
                         mi.ScalarTransform4f.scale([70,70,1])),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [15.0, 12.6, 10.2]}},
        },
        "fill_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(fp, sk, [0,0,1]) @
                         mi.ScalarTransform4f.scale([100,100,1])),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [0.45, 0.63, 1.26]}},
        },
        "skull": {
            "type": "obj",
            "filename": str(skull_obj),
            "bsdf": {
                "type": "roughplastic",
                "distribution": "ggx",
                "alpha": 0.25,
                "int_ior": 1.49,
                "diffuse_reflectance": {"type": "rgb", "value": [0.93, 0.89, 0.82]},
            },
        },
    }

    s = mi.load_dict(scene)
    t0 = time.time()
    img = mi.render(s, spp=spp)
    print(f"  Rendered in {time.time()-t0:.1f}s")
    return np.array(img)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True, help="Path to preprocessed NIfTI .nii.gz")
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--out", default=None, help="Output pair PNG path")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    nifti_path = Path(args.volume)
    vol_name   = nifti_path.stem.replace(".nii", "")
    out_dir    = ROOT / "data/renders/gt" / vol_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path  = Path(args.out) if args.out else ROOT / f"results/pair_{vol_name}_spp{args.spp}.png"

    print(f"\n{'='*60}")
    print(f"Volume: {vol_name}")
    print(f"SPP: {args.spp}")
    print(f"{'='*60}")

    # ── Step 1: Simple render ─────────────────────────────────────────────────
    print("\n[1/3] PyVista simple render ...")
    simple_rgb, surface, hu, zooms, cam = render_simple(nifti_path, args.size)
    nx, ny, nz = hu.shape
    sx, sy, sz = zooms
    aabb = [0.0, 0.0, 0.0, float(nx*sx), float(ny*sy), float(nz*sz)]

    simple_path = out_dir / "simple.png"
    Image.fromarray(simple_rgb).save(str(simple_path))
    print(f"  Simple: {simple_rgb.shape}  min={simple_rgb.min()}  max={simple_rgb.max()}  saved→{simple_path}")

    # ── Step 2: Save skull OBJ ────────────────────────────────────────────────
    print("\n[2/3] Extracting skull mesh ...")
    skull_obj = out_dir / "skull_surface.obj"
    if not skull_obj.exists():
        save_skull_obj(surface, skull_obj)
        print(f"  Saved {skull_obj}  ({skull_obj.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  Skull OBJ already exists, reusing.")

    # ── Step 3: Mitsuba GT ────────────────────────────────────────────────────
    print(f"\n[3/3] Mitsuba GT ({args.spp} SPP) ...")
    hdr = render_mitsuba_gt(skull_obj, cam, aabb, args.spp, args.size)
    gt_srgb = aces_filmic(hdr, exposure=1.0)
    gt_path = out_dir / f"gt_spp{args.spp}.png"
    Image.fromarray(gt_srgb).save(str(gt_path))
    print(f"  GT: min={gt_srgb.min()}  max={gt_srgb.max()}  std={gt_srgb.std():.1f}  saved→{gt_path}")

    # ── Pair image ────────────────────────────────────────────────────────────
    gap = 12
    W = args.size * 2 + gap
    H = args.size + 50
    canvas = Image.new("RGB", (W, H), (12, 12, 12))
    canvas.paste(Image.fromarray(simple_rgb[:, :, :3] if simple_rgb.ndim==3 and simple_rgb.shape[2]==4 else simple_rgb), (0, 40))
    canvas.paste(Image.fromarray(gt_srgb), (args.size + gap, 40))
    d = ImageDraw.Draw(canvas)
    d.text((5, 5),             f"Input: {vol_name} — Simple (PyVista Phong)", fill=(180,180,180))
    d.text((args.size+gap+5, 5), f"GT: Mitsuba path-traced ({args.spp} SPP, ACES)", fill=(180,180,180))
    canvas.save(str(pair_path))
    print(f"\nPair saved → {pair_path}")

    # Meta JSON
    meta = {"volume": str(nifti_path), "shape_nxyz": [int(x) for x in hu.shape],
            "spacing_mm": [float(x) for x in zooms], "aabb_mm": [float(x) for x in aabb],
            "camera": {k: ([float(v) for v in val] if isinstance(val, (list, np.ndarray)) else float(val))
                       for k, val in cam.items()}}
    with open(out_dir / f"{vol_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
