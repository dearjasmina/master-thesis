"""
render_pair_totalseg.py — one training pair from a TotalSegmentator CT subject.

  Simple render : multi-tissue PyVista Phong (fast conditioning image)
  GT render     : Mitsuba volpath with per-tissue SSS interior media (organs/muscle)
                  and roughplastic surface (bone/vessels), DoF, ACES tonemapping

Output:
  data/renders/totalseg/{subject}/simple_a{angle}.png
  data/renders/totalseg/{subject}/gt_spp{N}_a{angle}.png
  data/renders/totalseg/{subject}/meshes/*.obj
  results/totalseg_pairs/{subject}_grid_spp{N}.png

Usage:
    python scripts/render_pair_totalseg.py \\
        --subject s0329 \\
        --dataset /Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201 \\
        --spp 512 --angles 3
"""

import argparse
import math
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


# ── Tissue definitions ────────────────────────────────────────────────────────
# Order: deepest → foreground (muscles → bone → organs → vessels)
#
# Columns: (seg_name, method, simple_hex, gt_rgb, p5, p6)
#
#   method "bone"   : roughplastic GGX IOR=1.55  p5=alpha  p6=None
#   method "surface": roughplastic GGX IOR=1.40  p5=alpha  p6=None
#   method "sss"    : null BSDF + homogeneous interior medium
#                     p5=sigma_t (mm⁻¹ from Jacques 2013)  p6=g (HG phase)
#
# simple_hex : hex derived from gt_rgb so both sides share the same palette.
# gt_rgb     : actual tissue colour from surgical photography (NOT atlas labels).
#
# SSS sigma_t values (mm⁻¹) — Jacques PMB 2013, ~630 nm:
#   liver 32.6  kidney 27.5  spleen 30.0  muscle 27.3  stomach ~28  lung 1.5

TISSUES = [
    # muscles (deepest — render first so organs occlude correctly)
    # alpha=0.22 → moderately rough flesh; rim light (added in scene) provides SSS-like glow
    ("autochthon_left",       "soft",    "#8C3D2E", [0.55, 0.24, 0.18], 0.22, None),
    ("autochthon_right",      "soft",    "#8C3D2E", [0.55, 0.24, 0.18], 0.22, None),
    # lungs: lighter and more translucent-looking
    ("lung_lower_lobe_left",  "soft",    "#C7BFBD", [0.78, 0.75, 0.74], 0.35, None),
    ("lung_lower_lobe_right", "soft",    "#C7BFBD", [0.78, 0.75, 0.74], 0.35, None),
    ("lung_upper_lobe_left",  "soft",    "#C7BFBD", [0.78, 0.75, 0.74], 0.35, None),
    ("lung_upper_lobe_right", "soft",    "#C7BFBD", [0.78, 0.75, 0.74], 0.35, None),
    # vertebrae (dry cortical bone)
    ("vertebrae_T12",         "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    ("vertebrae_L1",          "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    ("vertebrae_L2",          "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    ("vertebrae_L3",          "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    ("vertebrae_L4",          "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    ("vertebrae_L5",          "bone",    "#EBE0CC", [0.92, 0.88, 0.80], 0.25, None),
    # abdominal organs — alpha=0.06 → very smooth/wet → strong specular highlight
    # rim light from behind gives SSS-like translucent edge glow
    ("stomach",               "soft",    "#B39E8F", [0.70, 0.62, 0.56], 0.06, None),
    ("gallbladder",           "soft",    "#8CB84D", [0.55, 0.72, 0.30], 0.06, None),
    ("spleen",                "soft",    "#611A29", [0.38, 0.10, 0.16], 0.06, None),
    ("kidney_right",          "soft",    "#854230", [0.52, 0.26, 0.19], 0.06, None),
    ("kidney_left",           "soft",    "#854230", [0.52, 0.26, 0.19], 0.06, None),
    ("liver",                 "soft",    "#7A261A", [0.48, 0.15, 0.10], 0.06, None),
    # aorta — thin smooth vessel wall
    ("aorta",                 "surface", "#E01F1A", [0.88, 0.12, 0.10], 0.06, None),
]


# ── Camera ────────────────────────────────────────────────────────────────────

def compute_camera(shape, zooms):
    """3/4 front-right-top view covering the full CT volume."""
    nx, ny, nz = shape
    sx, sy, sz = zooms
    cx, cy, cz = nx*sx/2, ny*sy/2, nz*sz/2
    radius = max(nx*sx, ny*sy, nz*sz) * 0.9
    pos = [cx + radius*0.6, cy - radius*1.1, cz + radius*0.5]
    fpt = [cx, cy, cz]
    return {
        "focal_point":    fpt,
        "position":       pos,
        "up":             [0.0, 0.0, 1.0],
        "fov_deg":        30.0,
        "focus_distance": float(np.linalg.norm(np.array(pos) - np.array(fpt))),
    }


def camera_at_angles(base_cam, thetas_deg):
    """Rotate camera around the Z-axis through the focal point."""
    cameras = []
    fx, fy, fz = base_cam["focal_point"]
    px, py, pz = base_cam["position"]
    dx0, dy0 = px - fx, py - fy
    for theta in thetas_deg:
        t = math.radians(theta)
        new_dx = dx0*math.cos(t) - dy0*math.sin(t)
        new_dy = dx0*math.sin(t) + dy0*math.cos(t)
        new_pos = [fx + new_dx, fy + new_dy, pz]
        cam = dict(base_cam)
        cam["position"]       = new_pos
        cam["focus_distance"] = float(np.linalg.norm(
            np.array(new_pos) - np.array(base_cam["focal_point"])))
        cam["angle_label"] = f"{theta:+.0f}°"
        cameras.append(cam)
    return cameras


# ── Mesh extraction ───────────────────────────────────────────────────────────

def extract_mesh(seg_path: Path, zooms, min_voxels=500):
    """
    NIfTI binary mask → PyVista surface mesh.
    Crops to tight bbox before marching cubes (much faster).
    Returns None if mask is empty or tiny.
    """
    data = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
    nz   = np.where(data > 0)
    if len(nz[0]) < min_voxels:
        return None

    pad = 2
    slices = tuple(
        slice(max(0, int(nz[i].min()) - pad), min(data.shape[i], int(nz[i].max()) + pad + 1))
        for i in range(3)
    )
    data_crop = data[slices]
    origin    = tuple(slices[i].start * float(zooms[i]) for i in range(3))

    grid = pv.ImageData()
    grid.dimensions = np.array(data_crop.shape) + 1
    grid.spacing    = [float(z) for z in zooms]
    grid.origin     = origin
    grid.cell_data["val"] = data_crop.flatten(order="F")

    mesh = (grid.cell_data_to_point_data()
                .contour(isosurfaces=[0.5], scalars="val"))
    if mesh.n_points == 0:
        return None

    mesh = (mesh.connectivity(extraction_mode="largest")
                .smooth(n_iter=30, relaxation_factor=0.05))
    return mesh


# ── OBJ export ────────────────────────────────────────────────────────────────

def save_obj(mesh: pv.PolyData, path: Path):
    """Triangulated OBJ with per-vertex normals."""
    m = mesh.triangulate().compute_normals(cell_normals=False, point_normals=True)
    verts   = m.points
    normals = m.point_normals
    faces   = m.faces.reshape(-1, 4)[:, 1:]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for tri in faces:
            a, b, c = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


# ── ACES tonemapping ──────────────────────────────────────────────────────────

def aces_filmic(hdr: np.ndarray, exposure: float = 1.2) -> np.ndarray:
    x = np.maximum(hdr, 0) * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)
    return (ldr ** (1/2.2) * 255).astype(np.uint8)


# ── Simple render (PyVista) ───────────────────────────────────────────────────

def render_simple(meshes_with_colors, cam, size: int = 512):
    """Multi-tissue Phong shading — fast conditioning image."""
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    for mesh, hex_color in meshes_with_colors:
        pl.add_mesh(mesh, color=hex_color,
                    ambient=0.08, diffuse=0.78, specular=0.28,
                    specular_power=40, smooth_shading=True)
    pl.add_light(pv.Light(position=( 1.0, -1.0,  1.5), intensity=0.9))
    pl.add_light(pv.Light(position=(-0.5,  1.0,  0.5), intensity=0.3))
    pl.camera.focal_point = cam["focal_point"]
    pl.camera.position    = cam["position"]
    pl.camera.up          = cam["up"]
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── GT render (Mitsuba path tracer + rim light SSS approximation) ────────────

def render_mitsuba_gt(tissue_objs, cam, aabb, spp: int, size: int = 512,
                      aperture_radius: float = 6.0):
    """
    Path-traced GT.
      - Bone    : roughplastic GGX IOR=1.55, alpha=0.25
      - Soft    : roughplastic GGX IOR=1.40, alpha=0.06 (very wet/glossy organs)
      - Surface : roughplastic GGX IOR=1.40, alpha=0.06 (vessels)
      - Lights  : warm key + cool fill + warm rim from behind (SSS-like edge glow)
                  + dim cool ambient
      - DoF     : thinlens camera
    """
    cx = (aabb[0]+aabb[3]) / 2
    cy = (aabb[1]+aabb[4]) / 2
    cz = (aabb[2]+aabb[5]) / 2
    scale = max(aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2])

    kp = [cx + scale*0.73, cy - scale*0.54, cz + scale*1.28]   # warm key
    fp = [cx - scale*0.59, cy + scale*1.52, cz + scale*1.48]   # cool fill
    rp = [cx - scale*0.40, cy + scale*0.70, cz - scale*0.50]   # warm rim (behind+below)
    sk = [cx, cy, cz]

    CAM_TF = mi.ScalarTransform4f.look_at(
        origin=cam["position"], target=cam["focal_point"], up=cam["up"])

    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 48},
        "sensor": {
            "type":            "thinlens",
            "fov":             cam["fov_deg"],
            "fov_axis":        "y",
            "to_world":        CAM_TF,
            "aperture_radius": aperture_radius,
            "focus_distance":  cam["focus_distance"],
            "film": {
                "type":         "hdrfilm",
                "width":        size,
                "height":       size,
                "rfilter":      {"type": "tent"},
                "pixel_format": "rgb",
            },
            "sampler": {"type": "multijitter", "sample_count": spp},
        },
        "key_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(kp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([scale*0.18, scale*0.18, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [82.0, 68.9, 55.8]}},
        },
        "fill_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(fp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([scale*0.25, scale*0.25, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [2.46, 3.44, 6.89]}},
        },
        # warm rim light from behind — backlit organs glow red-orange at edges,
        # visually approximating subsurface scattering
        "rim_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(rp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([scale*0.35, scale*0.35, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [18.0, 6.0, 4.0]}},
        },
        "ambient": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.03, 0.03, 0.06]},
        },
    }

    for seg_name, method, gt_rgb, p5, p6, obj_path in tissue_objs:
        ior   = 1.55 if method == "bone" else 1.40
        alpha = float(p5)
        scene[seg_name] = {
            "type": "obj",
            "filename": str(obj_path),
            "bsdf": {
                "type":                "roughplastic",
                "distribution":        "ggx",
                "alpha":               alpha,
                "int_ior":             ior,
                "diffuse_reflectance": {"type": "rgb", "value": gt_rgb},
            },
        }

    s = mi.load_dict(scene)
    t0 = time.time()
    img = mi.render(s, spp=spp)
    print(f"  Rendered in {time.time()-t0:.1f}s")
    return np.array(img)


# ── Main ──────────────────────────────────────────────────────────────────────

def bbox(arr):
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, c0, r1, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="s0329")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--spp",      type=int,   default=512)
    ap.add_argument("--size",     type=int,   default=512)
    ap.add_argument("--angles",   type=int,   default=3,
                    help="Camera angles to render (default 3: -40°, 0°, +40°)")
    ap.add_argument("--aperture", type=float, default=6.0,
                    help="DoF aperture radius in mm (0 = no DoF)")
    args = ap.parse_args()

    subj_dir = Path(args.dataset) / args.subject
    seg_dir  = subj_dir / "segmentations"
    ct_path  = subj_dir / "ct.nii.gz"

    out_dir  = Path("data/renders/totalseg") / args.subject
    mesh_dir = out_dir / "meshes"
    pair_out = Path("results/totalseg_pairs")
    for d in [out_dir, mesh_dir, pair_out]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Subject  : {args.subject}")
    print(f"SPP      : {args.spp}   Size: {args.size}×{args.size}   "
          f"Angles: {args.angles}   Aperture: {args.aperture} mm")
    print(f"{'='*60}")

    ct_img   = nib.load(str(ct_path))
    shape    = ct_img.shape[:3]
    zooms    = ct_img.header.get_zooms()[:3]
    base_cam = compute_camera(shape, zooms)
    aabb     = [0.0, 0.0, 0.0,
                float(shape[0]*zooms[0]), float(shape[1]*zooms[1]), float(shape[2]*zooms[2])]

    print(f"\n  CT       : {shape}  spacing {[round(float(z),2) for z in zooms]} mm")
    print(f"  Camera   : pos {[round(x,1) for x in base_cam['position']]}  "
          f"fpt {[round(x,1) for x in base_cam['focal_point']]}")
    print(f"  Focus d  : {base_cam['focus_distance']:.1f} mm")

    if args.angles == 1:
        angle_offsets = [0]
    elif args.angles == 2:
        angle_offsets = [-30, 30]
    else:
        half = (args.angles - 1) / 2
        step = 40 / half if half > 0 else 0
        angle_offsets = [round(-40 + i*step) for i in range(args.angles)]
    cameras = camera_at_angles(base_cam, angle_offsets)
    print(f"  Angles   : {angle_offsets}°")

    # ── Extract meshes (once, reused across all angles) ───────────────────────
    print("\n[1/3] Extracting tissue meshes ...")
    meshes_simple = []
    tissue_objs   = []

    for seg_name, method, simple_hex, gt_rgb, p5, p6 in TISSUES:
        seg_path = seg_dir / f"{seg_name}.nii.gz"
        if not seg_path.exists():
            print(f"  skip  {seg_name} (not in dataset)")
            continue

        t0   = time.time()
        mesh = extract_mesh(seg_path, zooms)
        if mesh is None:
            print(f"  skip  {seg_name} (empty)")
            continue

        obj_path = mesh_dir / f"{seg_name}.obj"
        if not obj_path.exists():
            save_obj(mesh, obj_path)

        meshes_simple.append((mesh, simple_hex))
        tissue_objs.append((seg_name, method, gt_rgb, p5, p6, obj_path))
        print(f"  OK    {seg_name:<30} {mesh.n_points:>8,} pts  ({method})  {time.time()-t0:.1f}s")

    if not meshes_simple:
        print("ERROR: no meshes loaded — check --dataset and --subject.")
        return
    print(f"\n  Loaded {len(meshes_simple)} tissues")

    # ── Render each angle ─────────────────────────────────────────────────────
    angle_rows = []

    for i, cam in enumerate(cameras):
        label = cam.get("angle_label", f"{i}")
        print(f"\n--- Angle {label} ({i+1}/{len(cameras)}) ---")

        print("  [2/3] Simple render (PyVista) ...")
        t0 = time.time()
        simple_img = render_simple(meshes_simple, cam, args.size)
        if simple_img.ndim == 3 and simple_img.shape[2] == 4:
            simple_img = simple_img[:, :, :3]
        print(f"    Done {time.time()-t0:.1f}s  std={simple_img.std():.1f}")
        Image.fromarray(simple_img).save(str(out_dir / f"simple_a{label}.png"))

        print(f"  [3/3] Mitsuba GT (volpath {args.spp} SPP, DoF {args.aperture}mm) ...")
        hdr    = render_mitsuba_gt(tissue_objs, cam, aabb, args.spp, args.size,
                                   aperture_radius=args.aperture)
        gt_img = aces_filmic(hdr)
        print(f"    GT std={gt_img.std():.1f}  max={gt_img.max()}")
        Image.fromarray(gt_img).save(str(out_dir / f"gt_spp{args.spp}_a{label}.png"))

        try:
            sb = bbox(np.array(Image.fromarray(simple_img).convert("L")))
            gb = bbox(np.array(Image.fromarray(gt_img).convert("L")))
            inter = (max(0, min(sb[2], gb[2]) - max(sb[0], gb[0])) *
                     max(0, min(sb[3], gb[3]) - max(sb[1], gb[1])))
            union = ((sb[2]-sb[0])*(sb[3]-sb[1]) +
                     (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
            print(f"    BBox IoU: {inter/union:.3f}")
        except (IndexError, ZeroDivisionError):
            pass

        angle_rows.append((simple_img, gt_img, label))

    # ── Assemble grid ─────────────────────────────────────────────────────────
    gap     = 10
    label_h = 32
    row_h   = args.size + label_h
    canvas  = Image.new("RGB", (args.size*2 + gap, row_h*len(angle_rows) + gap), (12, 12, 12))
    d       = ImageDraw.Draw(canvas)

    for row_i, (simple_img, gt_img, label) in enumerate(angle_rows):
        y0 = row_i * row_h + gap
        d.text((5, y0 + 4),
               f"Simple — {args.subject}  view {label}",
               fill=(160, 160, 160))
        d.text((args.size + gap + 5, y0 + 4),
               f"GT — volpath {args.spp} SPP  SSS  DoF {args.aperture}mm  view {label}",
               fill=(160, 160, 160))
        canvas.paste(Image.fromarray(simple_img), (0,               y0 + label_h))
        canvas.paste(Image.fromarray(gt_img),     (args.size + gap, y0 + label_h))

    grid_path = pair_out / f"{args.subject}_grid_spp{args.spp}.png"
    canvas.save(str(grid_path))
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
