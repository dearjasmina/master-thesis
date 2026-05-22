"""
render_pair_totalseg_v5_roughplastic_clearcoat.py

Lesson from v4: roughdielectric + homogeneous SSS gives dark images for opaque
organs — 96% of light enters the absorbing medium and doesn't escape.

roughplastic implicitly handles the diffuse SSS component (Lambertian lobe =
multiple-scatter approximation) and empirically matched user preference (v1).

v5 keeps roughplastic but adds:
  - Anatomically accurate tissue colors (not atlas label colors)
  - `blendbsdf` for visceral organs: 92% roughplastic + 8% roughdielectric →
    wet-sheen specular from thin Glisson capsule without going dark
  - Neutral-warm key light, cool fill, NO rim (rim caused orange bone shadows)
  - Multi-angle 3-row grid output
  - thinlens DoF, path integrator, 512 SPP, ACES exposure=1.0

Usage:
    python scripts/render_pair_totalseg_v5_roughplastic_clearcoat.py \\
        --subject s0329 --spp 512 --angles 3
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
# (seg_name, method, simple_hex, gt_rgb, alpha)
#
# method "bone"    → roughplastic IOR=1.55, alpha=0.25
# method "plastic" → roughplastic IOR=1.40, tuned alpha (vessels, aorta)
# method "capsule" → blendbsdf: 92% roughplastic + 8% roughdielectric (wet sheen)
#                     used for visceral organs that have a thin specular capsule
#
# gt_rgb: diffuse_reflectance for roughplastic base layer.
# Colors sourced from surgical endoscopy imagery (not atlas/label palette).

TISSUES = [
    # seg_name                  method     simple_hex   gt_rgb (diffuse)        alpha
    # Posterior muscles — dark brownish-red
    ("autochthon_left",        "plastic",  "#7A3020",  [0.48, 0.20, 0.14],     0.42),
    ("autochthon_right",       "plastic",  "#7A3020",  [0.48, 0.20, 0.14],     0.42),
    # Lungs — pale pink-gray (air-filled, very diffuse)
    ("lung_lower_lobe_left",   "plastic",  "#C2B4AF",  [0.74, 0.70, 0.68],     0.52),
    ("lung_lower_lobe_right",  "plastic",  "#C2B4AF",  [0.74, 0.70, 0.68],     0.52),
    ("lung_upper_lobe_left",   "plastic",  "#C2B4AF",  [0.74, 0.70, 0.68],     0.52),
    ("lung_upper_lobe_right",  "plastic",  "#C2B4AF",  [0.74, 0.70, 0.68],     0.52),
    # Vertebrae — dry cortical bone, tighter specular
    ("vertebrae_T12",          "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    ("vertebrae_L1",           "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    ("vertebrae_L2",           "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    ("vertebrae_L3",           "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    ("vertebrae_L4",           "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    ("vertebrae_L5",           "bone",     "#EAE0CC",  [0.90, 0.85, 0.77],     0.22),
    # Stomach — pale tan-pink mucosa
    ("stomach",                "capsule",  "#A88A7A",  [0.62, 0.50, 0.43],     0.30),
    # Gallbladder — bile green, thin specular wall
    ("gallbladder",            "capsule",  "#7DB54A",  [0.50, 0.68, 0.26],     0.12),
    # Spleen — deep purple-red, Glisson capsule (wet sheen)
    ("spleen",                 "capsule",  "#5E1824",  [0.34, 0.09, 0.14],     0.20),
    # Kidneys — medium brownish-red, perirenal capsule
    ("kidney_right",           "capsule",  "#7E3A22",  [0.48, 0.20, 0.14],     0.22),
    ("kidney_left",            "capsule",  "#7E3A22",  [0.48, 0.20, 0.14],     0.22),
    # Liver — dark burgundy-brown (darkest organ), Glisson capsule
    ("liver",                  "capsule",  "#6E2018",  [0.42, 0.12, 0.09],     0.20),
    # Aorta — bright red, slightly shiny vessel wall
    ("aorta",                  "plastic",  "#D81C18",  [0.85, 0.10, 0.08],     0.12),
]


# ── Camera ────────────────────────────────────────────────────────────────────

def compute_camera(shape, zooms):
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

def extract_mesh(seg_path, zooms, min_voxels=500):
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
    return (mesh.connectivity(extraction_mode="largest")
                .smooth(n_iter=30, relaxation_factor=0.05))


def save_obj(mesh, path):
    m = mesh.triangulate().compute_normals(cell_normals=False, point_normals=True)
    verts, normals = m.points, m.point_normals
    faces = m.faces.reshape(-1, 4)[:, 1:]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for tri in faces:
            a, b, c = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def aces_filmic(hdr, exposure=1.0):
    x = np.maximum(hdr, 0) * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    ldr = np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0)
    return (ldr ** (1/2.2) * 255).astype(np.uint8)


# ── Simple render ─────────────────────────────────────────────────────────────

def render_simple(meshes_with_colors, cam, size=512):
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    for mesh, hex_color in meshes_with_colors:
        pl.add_mesh(mesh, color=hex_color,
                    ambient=0.10, diffuse=0.78, specular=0.20,
                    specular_power=30, smooth_shading=True)
    pl.add_light(pv.Light(position=( 1.0, -1.0,  1.5), intensity=0.9))
    pl.add_light(pv.Light(position=(-0.5,  1.0,  0.5), intensity=0.3))
    pl.camera.focal_point = cam["focal_point"]
    pl.camera.position    = cam["position"]
    pl.camera.up          = cam["up"]
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── GT render ─────────────────────────────────────────────────────────────────

def render_mitsuba_gt(tissue_objs, cam, aabb, spp, size=512, aperture_radius=6.0):
    """
    BSDF strategy:
      bone    → roughplastic IOR=1.55 (dry cortical)
      plastic → roughplastic IOR=1.40 (muscles, lungs, vessels)
      capsule → blendbsdf: 92% roughplastic + 8% roughdielectric (wet sheen)
                Glisson capsule on liver, spleen, kidneys, gallbladder, stomach.
                Only 8% dielectric — avoids the "dark organ" problem of v4.

    Lights: large neutral-warm key + large cool fill, NO rim light.
    Integrator: path (no interior media → path is sufficient).
    DoF: thinlens with 6mm aperture.
    Tonemap: ACES exposure=1.0.
    """
    cx = (aabb[0]+aabb[3]) / 2
    cy = (aabb[1]+aabb[4]) / 2
    cz = (aabb[2]+aabb[5]) / 2
    sc = max(aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2])

    kp  = [cx + sc*0.73, cy - sc*0.54, cz + sc*1.28]
    fp  = [cx - sc*0.59, cy + sc*1.52, cz + sc*1.48]
    rim = [cx + sc*0.10, cy + sc*0.20, cz - sc*0.80]
    sk  = [cx, cy, cz]

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
        # Key: large neutral-warm (not orange) — avoids orange bone shadows
        "key_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(kp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([sc*0.28, sc*0.28, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [72.0, 66.0, 58.0]}},
        },
        # Fill: large cool blue, softens opposite shadows
        "fill_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(fp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([sc*0.32, sc*0.32, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [3.0, 4.0, 7.5]}},
        },
        # Subtle under-rim from below to lift shadow detail
        "under_rim": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(rim, sk, [0, 1, 0]) @
                         mi.ScalarTransform4f.scale([sc*0.20, sc*0.20, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [1.0, 1.5, 2.5]}},
        },
    }

    for seg_name, method, gt_rgb, alpha, obj_path in tissue_objs:
        base_bsdf = {
            "type":                "roughplastic",
            "distribution":        "ggx",
            "alpha":               alpha,
            "int_ior":             1.55 if method == "bone" else 1.40,
            "diffuse_reflectance": {"type": "rgb", "value": gt_rgb},
        }

        if method == "capsule":
            # 8% roughdielectric on top of roughplastic: wet-sheen specular
            # without the opacity/darkness issue of pure roughdielectric
            bsdf = {
                "type":   "blendbsdf",
                "weight": 0.08,
                "bsdf_0": base_bsdf,
                "bsdf_1": {
                    "type":         "roughdielectric",
                    "distribution": "ggx",
                    "alpha":        max(0.04, alpha * 0.4),
                    "int_ior":      1.36,
                    "ext_ior":      1.00,
                },
            }
        else:
            bsdf = base_bsdf

        scene[seg_name] = {
            "type":     "obj",
            "filename": str(obj_path),
            "bsdf":     bsdf,
        }

    s = mi.load_dict(scene)
    t0 = time.time()
    img = mi.render(s, spp=spp)
    print(f"  Rendered in {time.time()-t0:.1f}s")
    return np.array(img)


# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox(arr):
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    if not rows.any() or not cols.any():
        raise IndexError("no foreground pixels")
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, c0, r1, c1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="s0329")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--spp",      type=int,   default=512)
    ap.add_argument("--size",     type=int,   default=512)
    ap.add_argument("--angles",   type=int,   default=3)
    ap.add_argument("--aperture", type=float, default=6.0)
    ap.add_argument("--exposure", type=float, default=1.0)
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
    print(f"[v5-clearcoat]  subject={args.subject}  spp={args.spp}")
    print(f"BSDF: roughplastic + blendbsdf capsule  path  DoF  exp={args.exposure}")
    print(f"{'='*60}")

    ct_img   = nib.load(str(ct_path))
    shape    = ct_img.shape[:3]
    zooms    = ct_img.header.get_zooms()[:3]
    base_cam = compute_camera(shape, zooms)
    aabb     = [0.0, 0.0, 0.0,
                float(shape[0]*zooms[0]), float(shape[1]*zooms[1]), float(shape[2]*zooms[2])]

    if args.angles == 1:
        angle_offsets = [0]
    elif args.angles == 2:
        angle_offsets = [-30, 30]
    else:
        half = (args.angles - 1) / 2
        step = 40 / half if half > 0 else 0
        angle_offsets = [round(-40 + i*step) for i in range(args.angles)]
    cameras = camera_at_angles(base_cam, angle_offsets)

    print("\n[1/3] Extracting meshes ...")
    meshes_simple, tissue_objs = [], []

    for seg_name, method, simple_hex, gt_rgb, alpha in TISSUES:
        seg_path = seg_dir / f"{seg_name}.nii.gz"
        if not seg_path.exists():
            continue
        mesh = extract_mesh(seg_path, zooms)
        if mesh is None:
            continue
        obj_path = mesh_dir / f"{seg_name}.obj"
        if not obj_path.exists():
            save_obj(mesh, obj_path)
        meshes_simple.append((mesh, simple_hex))
        tissue_objs.append((seg_name, method, gt_rgb, alpha, obj_path))
        print(f"  {seg_name:<30} {method}  α={alpha}")

    print(f"\n  {len(meshes_simple)} tissues loaded")

    angle_rows = []
    for cam in cameras:
        label = cam.get("angle_label", "0°")
        print(f"\n--- Angle {label} ---")

        t0 = time.time()
        simple_img = render_simple(meshes_simple, cam, args.size)
        if simple_img.ndim == 3 and simple_img.shape[2] == 4:
            simple_img = simple_img[:, :, :3]
        print(f"  Simple: {time.time()-t0:.1f}s  std={simple_img.std():.1f}")
        Image.fromarray(simple_img).save(str(out_dir / f"simple_v5_{label}.png"))

        hdr    = render_mitsuba_gt(tissue_objs, cam, aabb, args.spp, args.size,
                                   aperture_radius=args.aperture)
        gt_img = aces_filmic(hdr, exposure=args.exposure)
        print(f"  GT   : std={gt_img.std():.1f}  max={gt_img.max()}")
        Image.fromarray(gt_img).save(str(out_dir / f"gt_v5_spp{args.spp}_{label}.png"))

        try:
            sb = bbox(np.array(Image.fromarray(simple_img).convert("L")))
            gb = bbox(np.array(Image.fromarray(gt_img).convert("L")))
            inter = (max(0, min(sb[2], gb[2]) - max(sb[0], gb[0])) *
                     max(0, min(sb[3], gb[3]) - max(sb[1], gb[1])))
            union = ((sb[2]-sb[0])*(sb[3]-sb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
            print(f"  IoU  : {inter/union:.3f}")
        except (IndexError, ZeroDivisionError):
            pass

        angle_rows.append((simple_img, gt_img, label))

    gap, label_h = 10, 32
    row_h  = args.size + label_h
    canvas = Image.new("RGB", (args.size*2+gap, row_h*len(angle_rows)+gap), (12, 12, 12))
    d = ImageDraw.Draw(canvas)
    for row_i, (simple_img, gt_img, label) in enumerate(angle_rows):
        y0 = row_i * row_h + gap
        d.text((5, y0+4), f"Simple — {args.subject} {label}", fill=(160, 160, 160))
        d.text((args.size+gap+5, y0+4),
               f"GT v5 — roughplastic+capsule path {args.spp}SPP {label}",
               fill=(160, 160, 160))
        canvas.paste(Image.fromarray(simple_img), (0,             y0+label_h))
        canvas.paste(Image.fromarray(gt_img),     (args.size+gap, y0+label_h))

    grid_path = pair_out / f"{args.subject}_v5_clearcoat_spp{args.spp}.png"
    canvas.save(str(grid_path))
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
