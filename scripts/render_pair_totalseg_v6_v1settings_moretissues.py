"""
render_pair_totalseg_v6_v1settings_moretissues.py

v1 settings preserved exactly:
  - roughplastic BSDF for all tissues (what v1 did and looked best)
  - Same two-light setup: warm key [82.0, 68.9, 55.8] + cool fill [2.46, 3.44, 6.89]
  - Same perspective camera (no DoF)
  - Same independent sampler
  - Same ACES exposure=1.5
  - Same path integrator max_depth=48

Additions over v1:
  - More tissues: heart, pancreas, IVC, portal vein, colon, small bowel,
    urinary bladder, esophagus — fills the scene visually
  - Multi-angle grid output (default 3 angles at -40°, 0°, +40°)

Usage:
    python scripts/render_pair_totalseg_v6_v1settings_moretissues.py \\
        --subject s0050 --spp 128 --angles 3
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
# (seg_name, bsdf_type, simple_hex, gt_rgb, roughness_alpha)
# bsdf_type "bone" → int_ior=1.55, "soft" → int_ior=1.40, "vessel" → int_ior=1.38

TISSUES = [
    # Posterior muscles
    ("autochthon_left",               "soft",   "#C05A28", [0.55, 0.24, 0.18], 0.45),
    ("autochthon_right",              "soft",   "#C05A28", [0.55, 0.24, 0.18], 0.45),
    # Lungs
    ("lung_lower_lobe_left",          "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_lower_lobe_right",         "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_upper_lobe_left",          "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_upper_lobe_right",         "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    # Vertebrae — v5's improved bone: alpha=0.22, slightly cooler cream
    ("vertebrae_T12",                 "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    ("vertebrae_L1",                  "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    ("vertebrae_L2",                  "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    ("vertebrae_L3",                  "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    ("vertebrae_L4",                  "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    ("vertebrae_L5",                  "bone",   "#EAE0CC", [0.90, 0.85, 0.77], 0.22),
    # Heart — dark myocardium red
    ("heart",                         "soft",   "#B03030", [0.52, 0.14, 0.12], 0.32),
    # Esophagus
    ("esophagus",                     "soft",   "#C87878", [0.60, 0.32, 0.30], 0.40),
    # Abdominal organs (same as v1)
    ("stomach",                       "soft",   "#E8C866", [0.70, 0.62, 0.56], 0.50),
    ("gallbladder",                   "soft",   "#60A840", [0.55, 0.72, 0.30], 0.40),
    ("spleen",                        "soft",   "#7B3A8C", [0.38, 0.10, 0.16], 0.35),
    ("kidney_right",                  "soft",   "#2A5EA8", [0.52, 0.26, 0.19], 0.35),
    ("kidney_left",                   "soft",   "#2A6EB8", [0.52, 0.26, 0.19], 0.35),
    ("liver",                         "soft",   "#9E3028", [0.48, 0.15, 0.10], 0.35),
    # Pancreas — pale yellowish-tan
    ("pancreas",                      "soft",   "#E8B870", [0.78, 0.62, 0.38], 0.42),
    # Bowel
    ("duodenum",                      "soft",   "#D4A080", [0.68, 0.50, 0.38], 0.48),
    ("small_bowel",                   "soft",   "#D4A080", [0.68, 0.50, 0.38], 0.50),
    ("colon",                         "soft",   "#C88060", [0.64, 0.44, 0.34], 0.48),
    # Urinary bladder — pale blue
    ("urinary_bladder",               "soft",   "#7888D4", [0.38, 0.44, 0.72], 0.38),
    # Vascular — aorta red, veins blue
    ("aorta",                         "vessel", "#DD1818", [0.88, 0.12, 0.10], 0.18),
    ("inferior_vena_cava",            "vessel", "#1840A0", [0.12, 0.28, 0.72], 0.18),
    ("portal_vein_and_splenic_vein",  "vessel", "#2050B8", [0.14, 0.30, 0.70], 0.20),
    ("superior_vena_cava",            "vessel", "#1840A0", [0.12, 0.28, 0.72], 0.18),
]


# ── Camera ────────────────────────────────────────────────────────────────────

def compute_camera(shape, zooms):
    nx, ny, nz = shape
    sx, sy, sz = zooms
    cx, cy, cz = nx*sx/2, ny*sy/2, nz*sz/2
    radius = max(nx*sx, ny*sy, nz*sz) * 0.9
    return {
        "focal_point": [cx, cy, cz],
        "position":    [cx + radius*0.6, cy - radius*1.1, cz + radius*0.5],
        "up":          [0.0, 0.0, 1.0],
        "fov_deg":     30.0,
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
        cam = dict(base_cam)
        cam["position"]    = [fx + new_dx, fy + new_dy, pz]
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


def aces_filmic(hdr, exposure=1.5):
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
                    ambient=0.20, diffuse=0.75, specular=0.10,
                    specular_power=10, smooth_shading=True)
    pl.add_light(pv.Light(position=( 1.0, -1.0,  1.5), intensity=0.9))
    pl.add_light(pv.Light(position=(-0.5,  1.0,  0.5), intensity=0.3))
    pl.camera.focal_point = cam["focal_point"]
    pl.camera.position    = cam["position"]
    pl.camera.up          = cam["up"]
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── GT render: v1 settings exactly ───────────────────────────────────────────

def render_mitsuba_gt(tissue_objs, cam, aabb, spp, size=512):
    cx = (aabb[0]+aabb[3]) / 2
    cy = (aabb[1]+aabb[4]) / 2
    cz = (aabb[2]+aabb[5]) / 2
    scale = max(aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2])
    sk = [cx, cy, cz]
    kp = [cx + scale*0.73, cy - scale*0.54, cz + scale*1.28]
    fp = [cx - scale*0.59, cy + scale*1.52, cz + scale*1.48]

    CAM_TF = mi.ScalarTransform4f.look_at(
        origin=cam["position"], target=cam["focal_point"], up=cam["up"])

    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 48},
        "sensor": {
            "type":     "perspective",
            "fov":      cam["fov_deg"],
            "fov_axis": "y",
            "to_world": CAM_TF,
            "film": {
                "type":         "hdrfilm",
                "width":        size,
                "height":       size,
                "rfilter":      {"type": "tent"},
                "pixel_format": "rgb",
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },
        # v1 lights — warm key, cool fill — unchanged
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
    }

    for seg_name, bsdf_type, gt_rgb, alpha, obj_path in tissue_objs:
        ior = 1.55 if bsdf_type == "bone" else (1.38 if bsdf_type == "vessel" else 1.40)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox(arr):
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    if not rows.any() or not cols.any():
        raise IndexError
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, c0, r1, c1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="s0050")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--spp",      type=int, default=128)
    ap.add_argument("--size",     type=int, default=512)
    ap.add_argument("--angles",   type=int, default=3)
    ap.add_argument("--exposure", type=float, default=1.5)
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
    print(f"[v6]  subject={args.subject}  spp={args.spp}  angles={args.angles}")
    print(f"v1 settings + more tissues + multi-angle")
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

    for seg_name, bsdf_type, simple_hex, gt_rgb, alpha in TISSUES:
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
        tissue_objs.append((seg_name, bsdf_type, gt_rgb, alpha, obj_path))
        print(f"  {seg_name}")

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
        Image.fromarray(simple_img).save(str(out_dir / f"simple_v6_{label}.png"))

        hdr    = render_mitsuba_gt(tissue_objs, cam, aabb, args.spp, args.size)
        gt_img = aces_filmic(hdr, exposure=args.exposure)
        print(f"  GT   : std={gt_img.std():.1f}  max={gt_img.max()}")
        Image.fromarray(gt_img).save(str(out_dir / f"gt_v6_spp{args.spp}_{label}.png"))

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
               f"GT v6 — roughplastic path {args.spp}SPP ACES×{args.exposure} {label}",
               fill=(160, 160, 160))
        canvas.paste(Image.fromarray(simple_img), (0,             y0+label_h))
        canvas.paste(Image.fromarray(gt_img),     (args.size+gap, y0+label_h))

    grid_path = pair_out / f"{args.subject}_v6_spp{args.spp}.png"
    canvas.save(str(grid_path))
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
