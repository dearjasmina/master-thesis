"""
render_pair_totalseg_v4_sss_scaled.py — research-backed SSS approach

Key insight from research (mitsuba2 GitHub #540, EndoPBR arXiv:2502.20669):
  - `principled` BSDF in Mitsuba 3 has NO SSS support
  - Physical sigma_t (~32 mm⁻¹) → MFP 30µm → path tracer gives black organs
  - Fix: use `homogeneous` medium with `scale` parameter (scale=0.05–0.10)
    → effective sigma_t drops to ~2 mm⁻¹ → MFP ~5mm → path-traceable
  - Use SPECTRAL sigma_t (RGB): R>G>B for red organs (differential scatter gives colour)
  - Surface BSDF: `roughdielectric` (not roughplastic, not null)
    → photons partially reflect off wet surface AND partially refract into medium
  - From EndoPBR: real organ roughness ~0.30–0.50 GGX; thin specular capsule IOR~1.36–1.40
  - Lights: larger area emitters (scale*0.30 key vs *0.18 before) to avoid billiard-ball specular
  - Integrator: `volpath` (required for interior media)

Saved states:
  v1_original             → roughplastic, 2 lights, ACES×1.5
  v2_roughplastic_rimlight→ roughplastic, rim light, DoF, principled attempt
  v4_sss_scaled (this)    → roughdielectric + homogeneous medium (scale), spectral SSS

Usage:
    python scripts/render_pair_totalseg_v4_sss_scaled.py \\
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
# (seg_name, method, simple_hex, gt_rgb, roughness, med_scale, sigma_t_rgb, albedo_rgb, g)
#
# method "bone"   → roughplastic IOR=1.55, no interior medium
# method "sss"    → roughdielectric IOR=1.36-1.40 + homogeneous interior (scaled SSS)
#
# sigma_t_rgb: spectral extinction [R, G, B].
#   R>G>B  → organ appears red/warm  (more red scatter reaches camera)
#   G>R>B  → organ appears green/yellow (gallbladder bile)
#   Values in physical units; med_scale brings MFP to ~3-10mm for path tracing.
#
# albedo_rgb: single-scatter albedo per channel [0-1].
#   Controls how much of scattered light survives each event.
#   High albedo → bright, diffuse glow. Low → dark, absorbing.
#
# Sources: Jacques 2013 (PMB), EndoPBR (arXiv:2502.20669), mitsuba2 issue #540

TISSUES = [
    # seg_name              method  simple_hex   gt_rgb(display)   rough  scale  σt_rgb               albedo_rgb          g
    # muscles
    ("autochthon_left",     "sss",  "#8C3D2E",  [0.55,0.24,0.18],  0.45, 0.07, [1.5, 0.9, 0.5],  [0.82, 0.55, 0.35], 0.93),
    ("autochthon_right",    "sss",  "#8C3D2E",  [0.55,0.24,0.18],  0.45, 0.07, [1.5, 0.9, 0.5],  [0.82, 0.55, 0.35], 0.93),
    # lungs: mostly air, low sigma_t, near-white scatter
    ("lung_lower_lobe_left",  "sss","#C7BFBD",  [0.78,0.75,0.74],  0.50, 0.05, [0.8, 0.8, 0.8],  [0.95, 0.93, 0.92], 0.90),
    ("lung_lower_lobe_right", "sss","#C7BFBD",  [0.78,0.75,0.74],  0.50, 0.05, [0.8, 0.8, 0.8],  [0.95, 0.93, 0.92], 0.90),
    ("lung_upper_lobe_left",  "sss","#C7BFBD",  [0.78,0.75,0.74],  0.50, 0.05, [0.8, 0.8, 0.8],  [0.95, 0.93, 0.92], 0.90),
    ("lung_upper_lobe_right", "sss","#C7BFBD",  [0.78,0.75,0.74],  0.50, 0.05, [0.8, 0.8, 0.8],  [0.95, 0.93, 0.92], 0.90),
    # vertebrae: dry cortical bone, no SSS
    ("vertebrae_T12",       "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    ("vertebrae_L1",        "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    ("vertebrae_L2",        "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    ("vertebrae_L3",        "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    ("vertebrae_L4",        "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    ("vertebrae_L5",        "bone", "#EBE0CC",  [0.92,0.88,0.80],  0.28, None, None, None, None),
    # abdominal organs: roughdielectric wet capsule + scaled homogeneous SSS medium
    ("stomach",             "sss",  "#B39E8F",  [0.70,0.62,0.56],  0.25, 0.07, [1.5, 1.2, 0.8],  [0.80, 0.70, 0.55], 0.90),
    ("gallbladder",         "sss",  "#8CB84D",  [0.55,0.72,0.30],  0.10, 0.05, [0.8, 1.2, 0.4],  [0.75, 0.85, 0.55], 0.85),
    ("spleen",              "sss",  "#611A29",  [0.38,0.10,0.16],  0.18, 0.07, [2.0, 1.5, 1.0],  [0.82, 0.60, 0.55], 0.92),
    ("kidney_right",        "sss",  "#854230",  [0.52,0.26,0.19],  0.20, 0.07, [1.8, 1.4, 0.9],  [0.78, 0.62, 0.40], 0.92),
    ("kidney_left",         "sss",  "#854230",  [0.52,0.26,0.19],  0.20, 0.07, [1.8, 1.4, 0.9],  [0.78, 0.62, 0.40], 0.92),
    ("liver",               "sss",  "#7A261A",  [0.48,0.15,0.10],  0.18, 0.07, [2.1, 1.6, 0.8],  [0.85, 0.72, 0.45], 0.92),
    # aorta: thin wall, SSS with very red scatter
    ("aorta",               "sss",  "#E01F1A",  [0.88,0.12,0.10],  0.06, 0.05, [2.5, 0.8, 0.5],  [0.90, 0.50, 0.30], 0.92),
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


def aces_filmic(hdr, exposure=1.1):
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


# ── GT render: roughdielectric + scaled homogeneous SSS ───────────────────────

def render_mitsuba_gt(tissue_objs, cam, aabb, spp, size=512, aperture_radius=6.0):
    """
    Surface: roughdielectric (wet capsule, IOR=1.36-1.55, alpha per tissue)
    Interior: homogeneous medium with `scale` bringing sigma_t to path-traceable range
    Integrator: volpath (required for participating interior media)
    Lights: large area key + fill only (no rim — it caused orange shadows on bone)
    """
    cx = (aabb[0]+aabb[3]) / 2
    cy = (aabb[1]+aabb[4]) / 2
    cz = (aabb[2]+aabb[5]) / 2
    scale = max(aabb[3]-aabb[0], aabb[4]-aabb[1], aabb[5]-aabb[2])

    kp = [cx + scale*0.73, cy - scale*0.54, cz + scale*1.28]
    fp = [cx - scale*0.59, cy + scale*1.52, cz + scale*1.48]
    sk = [cx, cy, cz]

    CAM_TF = mi.ScalarTransform4f.look_at(
        origin=cam["position"], target=cam["focal_point"], up=cam["up"])

    scene = {
        "type": "scene",
        # volpath required for interior media
        "integrator": {"type": "volpath", "max_depth": 64},
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
        # Larger key light (scale*0.30 vs *0.18 before) → softer highlights, less plastic
        # Neutral-warm colour (not orange) → no orange shadow cast onto bones
        "key_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(kp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([scale*0.30, scale*0.30, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [55.0, 52.0, 48.0]}},
        },
        "fill_light": {
            "type": "rectangle",
            "to_world": (mi.ScalarTransform4f.look_at(fp, sk, [0, 0, 1]) @
                         mi.ScalarTransform4f.scale([scale*0.35, scale*0.35, 1])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [2.0, 2.8, 5.5]}},
        },
        # Dim ambient only — no rim (rim caused orange bone shadows)
        "ambient": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.025, 0.025, 0.04]},
        },
    }

    for row in tissue_objs:
        seg_name, method, gt_rgb, roughness, med_scale, sigma_t_rgb, albedo_rgb, g, obj_path = row

        if method == "bone":
            scene[seg_name] = {
                "type": "obj",
                "filename": str(obj_path),
                "bsdf": {
                    "type":                "roughplastic",
                    "distribution":        "ggx",
                    "alpha":               roughness,
                    "int_ior":             1.55,
                    "diffuse_reflectance": {"type": "rgb", "value": gt_rgb},
                },
            }
        else:  # "sss"
            # Normalize sigma_t so max channel ≤ 1.0 (Mitsuba rgb spectrum clamps to [0,1])
            # Absorb the rescaling factor into scale to preserve effective MFP.
            st = list(sigma_t_rgb)
            st_max = max(st)
            if st_max > 1.0:
                st = [v / st_max for v in st]
                eff_scale = float(med_scale) * st_max
            else:
                eff_scale = float(med_scale)

            # roughdielectric surface: partially reflects (wet sheen) + partially
            # refracts into the interior medium → physically correct SSS entry point
            scene[seg_name] = {
                "type":     "obj",
                "filename": str(obj_path),
                "bsdf": {
                    "type":     "roughdielectric",
                    "distribution": "ggx",
                    "alpha":    roughness,
                    "int_ior":  1.40,
                    "ext_ior":  1.00,
                },
                # interior medium: scaled homogeneous SSS
                # scale brings effective sigma_t to ~MFP 5-7mm → path-traceable
                "interior": {
                    "type":    "homogeneous",
                    "scale":   eff_scale,
                    "sigma_t": {"type": "rgb", "value": st},
                    "albedo":  {"type": "rgb", "value": albedo_rgb},
                    "phase":   {"type": "hg", "g": float(g)},
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
    ap.add_argument("--angles",   type=int,   default=3)
    ap.add_argument("--aperture", type=float, default=6.0)
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
    print(f"[v4-sss-scaled]  subject={args.subject}  spp={args.spp}")
    print(f"BSDF: roughdielectric + homogeneous(scale)  volpath  DoF")
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

    for seg_name, method, simple_hex, gt_rgb, roughness, med_scale, sigma_t_rgb, albedo_rgb, g in TISSUES:
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
        tissue_objs.append((seg_name, method, gt_rgb, roughness,
                            med_scale, sigma_t_rgb, albedo_rgb, g, obj_path))
        sss_str = f"scale={med_scale} σt={sigma_t_rgb}" if method == "sss" else "bone(roughplastic)"
        print(f"  {seg_name:<30} {sss_str}")

    print(f"\n  {len(meshes_simple)} tissues loaded")

    angle_rows = []
    for i, cam in enumerate(cameras):
        label = cam.get("angle_label", f"{i}")
        print(f"\n--- Angle {label} ---")

        t0 = time.time()
        simple_img = render_simple(meshes_simple, cam, args.size)
        if simple_img.ndim == 3 and simple_img.shape[2] == 4:
            simple_img = simple_img[:, :, :3]
        print(f"  Simple: {time.time()-t0:.1f}s  std={simple_img.std():.1f}")
        Image.fromarray(simple_img).save(str(out_dir / f"simple_a{label}.png"))

        hdr    = render_mitsuba_gt(tissue_objs, cam, aabb, args.spp, args.size,
                                   aperture_radius=args.aperture)
        gt_img = aces_filmic(hdr)
        print(f"  GT   : std={gt_img.std():.1f}  max={gt_img.max()}")
        Image.fromarray(gt_img).save(str(out_dir / f"gt_v4_spp{args.spp}_a{label}.png"))

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
        d.text((5, y0+4), f"Simple — {args.subject} {label}", fill=(160,160,160))
        d.text((args.size+gap+5, y0+4),
               f"GT v4 — roughdielectric+SSS(scale) volpath {args.spp}SPP {label}",
               fill=(160,160,160))
        canvas.paste(Image.fromarray(simple_img), (0,             y0+label_h))
        canvas.paste(Image.fromarray(gt_img),     (args.size+gap, y0+label_h))

    grid_path = pair_out / f"{args.subject}_v4_sss_scaled_spp{args.spp}.png"
    canvas.save(str(grid_path))
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
