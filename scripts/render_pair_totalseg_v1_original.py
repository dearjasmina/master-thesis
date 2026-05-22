"""
render_pair_totalseg_v1_original.py — original first 128-SPP version
Restored from initial run (produced s0329_pair_spp128.png / _spp32.png).

This was the first version used in the session — roughplastic for all tissues,
two area lights (key + fill), no DoF, no rim, ACES×1.5, spp=128.

Usage:
    python scripts/render_pair_totalseg_v1_original.py \\
        --subject s0329 \\
        --dataset /path/to/Totalsegmentator_dataset_v201 \\
        --spp 128
"""

import argparse
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


TISSUES = [
    # (seg_name, bsdf_type, simple_hex, gt_rgb, roughness_alpha)
    ("autochthon_left",        "soft",   "#C05A28", [0.55, 0.24, 0.18], 0.45),
    ("autochthon_right",       "soft",   "#C05A28", [0.55, 0.24, 0.18], 0.45),
    ("lung_lower_lobe_left",   "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_lower_lobe_right",  "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_upper_lobe_left",   "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("lung_upper_lobe_right",  "soft",   "#88AABC", [0.78, 0.75, 0.74], 0.55),
    ("vertebrae_T12",          "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("vertebrae_L1",           "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("vertebrae_L2",           "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("vertebrae_L3",           "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("vertebrae_L4",           "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("vertebrae_L5",           "bone",   "#F5EDD8", [0.92, 0.88, 0.80], 0.25),
    ("stomach",                "soft",   "#E8C866", [0.70, 0.62, 0.56], 0.50),
    ("gallbladder",            "soft",   "#60A840", [0.55, 0.72, 0.30], 0.40),
    ("spleen",                 "soft",   "#7B3A8C", [0.38, 0.10, 0.16], 0.35),
    ("kidney_right",           "soft",   "#2A5EA8", [0.52, 0.26, 0.19], 0.35),
    ("kidney_left",            "soft",   "#2A6EB8", [0.52, 0.26, 0.19], 0.35),
    ("liver",                  "soft",   "#9E3028", [0.48, 0.15, 0.10], 0.35),
    ("aorta",                  "vessel", "#DD1818", [0.88, 0.12, 0.10], 0.18),
]


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
        ior = 1.55 if bsdf_type == "bone" else 1.40
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="s0329")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--spp",  type=int, default=128)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    subj_dir = Path(args.dataset) / args.subject
    seg_dir  = subj_dir / "segmentations"
    ct_path  = subj_dir / "ct.nii.gz"
    out_dir  = Path("data/renders/totalseg") / args.subject
    mesh_dir = out_dir / "meshes"
    pair_out = Path("results/totalseg_pairs")
    for d in [out_dir, mesh_dir, pair_out]:
        d.mkdir(parents=True, exist_ok=True)

    ct_img = nib.load(str(ct_path))
    shape  = ct_img.shape[:3]
    zooms  = ct_img.header.get_zooms()[:3]
    cam    = compute_camera(shape, zooms)
    aabb   = [0.0, 0.0, 0.0,
              float(shape[0]*zooms[0]), float(shape[1]*zooms[1]), float(shape[2]*zooms[2])]

    print(f"\n[v1-original]  subject={args.subject}  spp={args.spp}")
    print("[1/3] Extracting meshes ...")
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

    print("\n[2/3] Simple render ...")
    simple_img = render_simple(meshes_simple, cam, args.size)
    if simple_img.ndim == 3 and simple_img.shape[2] == 4:
        simple_img = simple_img[:, :, :3]
    Image.fromarray(simple_img).save(str(out_dir / "simple_v1.png"))

    print(f"[3/3] Mitsuba GT ({args.spp} SPP) ...")
    hdr    = render_mitsuba_gt(tissue_objs, cam, aabb, args.spp, args.size)
    gt_img = aces_filmic(hdr, exposure=1.5)
    print(f"  GT std={gt_img.std():.1f}  max={gt_img.max()}")
    Image.fromarray(gt_img).save(str(out_dir / f"gt_v1_spp{args.spp}.png"))

    gap    = 12
    canvas = Image.new("RGB", (args.size*2 + gap, args.size + 40), (12, 12, 12))
    canvas.paste(Image.fromarray(simple_img), (0, 40))
    canvas.paste(Image.fromarray(gt_img),     (args.size + gap, 40))
    d = ImageDraw.Draw(canvas)
    d.text((5, 5),               f"Simple — {args.subject} [v1]", fill=(180,180,180))
    d.text((args.size+gap+5, 5), f"GT — roughplastic {args.spp}SPP ACES×1.5 [v1]", fill=(180,180,180))
    out = pair_out / f"{args.subject}_v1_original_spp{args.spp}.png"
    canvas.save(str(out))
    print(f"\nPair → {out}")


if __name__ == "__main__":
    main()
