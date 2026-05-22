"""
render_pair_totalseg_v10_cycles.py — Blender Cycles + Random Walk SSS

INSTALL BLENDER FIRST (one time):
    brew install --cask blender

USAGE (run from repo root):
    blender --background --python scripts/render_pair_totalseg_v10_cycles.py -- \\
        --subject s0050 --spp 128 --angles 3

PRE-REQUISITE: mesh OBJ files must already exist. Generate them first:
    source venv/bin/activate
    python scripts/render_pair_totalseg_v7_bumpmap.py --subject s0050 --spp 1 --angles 1

WHY THIS IS BETTER THAN MITSUBA v7:
  - Blender's Principled BSDF has REAL Random Walk SSS
    (Mitsuba 3's principled has no SSS; our homogeneous-medium attempts gave dark images)
  - Per-tissue SSS radius per RGB channel → tissue color comes from differential scattering
  - Wet specular via Coat node (thin fluid layer / Glisson capsule)
  - Rim light positioned BEHIND anatomy → SSS glow through thin tissue walls
    (intestinal walls, vessels glow warm red = very cinematic)
  - AgX color management (better than ACES for organic materials)
  - Intel OIDN denoiser (clean 128-SPP results)
  - GPU-ready: add --device GPU to cycles settings if you have one

TISSUE SSS PARAMETERS (based on Jacques PMB 2013 + EndoPBR):
  sss_weight:  how translucent the tissue is (0=opaque, 1=full SSS)
  sss_scale:   mean free path in mm (1mm = skin depth scale)
  sss_radius:  (R, G, B) — wavelength-dependent MFP ratios.
               R>G>B → organ appears red/brown (red light exits farther from entry)
  coat_weight: thin fluid/capsule layer (Glisson capsule on liver, peritoneal fluid)
"""

import bpy
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path


# ── Parse args passed after "--" ─────────────────────────────────────────────
def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="s0050")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--spp",      type=int,   default=128)
    ap.add_argument("--size",     type=int,   default=512)
    ap.add_argument("--angles",   type=int,   default=3)
    ap.add_argument("--device",   default="CPU", choices=["CPU", "GPU"])
    return ap.parse_args(argv)


# ── Tissue definitions ────────────────────────────────────────────────────────
# (seg_name, simple_hex, base_color_rgb, roughness, ior,
#  sss_weight, sss_scale_mm, sss_radius_rgb,
#  coat_weight, coat_roughness,
#  bump_type, bump_scale, uv_mode)
#
# base_color_rgb: diffuse albedo [0-1] — actual tissue color, not atlas label
# sss_radius_rgb: per-channel MFP ratio. Red light travels ~10x farther than blue
#                 in most soft tissue → (1.0, 0.25, 0.1) = typical soft organ
# coat:           simulates peritoneal fluid / Glisson capsule thin layer

TISSUES = [
    # (name,               hex,       base_rgb,              rough  ior   sss_w  sss_mm  sss_r               coat_w  coat_r  bump        bs    uv)
    ("autochthon_left",   "#C05A28", [0.55,0.24,0.18],       0.38, 1.40,  0.25,  3.0,  (1.0, 0.3, 0.15),    0.2,   0.10, "fibrous",  0.35, "plane_xz"),
    ("autochthon_right",  "#C05A28", [0.55,0.24,0.18],       0.38, 1.40,  0.25,  3.0,  (1.0, 0.3, 0.15),    0.2,   0.10, "fibrous",  0.35, "plane_xz"),
    ("lung_lower_lobe_left",  "#88AABC",[0.78,0.75,0.74],    0.50, 1.35,  0.20,  8.0,  (1.5, 1.2, 1.0),     0.1,   0.15, "smooth",   0.20, "sphere"),
    ("lung_lower_lobe_right", "#88AABC",[0.78,0.75,0.74],    0.50, 1.35,  0.20,  8.0,  (1.5, 1.2, 1.0),     0.1,   0.15, "smooth",   0.20, "sphere"),
    ("lung_upper_lobe_left",  "#88AABC",[0.78,0.75,0.74],    0.50, 1.35,  0.20,  8.0,  (1.5, 1.2, 1.0),     0.1,   0.15, "smooth",   0.20, "sphere"),
    ("lung_upper_lobe_right", "#88AABC",[0.78,0.75,0.74],    0.50, 1.35,  0.20,  8.0,  (1.5, 1.2, 1.0),     0.1,   0.15, "smooth",   0.20, "sphere"),
    # Bone — no SSS, dry surface
    ("vertebrae_T12",     "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    ("vertebrae_L1",      "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    ("vertebrae_L2",      "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    ("vertebrae_L3",      "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    ("vertebrae_L4",      "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    ("vertebrae_L5",      "#EAE0CC", [0.90,0.85,0.77],       0.30, 1.55,  0.0,   0.0,  (1.0, 0.8, 0.6),     0.0,   0.30, "none",     0.0,  "none"),
    # Heart
    ("heart",             "#B03030", [0.52,0.14,0.12],       0.25, 1.40,  0.35,  4.0,  (1.2, 0.2, 0.1),     0.5,   0.06, "lobular",  0.45, "sphere"),
    ("esophagus",         "#C87878", [0.60,0.32,0.30],       0.32, 1.40,  0.20,  3.0,  (1.0, 0.3, 0.15),    0.3,   0.08, "vessel",   0.25, "plane_xz"),
    # Visceral organs — Glisson capsule gives strong wet coat
    ("stomach",           "#E8C866", [0.70,0.62,0.56],       0.28, 1.40,  0.30,  4.0,  (0.9, 0.5, 0.2),     0.6,   0.05, "wrinkled", 0.60, "sphere"),
    ("gallbladder",       "#60A840", [0.55,0.72,0.30],       0.12, 1.40,  0.25,  3.0,  (0.4, 1.0, 0.3),     0.8,   0.04, "lobular",  0.30, "sphere"),
    ("spleen",            "#7B3A8C", [0.38,0.10,0.16],       0.18, 1.40,  0.40,  4.0,  (1.2, 0.15, 0.2),    0.7,   0.04, "lobular",  0.55, "sphere"),
    ("kidney_right",      "#2A5EA8", [0.52,0.26,0.19],       0.20, 1.42,  0.35,  4.0,  (1.2, 0.3, 0.15),    0.6,   0.05, "lobular",  0.60, "sphere"),
    ("kidney_left",       "#2A6EB8", [0.52,0.26,0.19],       0.20, 1.42,  0.35,  4.0,  (1.2, 0.3, 0.15),    0.6,   0.05, "lobular",  0.60, "sphere"),
    # Liver — darkest organ, strong Glisson capsule
    ("liver",             "#9E3028", [0.48,0.15,0.10],       0.18, 1.38,  0.40,  5.0,  (1.5, 0.2, 0.1),     0.8,   0.04, "lobular",  0.70, "sphere"),
    ("pancreas",          "#E8B870", [0.78,0.62,0.38],       0.35, 1.40,  0.30,  4.0,  (1.0, 0.6, 0.3),     0.4,   0.06, "lobular",  0.50, "sphere"),
    ("duodenum",          "#D4A080", [0.68,0.50,0.38],       0.35, 1.40,  0.25,  4.0,  (0.9, 0.4, 0.2),     0.4,   0.07, "wrinkled", 0.55, "plane_xy"),
    ("small_bowel",       "#D4A080", [0.68,0.50,0.38],       0.35, 1.40,  0.25,  4.0,  (0.9, 0.4, 0.2),     0.4,   0.07, "wrinkled", 0.55, "sphere"),
    ("colon",             "#C88060", [0.64,0.44,0.34],       0.35, 1.40,  0.25,  4.0,  (0.8, 0.35, 0.18),   0.4,   0.07, "wrinkled", 0.55, "sphere"),
    ("urinary_bladder",   "#7888D4", [0.38,0.44,0.72],       0.25, 1.40,  0.15,  3.0,  (0.6, 0.5, 1.0),     0.6,   0.05, "smooth",   0.20, "sphere"),
    # Vascular — very wet, strong specular, thin-wall SSS for rim glow
    ("aorta",             "#DD1818", [0.88,0.12,0.10],       0.08, 1.38,  0.35,  2.0,  (2.0, 0.4, 0.2),     0.8,   0.03, "vessel",   0.25, "plane_xz"),
    ("inferior_vena_cava","#1840A0", [0.12,0.28,0.72],       0.10, 1.38,  0.20,  2.0,  (0.3, 0.5, 2.0),     0.8,   0.03, "vessel",   0.20, "plane_xz"),
    ("portal_vein_and_splenic_vein","#2050B8",[0.14,0.30,0.70],0.10,1.38, 0.20,  2.0,  (0.3, 0.5, 2.0),     0.7,   0.03, "vessel",   0.20, "plane_xy"),
    ("superior_vena_cava","#1840A0", [0.12,0.28,0.72],       0.10, 1.38,  0.20,  2.0,  (0.3, 0.5, 2.0),     0.8,   0.03, "vessel",   0.20, "plane_xz"),
]

TEX_DIR = Path("data/renders/textures")


# ── Scene setup ───────────────────────────────────────────────────────────────

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def setup_render(spp, size, device):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = spp
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.device = device
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    bg.inputs['Strength'].default_value = 0.0
    # AgX color management
    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass  # older Blender — AgX look names may differ
    # Scene units: 1 Blender unit = 1 mm
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001


# ── Material creation ─────────────────────────────────────────────────────────

def make_material(seg_name, base_rgb, roughness, ior,
                  sss_weight, sss_scale_mm, sss_radius,
                  coat_weight, coat_roughness,
                  bump_type, bump_scale):
    mat = bpy.data.materials.new(name=f"{seg_name}_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output     = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')

    # Base color
    principled.inputs['Base Color'].default_value = (*base_rgb, 1.0)
    principled.inputs['Roughness'].default_value  = roughness
    principled.inputs['IOR'].default_value        = ior

    # Random Walk SSS
    if sss_weight > 0:
        principled.subsurface_method = 'RANDOM_WALK'
        principled.inputs['Subsurface Weight'].default_value = sss_weight
        principled.inputs['Subsurface Scale'].default_value  = sss_scale_mm
        principled.inputs['Subsurface Radius'].default_value = sss_radius

    # Coat (Glisson capsule / peritoneal fluid)
    if coat_weight > 0:
        principled.inputs['Coat Weight'].default_value    = coat_weight
        principled.inputs['Coat Roughness'].default_value = coat_roughness
        principled.inputs['Coat IOR'].default_value       = 1.333  # water

    # Specular
    principled.inputs['Specular IOR Level'].default_value = 0.5

    # Bump map
    if bump_type != "none" and bump_scale > 0:
        bump_path = TEX_DIR / f"bump_{bump_type}.png"
        if bump_path.exists():
            uv_node  = nodes.new('ShaderNodeTexCoord')
            map_node = nodes.new('ShaderNodeMapping')
            map_node.inputs['Scale'].default_value = (4.0, 4.0, 4.0)  # tile 4x

            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.image = bpy.data.images.load(str(bump_path))
            tex_node.image.colorspace_settings.name = 'Non-Color'
            tex_node.extension = 'REPEAT'

            bump_node = nodes.new('ShaderNodeBump')
            bump_node.inputs['Strength'].default_value = bump_scale
            bump_node.inputs['Distance'].default_value = 1.0

            links.new(uv_node.outputs['UV'],          map_node.inputs['Vector'])
            links.new(map_node.outputs['Vector'],      tex_node.inputs['Vector'])
            links.new(tex_node.outputs['Color'],       bump_node.inputs['Height'])
            links.new(bump_node.outputs['Normal'],     principled.inputs['Normal'])

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ── Mesh import ───────────────────────────────────────────────────────────────

def import_obj(obj_path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis='Y',
        up_axis='Z',
    )
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    return new_objs[0] if new_objs else None


# ── Lighting ──────────────────────────────────────────────────────────────────

def setup_lights(cx, cy, cz, scene_scale):
    sc = scene_scale

    # Key: large warm area light, upper-left front
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*0.73, cy - sc*0.54, cz + sc*1.28))
    key = bpy.context.object
    key.data.energy = 40000
    key.data.color  = (1.00, 0.92, 0.82)  # warm white
    key.data.size   = sc * 0.30
    key.data.shape  = 'RECTANGLE'
    key.data.size_y = sc * 0.30
    # Point toward scene center
    _track_to(key, (cx, cy, cz))

    # Fill: large cool area light, opposite side
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.59, cy + sc*1.52, cz + sc*1.48))
    fill = bpy.context.object
    fill.data.energy = 4000
    fill.data.color  = (0.55, 0.68, 1.00)  # cool blue
    fill.data.size   = sc * 0.38
    _track_to(fill, (cx, cy, cz))

    # Rim: warm-orange from behind/below — activates SSS glow in thin tissue
    # Placed opposite to camera so light passes THROUGH thin organs
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.40, cy + sc*0.60, cz - sc*0.50))
    rim = bpy.context.object
    rim.data.energy = 8000
    rim.data.color  = (1.00, 0.45, 0.18)  # warm orange-red
    rim.data.size   = sc * 0.22
    _track_to(rim, (cx, cy, cz))


def _track_to(obj, target_xyz):
    """Point object's -Z axis toward target (standard light/camera convention)."""
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


# ── Camera ────────────────────────────────────────────────────────────────────

def setup_camera(size, fov_deg=30):
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam_obj = bpy.context.scene.camera = bpy.data.objects['Camera']
    cam_obj.data.type = 'PERSP'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle = math.radians(fov_deg)
    cam_obj.data.clip_start  = 1.0
    cam_obj.data.clip_end    = 10000.0
    # No DoF for now (adds noise, need high SPP)
    cam_obj.data.dof.use_dof = False
    return cam_obj


def point_camera(cam_obj, position, target):
    cam_obj.location = position
    _track_to(cam_obj, target)
    bpy.context.view_layer.update()


# ── Simple render (flat shading via Eevee for speed) ─────────────────────────

def setup_simple_material(seg_name, simple_hex):
    mat = bpy.data.materials.new(name=f"{seg_name}_simple")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output   = nodes.new('ShaderNodeOutputMaterial')
    diffuse  = nodes.new('ShaderNodeBsdfDiffuse')
    r = int(simple_hex[1:3], 16) / 255.0
    g = int(simple_hex[3:5], 16) / 255.0
    b = int(simple_hex[5:7], 16) / 255.0
    diffuse.inputs['Color'].default_value = (r, g, b, 1.0)
    diffuse.inputs['Roughness'].default_value = 0.8
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return mat


def render_simple(objs_with_mats, cam_obj, out_path, size):
    """Swap to flat Diffuse materials + EEVEE for the simple render."""
    scene = bpy.context.scene
    orig_engine = scene.render.engine
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 4

    for obj, simple_mat, _ in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(simple_mat)

    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    scene.render.engine = orig_engine


def restore_gt_materials(objs_with_mats):
    for obj, _, gt_mat in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(gt_mat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_png_as_numpy(path):
    """Load a Blender-rendered PNG back as a numpy (H,W,3) uint8 array."""
    img = bpy.data.images.load(str(path))
    w, h = img.size
    px = np.array(img.pixels, dtype=np.float32).reshape(h, w, 4)
    px = np.flipud(px)  # Blender origin is bottom-left
    px_u8 = (np.clip(px[:, :, :3], 0, 1) * 255).astype(np.uint8)
    bpy.data.images.remove(img)
    return px_u8


def save_numpy_as_png(arr, path):
    """Save (H,W,3) uint8 numpy array as PNG via Blender image API."""
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_tmp_out", width=w, height=h, alpha=False)
    # Blender wants float RGBA, bottom-left origin
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = arr.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    rgba = np.flipud(rgba)  # flip back to bottom-left
    img.pixels = rgba.flatten().tolist()
    img.filepath_raw = str(path)
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    mesh_dir = Path("data/renders/totalseg") / args.subject / "meshes"
    out_dir  = Path("data/renders/totalseg") / args.subject
    pair_out = Path("results/totalseg_pairs")
    pair_out.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_dir.exists():
        print(f"ERROR: mesh_dir not found: {mesh_dir}")
        print("Run v7 script first to generate meshes:")
        print(f"  python scripts/render_pair_totalseg_v7_bumpmap.py --subject {args.subject} --spp 1 --angles 1")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[v10-cycles]  subject={args.subject}  spp={args.spp}  angles={args.angles}")
    print(f"Blender Cycles + Random Walk SSS + Coat + Rim light + AgX")
    print(f"{'='*60}")

    # Scene units
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 0.001  # 1 unit = 1mm

    # Load CT metadata to get scene bounds (same camera formula as v1-v7)
    import nibabel as nib
    dataset_dir = Path(args.dataset)
    ct_path = dataset_dir / args.subject / "ct.nii.gz"
    ct_img  = nib.load(str(ct_path))
    shape   = ct_img.shape[:3]
    zooms   = ct_img.header.get_zooms()[:3]
    nx,ny,nz = shape
    sx,sy,sz = zooms
    cx,cy,cz = nx*sx/2, ny*sy/2, nz*sz/2
    radius   = max(nx*sx, ny*sy, nz*sz) * 0.9
    scene_scale = max(nx*sx, ny*sy, nz*sz)

    # Camera angles
    if args.angles == 1:
        offsets = [0]
    elif args.angles == 2:
        offsets = [-30, 30]
    else:
        half = (args.angles - 1) / 2
        step = 40 / half if half > 0 else 0
        offsets = [round(-40 + i*step) for i in range(args.angles)]

    def cam_pos_at_angle(theta_deg):
        t      = math.radians(theta_deg)
        dx_rel = radius * 0.6    # offset from scene centre in X
        dy_rel = -radius * 1.1   # offset from scene centre in Y
        return [cx + dx_rel*math.cos(t) - dy_rel*math.sin(t),
                cy + dx_rel*math.sin(t) + dy_rel*math.cos(t),
                cz + radius*0.5]

    # ── 1. Reset scene and set up render engine ───────────────────────────────
    reset_scene()
    setup_render(args.spp, args.size, args.device)
    cam_obj = setup_camera(args.size)
    setup_lights(cx, cy, cz, scene_scale)

    # ── 2. Import meshes and assign materials ─────────────────────────────────
    print("\n[1/3] Importing meshes ...")
    objs_with_mats = []   # [(blender_obj, simple_mat, gt_mat), ...]
    loaded = 0

    for row in TISSUES:
        (seg_name, simple_hex, base_rgb, roughness, ior,
         sss_weight, sss_scale_mm, sss_radius,
         coat_weight, coat_roughness,
         bump_type, bump_scale, uv_mode) = row

        # Prefer UV-mapped OBJ (for bump maps), fall back to plain
        obj_path = mesh_dir / f"{seg_name}_uv.obj"
        if not obj_path.exists():
            obj_path = mesh_dir / f"{seg_name}.obj"
        if not obj_path.exists():
            continue

        blender_obj = import_obj(obj_path)
        if blender_obj is None:
            continue

        simple_mat = setup_simple_material(seg_name, simple_hex)
        gt_mat     = make_material(seg_name, base_rgb, roughness, ior,
                                   sss_weight, sss_scale_mm, sss_radius,
                                   coat_weight, coat_roughness,
                                   bump_type, bump_scale)
        blender_obj.data.materials.clear()
        blender_obj.data.materials.append(gt_mat)
        objs_with_mats.append((blender_obj, simple_mat, gt_mat))
        print(f"  {seg_name:<35} SSS={sss_weight}  coat={coat_weight}")
        loaded += 1

    print(f"\n  {loaded} tissues loaded")

    # ── 3. Render each angle ──────────────────────────────────────────────────
    angle_rows = []

    for theta in offsets:
        label = f"{theta:+.0f}°"
        print(f"\n--- Angle {label} ---")
        cam_position = cam_pos_at_angle(theta)
        point_camera(cam_obj, cam_position, (cx, cy, cz))

        # Simple render (flat, Eevee)
        simple_path = out_dir / f"simple_v10_{label}.png"
        render_simple(objs_with_mats, cam_obj, simple_path, args.size)
        restore_gt_materials(objs_with_mats)
        print(f"  Simple render → {simple_path.name}")

        # GT render (Cycles + SSS)
        gt_path = out_dir / f"gt_v10_spp{args.spp}_{label}.png"
        bpy.context.scene.render.filepath = str(gt_path)
        bpy.ops.render.render(write_still=True)
        print(f"  GT render     → {gt_path.name}")

        angle_rows.append((simple_path, gt_path, label))

    # ── 4. Assemble grid ──────────────────────────────────────────────────────
    print("\n[3/3] Assembling grid ...")
    gap, label_h = 10, 32
    sz = args.size
    row_h = sz + label_h
    n_rows = len(angle_rows)
    grid = np.zeros((row_h * n_rows + gap, sz*2 + gap, 3), dtype=np.uint8)
    grid[:] = 12  # dark background

    for row_i, (sp, gp, label) in enumerate(angle_rows):
        y0 = row_i * row_h + gap
        if sp.exists():
            grid[y0+label_h : y0+label_h+sz, 0:sz] = load_png_as_numpy(sp)[:sz, :sz]
        if gp.exists():
            grid[y0+label_h : y0+label_h+sz, sz+gap:sz*2+gap] = load_png_as_numpy(gp)[:sz, :sz]

    grid_path = pair_out / f"{args.subject}_v10_cycles_spp{args.spp}.png"
    save_numpy_as_png(grid, grid_path)
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
