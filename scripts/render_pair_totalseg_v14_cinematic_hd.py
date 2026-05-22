"""
render_pair_totalseg_v14_cinematic_hd.py — Heterogeneous Anatomy & Balanced Spectrum

v14 upgrades over v13:
  - Strict temperature clustering: eradicated universal red-channel albedo bias
  - Diffuse bounces capped at 2: forces deep spaces to swallow light, eliminates pink bounce flood
  - Key light tuned to clinical slate-blue to neutralize red surface reflections
  - Warm rim desaturated to low-energy neutral tungsten ivory (stops orange glossy contamination)
  - Cool rim 2 strengthened to 125W (separates dark visceral edges from void)
  - Subtle cool VolumeScatter atmosphere (density=0.004) for filmic depth and cooled deep shadows
  - HSV saturation node (0.85) after AO mix prevents meat-colored specular overload
  - SSS scale reduced to 0.0025x (from 0.003x) — tighter opaque surface threshold
  - FOV narrowed to 18° (tighter cinematic lens compression)
  - f/6.3 aperture (vs f/8), focus pull at 94%
  - AgX High Contrast (from Medium High Contrast) + exposure -0.1
  - AO: 3.5mm, factor 0.35

Run:
    blender --background --python scripts/render_pair_totalseg_v14_cinematic_hd.py -- \\
        --subject s0050 --spp 384 --size 1024 --angles 1
"""

import bpy
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path


# ── Parse args ────────────────────────────────────────────────────────────────
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
    ap.add_argument("--spp",    type=int, default=384)
    ap.add_argument("--size",   type=int, default=1024)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    return ap.parse_args(argv)


# ── Tissue definitions — temperature-clustered, luminance-orchestrated palette ─
# (name, simple_hex, base_rgb, rough, ior,
#  sss_weight, sss_scale_mm, sss_radius_rgb,
#  coat_weight, coat_roughness, bump_type, bump_scale)
TISSUES = [
    # Structural neutral — near-black backdrop, completely stripped of warm competition
    ("autochthon_left",   "#4A3E3D", [0.05, 0.03, 0.025], 0.50, 1.40, 0.01, 1.0, (0.5,0.1,0.01),  0.05, 0.20, "fibrous",  0.15),
    ("autochthon_right",  "#4A3E3D", [0.05, 0.03, 0.025], 0.50, 1.40, 0.01, 1.0, (0.5,0.1,0.01),  0.05, 0.20, "fibrous",  0.15),
    # Cool midtones — desaturated blue-gray/slate lungs counter warm scatter
    ("lung_lower_lobe_left",  "#6B7A82", [0.09, 0.10, 0.11], 0.45, 1.35, 0.02, 1.5, (0.4,0.2,0.1),  0.10, 0.15, "smooth", 0.10),
    ("lung_lower_lobe_right", "#6B7A82", [0.09, 0.10, 0.11], 0.45, 1.35, 0.02, 1.5, (0.4,0.2,0.1),  0.10, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_left",  "#6B7A82", [0.09, 0.10, 0.11], 0.45, 1.35, 0.02, 1.5, (0.4,0.2,0.1),  0.10, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_right", "#6B7A82", [0.09, 0.10, 0.11], 0.45, 1.35, 0.02, 1.5, (0.4,0.2,0.1),  0.10, 0.15, "smooth", 0.10),
    # Structural neutral — dry aged matte ivory bone
    ("vertebrae_T12", "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L1",  "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L2",  "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L3",  "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L4",  "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L5",  "#C2BBB0", [0.20, 0.18, 0.15], 0.75, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    # Warm vascular — specular carries wetness, not albedo
    ("heart",      "#8A2B2B", [0.18, 0.045, 0.04], 0.30, 1.40, 0.03, 2.0, (1.0,0.1,0.02),  0.25, 0.06, "lobular", 0.35),
    ("esophagus",  "#9E6464", [0.22, 0.11, 0.10],  0.38, 1.40, 0.02, 1.5, (0.8,0.15,0.04), 0.12, 0.10, "vessel",  0.20),
    # Core value anchor — deep mahogany, heavily desaturated, stops acting as red mirror
    ("liver",      "#5C2420", [0.045,0.028,0.018], 0.24, 1.38, 0.04, 1.8, (1.2,0.08,0.01), 0.45, 0.03, "lobular", 0.40),
    # Neutral fleshy — sickly olive/ochre shatters color monotony against liver
    ("stomach",    "#9E916B", [0.22, 0.20, 0.13],  0.32, 1.40, 0.04, 2.0, (0.8,0.3,0.05),  0.18, 0.08, "wrinkled",0.45),
    # Visceral accent — dark forest green gallbladder
    ("gallbladder","#3A5E35", [0.03, 0.08, 0.03],  0.18, 1.40, 0.05, 1.8, (0.1,0.8,0.1),   0.35, 0.04, "lobular", 0.30),
    # Cool violet — bruised dark purple-gray spleen
    ("spleen",     "#523559", [0.08, 0.03, 0.09],  0.22, 1.40, 0.04, 2.0, (0.8,0.1,0.15),  0.35, 0.04, "lobular", 0.40),
    # Cool cyan family — deep Prussian blue kidneys, cold foil against hot aorta
    ("kidney_right","#204369", [0.04, 0.07, 0.11], 0.25, 1.42, 0.03, 2.0, (0.2,0.3,0.8),   0.30, 0.05, "lobular", 0.45),
    ("kidney_left", "#204369", [0.04, 0.07, 0.11], 0.25, 1.42, 0.03, 2.0, (0.2,0.3,0.8),   0.30, 0.05, "lobular", 0.45),
    # Neutral fleshy / brightest warm band — highly desaturated organic clay-taupe bowel
    ("pancreas",   "#B09170", [0.25, 0.19, 0.13],  0.40, 1.40, 0.03, 1.5, (0.7,0.4,0.1),   0.15, 0.09, "lobular", 0.40),
    ("duodenum",   "#A38470", [0.22, 0.16, 0.13],  0.36, 1.40, 0.02, 2.0, (0.6,0.2,0.08),  0.18, 0.08, "wrinkled",0.40),
    ("small_bowel","#A38470", [0.22, 0.16, 0.13],  0.36, 1.40, 0.02, 2.0, (0.6,0.2,0.08),  0.18, 0.08, "wrinkled",0.40),
    ("colon",      "#8F6E5C", [0.18, 0.12, 0.10],  0.36, 1.40, 0.02, 2.0, (0.5,0.2,0.08),  0.18, 0.08, "wrinkled",0.40),
    ("urinary_bladder","#5E688A",[0.08,0.09,0.16], 0.28, 1.40, 0.02, 1.5, (0.2,0.2,0.6),   0.20, 0.06, "smooth",  0.20),
    # Selective saturation — bright surgical tracking lines draw the eye across the dark canvas
    ("aorta",                       "#A31414", [0.32,0.02,0.01], 0.12, 1.38, 0.05, 1.2, (1.5,0.05,0.01), 0.40, 0.03, "vessel", 0.20),
    ("inferior_vena_cava",          "#14398A", [0.01,0.03,0.22], 0.14, 1.38, 0.04, 1.2, (0.05,0.1,1.5),  0.35, 0.03, "vessel", 0.15),
    ("portal_vein_and_splenic_vein","#1A439E", [0.01,0.04,0.20], 0.14, 1.38, 0.04, 1.2, (0.05,0.1,1.5),  0.30, 0.03, "vessel", 0.15),
    ("superior_vena_cava",          "#14398A", [0.01,0.03,0.22], 0.14, 1.38, 0.04, 1.2, (0.05,0.1,1.5),  0.35, 0.03, "vessel", 0.15),
]

TEX_DIR = Path("data/renders/textures")


# ── Scene helpers ─────────────────────────────────────────────────────────────

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def setup_render(spp, size, device):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = spp
    scene.cycles.use_denoising = False
    scene.cycles.device = device
    scene.render.film_transparent = False

    # Bounce choke: 2 diffuse bounces force internal cavities to stay dark
    scene.cycles.max_bounces             = 12
    scene.cycles.diffuse_bounces         = 2
    scene.cycles.glossy_bounces          = 3
    scene.cycles.transmission_bounces    = 6
    scene.cycles.volume_bounces          = 2   # needed for the cool atmosphere scatter
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.blur_glossy             = 0.2

    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width      = 0.5

    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = 'PNG'

    # World: deep near-black + cool filmic depth atmosphere
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    wt = scene.world.node_tree
    wt.nodes.clear()
    wout = wt.nodes.new('ShaderNodeOutputWorld')

    bg = wt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value    = (0.001, 0.001, 0.002, 1)
    bg.inputs['Strength'].default_value = 0.02
    wt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # Cool volume scatter — cools deep bounce and gives filmic air depth
    vol = wt.nodes.new('ShaderNodeVolumeScatter')
    vol.inputs['Color'].default_value     = (0.55, 0.70, 1.0, 1)  # slate-blue atmosphere
    vol.inputs['Density'].default_value   = 0.004
    vol.inputs['Anisotropy'].default_value = 0.35
    wt.links.new(vol.outputs['Volume'], wout.inputs['Volume'])

    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - High Contrast'
    except Exception:
        try:
            scene.view_settings.look = 'AgX - Medium High Contrast'
        except Exception:
            pass
    scene.view_settings.exposure = -0.1

    scene.unit_settings.system = 'METRIC'


def setup_compositor(scene):
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        if tree is None:
            return
        tree.nodes.clear()
        rl  = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(rl.outputs['Image'], out.inputs['Image'])
    except Exception:
        pass


def teardown_compositor(scene):
    try:
        scene.use_nodes = False
    except Exception:
        pass


# ── Negative fill planes ──────────────────────────────────────────────────────

def add_negative_fill_planes(cx, cy, cz, scene_scale):
    sc = scene_scale
    mat = bpy.data.materials.new("NegFill")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0, 0, 0, 1)
        bsdf.inputs['Roughness'].default_value  = 1.0

    def make_plane(loc, rot_euler, name):
        bpy.ops.mesh.primitive_plane_add(size=sc * 4.0, location=loc)
        p = bpy.context.object
        p.rotation_euler = rot_euler
        p.data.materials.append(mat)
        p.name = name

    make_plane((cx - sc*2.0, cy + sc*0.2, cz), (0, math.pi/2, 0), "NegFill_Left")
    make_plane((cx + sc*2.0, cy + sc*0.2, cz), (0, math.pi/2, 0), "NegFill_Right")


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
    principled.inputs['IOR'].default_value = ior

    # AO contact shadows — tight 3.5mm crease stamp
    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 0.0035
    mix_ao = nodes.new('ShaderNodeMix')
    mix_ao.data_type  = 'RGBA'
    mix_ao.blend_type = 'MULTIPLY'
    mix_ao.inputs['Factor'].default_value = 0.35
    mix_ao.inputs[6].default_value = (*base_rgb, 1.0)
    links.new(ao_node.outputs['Color'], mix_ao.inputs[7])

    # Surgical albedo desaturation — prevents meat-colored specular overload
    hsv_node = nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Saturation'].default_value = 0.85
    links.new(mix_ao.outputs[2], hsv_node.inputs['Color'])
    links.new(hsv_node.outputs['Color'], principled.inputs['Base Color'])

    # UV + mapping
    uv_node  = nodes.new('ShaderNodeTexCoord')
    map_node = nodes.new('ShaderNodeMapping')
    map_node.inputs['Scale'].default_value = (3.5, 3.5, 3.5)
    links.new(uv_node.outputs['UV'], map_node.inputs['Vector'])

    # Roughness noise — tighter frequency for wrinkled structures
    noise_r = nodes.new('ShaderNodeTexNoise')
    _wrinkled = ("small_bowel", "colon", "duodenum", "stomach", "pancreas")
    if any(w in seg_name for w in _wrinkled):
        noise_r.inputs['Scale'].default_value  = 48.0
        noise_r.inputs['Detail'].default_value = 12.0
    else:
        noise_r.inputs['Scale'].default_value  = 22.0
        noise_r.inputs['Detail'].default_value = 8.0
    noise_r.inputs['Roughness'].default_value  = 0.60
    noise_r.inputs['Distortion'].default_value = 0.15
    links.new(map_node.outputs['Vector'], noise_r.inputs['Vector'])

    val_map = nodes.new('ShaderNodeMapRange')
    val_map.inputs['From Min'].default_value = 0.0
    val_map.inputs['From Max'].default_value = 1.0
    val_map.inputs['To Min'].default_value   = roughness * 0.4
    val_map.inputs['To Max'].default_value   = roughness * 1.0
    links.new(noise_r.outputs['Factor'], val_map.inputs['Value'])
    links.new(val_map.outputs['Result'], principled.inputs['Roughness'])

    # Random Walk SSS — skip hollow thin-walled structures
    _hollow = ("small_bowel", "colon", "duodenum", "stomach", "esophagus")
    if sss_weight > 0 and not any(h in seg_name for h in _hollow):
        principled.subsurface_method = 'RANDOM_WALK'
        principled.inputs['Subsurface Weight'].default_value = sss_weight
        principled.inputs['Subsurface Scale'].default_value  = sss_scale_mm / 1000.0 * 0.0025
        principled.inputs['Subsurface Radius'].default_value = sss_radius

    # Wet peritoneal coat — per-tissue roughness
    if coat_weight > 0:
        principled.inputs['Coat Weight'].default_value    = coat_weight
        principled.inputs['Coat Roughness'].default_value = coat_roughness
        principled.inputs['Coat IOR'].default_value       = 1.41

    principled.inputs['Specular IOR Level'].default_value = 0.5

    bevel_node = nodes.new('ShaderNodeBevel')
    bevel_node.samples = 4
    bevel_node.inputs['Radius'].default_value = 0.00012

    if bump_type != "none" and bump_scale > 0:
        bump_path = TEX_DIR / f"bump_{bump_type}.png"
        if bump_path.exists():
            map_b = nodes.new('ShaderNodeMapping')
            map_b.inputs['Scale'].default_value = (3.5, 3.5, 3.5)
            links.new(uv_node.outputs['UV'], map_b.inputs['Vector'])

            tex_b = nodes.new('ShaderNodeTexImage')
            tex_b.image = bpy.data.images.load(str(bump_path))
            tex_b.image.colorspace_settings.name = 'Non-Color'
            tex_b.extension = 'REPEAT'
            links.new(map_b.outputs['Vector'], tex_b.inputs['Vector'])

            bump_n = nodes.new('ShaderNodeBump')
            bump_n.inputs['Strength'].default_value = bump_scale * 1.15
            bump_n.inputs['Distance'].default_value = 0.00022
            links.new(bevel_node.outputs['Normal'], bump_n.inputs['Normal'])
            links.new(tex_b.outputs['Color'],   bump_n.inputs['Height'])
            links.new(bump_n.outputs['Normal'], principled.inputs['Normal'])
    else:
        links.new(bevel_node.outputs['Normal'], principled.inputs['Normal'])

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ── Mesh import ───────────────────────────────────────────────────────────────

def import_obj(obj_path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis='Y',
        up_axis='Z',
        global_scale=0.001,
    )
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    return new_objs[0] if new_objs else None


# ── Lighting ──────────────────────────────────────────────────────────────────

def setup_lights(cx, cy, cz, scene_scale):
    sc = scene_scale

    # Key: cold slate-cyan — actively neutralizes warm tissue reflections
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*1.4, cy + sc*0.3, cz + sc*0.8))
    key = bpy.context.object
    key.data.energy = 115
    key.data.color  = (0.76, 0.86, 1.00)
    key.data.size   = sc * 0.11
    key.data.shape  = 'SQUARE'
    _track_to(key, (cx, cy, cz))

    # Fill: minimal ambient recovery only
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*1.0, cy - sc*1.0, cz + sc*0.5))
    fill = bpy.context.object
    fill.data.energy = 2.0
    fill.data.color  = (0.60, 0.72, 0.95)
    fill.data.size   = sc * 0.50
    _track_to(fill, (cx, cy, cz))

    # Rim 1: desaturated neutral tungsten ivory — removed hot orange contamination
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.5, cy + sc*1.4, cz + sc*0.3))
    rim1 = bpy.context.object
    rim1.data.energy = 25
    rim1.data.color  = (1.00, 0.78, 0.62)
    rim1.data.size   = sc * 0.02
    _track_to(rim1, (cx, cy, cz))

    # Rim 2: clean cyan-blue kicker — separates dark visceral edges from the void
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*0.4, cy - sc*1.4, cz - sc*0.2))
    rim2 = bpy.context.object
    rim2.data.energy = 125
    rim2.data.color  = (0.35, 0.65, 1.00)
    rim2.data.size   = sc * 0.02
    _track_to(rim2, (cx, cy, cz))


def _track_to(obj, target_xyz):
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


# ── Camera ────────────────────────────────────────────────────────────────────

def setup_camera(size, fov_deg=18):  # 18° = tighter cinematic lens compression
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam_obj = bpy.context.scene.camera = bpy.data.objects['Camera']
    cam_obj.data.type      = 'PERSP'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle     = math.radians(fov_deg)
    cam_obj.data.clip_start = 0.001
    cam_obj.data.clip_end   = 100.0
    cam_obj.data.dof.use_dof        = True
    cam_obj.data.dof.aperture_fstop = 6.3
    cam_obj.data.shift_x = -0.05
    cam_obj.data.shift_y =  0.02
    return cam_obj


def point_camera(cam_obj, position, target):
    cam_obj.location = position
    _track_to(cam_obj, target)
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(position, target)))
    cam_obj.data.dof.focus_distance = dist * 0.94
    bpy.context.view_layer.update()


# ── Simple render (flat EEVEE) ────────────────────────────────────────────────

def setup_simple_material(seg_name, simple_hex):
    mat = bpy.data.materials.new(name=f"{seg_name}_simple")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output  = nodes.new('ShaderNodeOutputMaterial')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    r = int(simple_hex[1:3], 16) / 255.0
    g = int(simple_hex[3:5], 16) / 255.0
    b = int(simple_hex[5:7], 16) / 255.0
    diffuse.inputs['Color'].default_value    = (r, g, b, 1.0)
    diffuse.inputs['Roughness'].default_value = 0.8
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return mat


def render_simple(objs_with_mats, cam_obj, out_path, size):
    scene = bpy.context.scene
    orig_engine = scene.render.engine
    teardown_compositor(scene)
    cam_obj.data.dof.use_dof = False
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 4
    for obj, simple_mat, _ in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(simple_mat)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    scene.render.engine = orig_engine
    cam_obj.data.dof.use_dof = True
    setup_compositor(scene)


def restore_gt_materials(objs_with_mats):
    for obj, _, gt_mat in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(gt_mat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_png_as_numpy(path):
    img = bpy.data.images.load(str(path))
    w, h = img.size
    px = np.array(img.pixels, dtype=np.float32).reshape(h, w, 4)
    px = np.flipud(px)
    px_u8 = (np.clip(px[:, :, :3], 0, 1) * 255).astype(np.uint8)
    bpy.data.images.remove(img)
    return px_u8


def save_numpy_as_png(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_tmp_out", width=w, height=h, alpha=False)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = arr.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    rgba = np.flipud(rgba)
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
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"[v14-cinematic-hd]  subject={args.subject}  spp={args.spp}  size={args.size}px")
    print(f"Temperature clustering + bounce choke + cool atmosphere")
    print(f"{'='*65}")

    import nibabel as nib
    ct_img   = nib.load(str(Path(args.dataset) / args.subject / "ct.nii.gz"))
    shape    = ct_img.shape[:3]
    zooms    = ct_img.header.get_zooms()[:3]
    nx,ny,nz = shape
    sx,sy,sz = zooms
    cx, cy, cz  = nx*sx/2/1000, ny*sy/2/1000, nz*sz/2/1000
    radius      = max(nx*sx, ny*sy, nz*sz) * 0.9 / 1000
    scene_scale = max(nx*sx, ny*sy, nz*sz) / 1000

    if args.angles == 1:
        offsets = [-20]
    elif args.angles == 2:
        offsets = [-40, 40]
    else:
        offsets = [-40, 0, 40]

    def cam_pos_at_angle(theta_deg):
        t      = math.radians(theta_deg)
        dx_rel = radius * 0.6
        dy_rel = radius * 1.15
        return [cx + dx_rel*math.cos(t) - dy_rel*math.sin(t),
                cy + dx_rel*math.sin(t) + dy_rel*math.cos(t),
                cz + radius*0.45]

    reset_scene()
    setup_render(args.spp, args.size, args.device)
    cam_obj = setup_camera(args.size)
    setup_lights(cx, cy, cz, scene_scale)
    add_negative_fill_planes(cx, cy, cz, scene_scale)
    setup_compositor(bpy.context.scene)

    print("\n[1/3] Importing meshes ...")
    objs_with_mats = []
    loaded = 0

    for row in TISSUES:
        (seg_name, simple_hex, base_rgb, roughness, ior,
         sss_weight, sss_scale_mm, sss_radius,
         coat_weight, coat_roughness,
         bump_type, bump_scale) = row

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
        print(f"  {seg_name:<38} sss={sss_weight:.2f}  coat={coat_weight:.2f}")
        loaded += 1

    print(f"\n  {loaded} tissues loaded")

    print("\n[2/3] Rendering angles ...")
    angle_rows = []

    for theta in offsets:
        label = f"{theta:+.0f}°"
        print(f"\n--- Angle {label} ---")
        cam_position = cam_pos_at_angle(theta)
        point_camera(cam_obj, cam_position, (cx, cy, cz))

        simple_path = out_dir / f"simple_v14_{label}.png"
        render_simple(objs_with_mats, cam_obj, simple_path, args.size)
        restore_gt_materials(objs_with_mats)
        print(f"  Simple → {simple_path.name}")

        gt_path = out_dir / f"gt_v14_spp{args.spp}_{label}.png"
        bpy.context.scene.render.filepath = str(gt_path)
        bpy.ops.render.render(write_still=True)
        print(f"  GT     → {gt_path.name}")

        angle_rows.append((simple_path, gt_path, label))

    print("\n[3/3] Assembling grid ...")
    gap, label_h = 15, 40
    sz = args.size
    n  = len(angle_rows)
    grid = np.zeros(((sz + label_h) * n + gap, sz*2 + gap, 3), dtype=np.uint8)
    grid[:] = 10

    for i, (sp, gp, _lbl) in enumerate(angle_rows):
        y0 = i * (sz + label_h) + gap
        if sp.exists():
            grid[y0+label_h : y0+label_h+sz, 0:sz]            = load_png_as_numpy(sp)[:sz, :sz]
        if gp.exists():
            grid[y0+label_h : y0+label_h+sz, sz+gap:sz*2+gap] = load_png_as_numpy(gp)[:sz, :sz]

    grid_path = pair_out / f"{args.subject}_v14_cinematic_hd_spp{args.spp}.png"
    save_numpy_as_png(grid, grid_path)
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
