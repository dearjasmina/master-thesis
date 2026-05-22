"""
render_pair_totalseg_v12_cinematic.py — Ultra-HD Cinematic Blender Cycles

v12 upgrades over v11:
  - OIDN denoiser OFF — raw pixel sharpness, SSS/specular edges no longer smeared
  - AO contact shadow node — organ separation, crease darkening baked into material
  - Bevel shading node — Hollywood edge specular catch without geometry changes
  - SSS scale 0.005× (from 0.012) — surfaces stay opaque, no wax bleed
  - Roughness noise Scale=20 (coherent wet variation, not micro-glitter)
  - Key light size sc*0.12 (broader = clean cinematic falloff, no noisy breakup)
  - Dual rim: warm amber + cool blue kicker (traces dark organs from both sides)
  - FOV 20° (cinematic lens compression vs 28°)
  - DoF f/4.5 (sharp midground, subtle fore/background bokeh)
  - Camera shift -0.05x/+0.02y (asymmetric composition)
  - Asymmetric angle sweep [-55, -20, +15] (cinematic, not dataset-clinical)
  - 384 SPP minimum (compensates for no denoiser; run 512 for production)

Run:
    blender --background --python scripts/render_pair_totalseg_v12_cinematic.py -- \\
        --subject s0050 --spp 384 --size 1024 --angles 3
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
    ap.add_argument("--spp",    type=int, default=256)
    ap.add_argument("--size",   type=int, default=1024)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    return ap.parse_args(argv)


# ── Tissue definitions — high-contrast cinematic palette ─────────────────────
# (name, simple_hex, base_rgb, rough, ior,
#  sss_weight, sss_scale_mm, sss_radius_rgb,
#  coat_weight, coat_roughness, bump_type, bump_scale, uv_mode)
TISSUES = [
    # Muscles: very dark — heavy opaque background, not competing with organs
    ("autochthon_left",   "#C05A28", [0.12,0.04,0.03], 0.35, 1.40, 0.06, 1.5, (1.0,0.1,0.02),  0.15, 0.10, "fibrous",  0.25, "plane_xz"),
    ("autochthon_right",  "#C05A28", [0.12,0.04,0.03], 0.35, 1.40, 0.06, 1.5, (1.0,0.1,0.02),  0.15, 0.10, "fibrous",  0.25, "plane_xz"),
    # Lungs: dark blood-perfused, tight SSS to prevent paper-lantern glowing
    ("lung_lower_lobe_left",  "#88AABC", [0.16,0.08,0.09], 0.40, 1.35, 0.10, 2.0, (1.2,0.2,0.05), 0.15, 0.15, "smooth", 0.15, "sphere"),
    ("lung_lower_lobe_right", "#88AABC", [0.16,0.08,0.09], 0.40, 1.35, 0.10, 2.0, (1.2,0.2,0.05), 0.15, 0.15, "smooth", 0.15, "sphere"),
    ("lung_upper_lobe_left",  "#88AABC", [0.16,0.08,0.09], 0.40, 1.35, 0.10, 2.0, (1.2,0.2,0.05), 0.15, 0.15, "smooth", 0.15, "sphere"),
    ("lung_upper_lobe_right", "#88AABC", [0.16,0.08,0.09], 0.40, 1.35, 0.10, 2.0, (1.2,0.2,0.05), 0.15, 0.15, "smooth", 0.15, "sphere"),
    # Bone: dry, matte ivory — structural shadow element, zero SSS, zero coat
    ("vertebrae_T12", "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    ("vertebrae_L1",  "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    ("vertebrae_L2",  "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    ("vertebrae_L3",  "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    ("vertebrae_L4",  "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    ("vertebrae_L5",  "#EAE0CC", [0.25,0.22,0.18], 0.65, 1.55, 0.0, 0.0, (1.0,0.8,0.6), 0.0, 0.30, "none", 0.0, "none"),
    # Heart: rich muscular crimson — tight SSS, full wet coat
    ("heart",      "#B03030", [0.28,0.06,0.05], 0.28, 1.40, 0.08, 3.0, (1.2,0.15,0.05), 0.25, 0.06, "lobular", 0.45, "sphere"),
    ("esophagus",  "#C87878", [0.35,0.15,0.14], 0.35, 1.40, 0.07, 2.0, (1.0,0.2,0.08),  0.15, 0.08, "vessel",  0.25, "plane_xz"),
    # Liver: deep crimson anchor — darkest organ, maximum value separation from bowel
    ("liver",      "#9E3028", [0.07,0.015,0.012], 0.22, 1.38, 0.12, 2.0, (1.5,0.1,0.02), 0.40, 0.04, "lobular", 0.50, "sphere"),
    # Stomach: olive/ochre — contrasts against deep liver
    ("stomach",    "#E8C866", [0.38,0.32,0.22], 0.30, 1.40, 0.10, 2.5, (1.0,0.4,0.1),   0.30, 0.05, "wrinkled",0.50, "sphere"),
    # Gallbladder: dark forest green
    ("gallbladder","#60A840", [0.08,0.18,0.05], 0.15, 1.40, 0.10, 2.0, (0.2,1.0,0.2),   0.35, 0.04, "lobular", 0.30, "sphere"),
    # Spleen: deep purplish-red
    ("spleen",     "#7B3A8C", [0.18,0.03,0.08], 0.20, 1.40, 0.10, 2.5, (1.2,0.1,0.15),  0.35, 0.04, "lobular", 0.45, "sphere"),
    # Kidneys
    ("kidney_right","#2A5EA8",[0.15,0.06,0.10], 0.22, 1.42, 0.08, 2.5, (1.2,0.2,0.1),   0.30, 0.05, "lobular", 0.50, "sphere"),
    ("kidney_left", "#2A6EB8",[0.15,0.06,0.10], 0.22, 1.42, 0.08, 2.5, (1.2,0.2,0.1),   0.30, 0.05, "lobular", 0.50, "sphere"),
    # Bowel group: tan/ochre — brightest tonal band, pops against deep liver
    ("pancreas",   "#E8B870", [0.42,0.30,0.16], 0.38, 1.40, 0.08, 2.0, (1.0,0.5,0.2),   0.20, 0.06, "lobular", 0.45, "sphere"),
    ("duodenum",   "#D4A080", [0.38,0.24,0.16], 0.35, 1.40, 0.06, 2.5, (0.9,0.3,0.1),   0.25, 0.07, "wrinkled",0.45, "plane_xy"),
    ("small_bowel","#D4A080", [0.38,0.24,0.16], 0.35, 1.40, 0.06, 2.5, (0.9,0.3,0.1),   0.25, 0.07, "wrinkled",0.45, "sphere"),
    ("colon",      "#C88060", [0.34,0.20,0.14], 0.35, 1.40, 0.06, 2.5, (0.8,0.3,0.1),   0.25, 0.07, "wrinkled",0.45, "sphere"),
    ("urinary_bladder","#7888D4",[0.16,0.18,0.35],0.25,1.40,0.07, 2.0, (0.4,0.4,1.0),   0.25, 0.05, "smooth",  0.20, "sphere"),
    # Vessels: pure saturated red/blue — brightest specular under key
    ("aorta",                    "#DD1818",[0.45,0.03,0.02],0.10,1.38,0.15,1.5,(2.0,0.1,0.02),0.40,0.03,"vessel",0.20,"plane_xz"),
    ("inferior_vena_cava",       "#1840A0",[0.02,0.06,0.30],0.12,1.38,0.10,1.5,(0.1,0.2,2.0), 0.35,0.03,"vessel",0.15,"plane_xz"),
    ("portal_vein_and_splenic_vein","#2050B8",[0.03,0.08,0.28],0.12,1.38,0.10,1.5,(0.1,0.2,2.0),0.30,0.03,"vessel",0.15,"plane_xy"),
    ("superior_vena_cava",       "#1840A0",[0.02,0.06,0.30],0.12,1.38,0.10,1.5,(0.1,0.2,2.0), 0.35,0.03,"vessel",0.15,"plane_xz"),
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
    scene.cycles.use_denoising = False  # raw pixels — OIDN smears SSS edges and wet speculars
    scene.cycles.device = device
    scene.render.film_transparent = False

    # ── Full cinematic light transport budget ─────────────────────────────────
    scene.cycles.max_bounces             = 24
    scene.cycles.diffuse_bounces         = 8
    scene.cycles.glossy_bounces          = 8
    scene.cycles.transmission_bounces    = 12
    scene.cycles.volume_bounces          = 6
    scene.cycles.transparent_max_bounces = 24
    scene.cycles.blur_glossy             = 0.1   # tame fireflies without crushing caustics

    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = 'PNG'

    # ── World: deep near-black + subtle warm volumetric haze ──────────────────
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    wt = scene.world.node_tree
    wt.nodes.clear()
    wout = wt.nodes.new('ShaderNodeOutputWorld')

    bg = wt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value    = (0.003, 0.003, 0.004, 1)  # near-void — just enough to avoid pure crush
    bg.inputs['Strength'].default_value = 0.03

    wt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # ── AgX color management ──────────────────────────────────────────────────
    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        try:
            scene.view_settings.look = 'AgX - High Contrast'
        except Exception:
            pass
    scene.view_settings.exposure = -0.2  # darkness from scene lighting, not global exposure clamp

    scene.unit_settings.system = 'METRIC'


def setup_compositor(scene):
    """Clean pass-through compositor — no bloom (adds clipping artifacts on translucent organs)."""
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
    """Disable compositor for simple/EEVEE renders."""
    try:
        scene.use_nodes = False
    except Exception:
        pass


# ── Negative fill planes ──────────────────────────────────────────────────────

def add_negative_fill_planes(cx, cy, cz, scene_scale):
    """
    Black absorption planes flanking the anatomy.
    Prevents light from wrapping around and washing out shadow contrast.
    Must be placed off to the sides — not in the camera's field of view.
    """
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

    # Flank the anatomy at sc*2.2 — absorbs bounce light without intercepting light sources
    make_plane((cx - sc*2.2, cy + sc*0.2, cz), (0, math.pi/2, 0), "NegFill_Left")
    make_plane((cx + sc*2.2, cy + sc*0.2, cz), (0, math.pi/2, 0), "NegFill_Right")


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

    # ── AO contact shadows — darkens creases/crevices between organs ──────────
    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 0.008   # 8mm crease darkening radius
    mix_ao = nodes.new('ShaderNodeMix')
    mix_ao.data_type  = 'RGBA'
    mix_ao.blend_type = 'MULTIPLY'
    mix_ao.inputs['Factor'].default_value = 0.20       # 20% AO darkening — subtle shadow stamp
    mix_ao.inputs[6].default_value = (*base_rgb, 1.0)  # Color A (index 6 in RGBA mode)
    links.new(ao_node.outputs['Color'], mix_ao.inputs[7])  # Color B = AO output (white=open, dark=occluded)
    links.new(mix_ao.outputs[2], principled.inputs['Base Color'])  # Result → Base Color

    # ── UV + shared mapping ───────────────────────────────────────────────────
    uv_node  = nodes.new('ShaderNodeTexCoord')
    map_node = nodes.new('ShaderNodeMapping')
    map_node.inputs['Scale'].default_value = (3.0, 3.0, 3.0)
    links.new(uv_node.outputs['UV'], map_node.inputs['Vector'])

    # ── Roughness with noise variation ────────────────────────────────────────
    noise_r = nodes.new('ShaderNodeTexNoise')
    noise_r.inputs['Scale'].default_value      = 20.0  # coherent organic waves — no micro-glitter
    noise_r.inputs['Detail'].default_value     = 8.0   # high detail contrast within each wave
    noise_r.inputs['Roughness'].default_value  = 0.60
    noise_r.inputs['Distortion'].default_value = 0.15
    links.new(map_node.outputs['Vector'], noise_r.inputs['Vector'])

    val_map = nodes.new('ShaderNodeMapRange')
    val_map.inputs['From Min'].default_value = 0.0
    val_map.inputs['From Max'].default_value = 1.0
    val_map.inputs['To Min'].default_value   = roughness * 0.2   # smooth base under wet coat
    val_map.inputs['To Max'].default_value   = roughness * 0.85
    links.new(noise_r.outputs['Factor'], val_map.inputs['Value'])
    links.new(val_map.outputs['Result'], principled.inputs['Roughness'])

    # ── Random Walk SSS ───────────────────────────────────────────────────────
    if sss_weight > 0:
        principled.subsurface_method = 'RANDOM_WALK'
        principled.inputs['Subsurface Weight'].default_value = sss_weight
        principled.inputs['Subsurface Scale'].default_value  = sss_scale_mm / 1000.0 * 0.005
        principled.inputs['Subsurface Radius'].default_value = sss_radius

    # ── Coat: peritoneal fluid / Glisson capsule ──────────────────────────────
    if coat_weight > 0:
        principled.inputs['Coat Weight'].default_value    = 1.0
        principled.inputs['Coat Roughness'].default_value = 0.015   # razor-sharp glassy reflection
        principled.inputs['Coat IOR'].default_value       = 1.41

    principled.inputs['Specular IOR Level'].default_value = 0.5

    # ── Bevel shading + bump map ──────────────────────────────────────────────
    # Bevel node softens shading transition at edges → expensive-CG edge specular catch
    bevel_node = nodes.new('ShaderNodeBevel')
    bevel_node.samples = 6
    bevel_node.inputs['Radius'].default_value = 0.00035  # 0.35mm micro-bevel

    if bump_type != "none" and bump_scale > 0:
        bump_path = TEX_DIR / f"bump_{bump_type}.png"
        if bump_path.exists():
            map_b = nodes.new('ShaderNodeMapping')
            map_b.inputs['Scale'].default_value = (3.0, 3.0, 3.0)
            links.new(uv_node.outputs['UV'], map_b.inputs['Vector'])

            tex_b = nodes.new('ShaderNodeTexImage')
            tex_b.image = bpy.data.images.load(str(bump_path))
            tex_b.image.colorspace_settings.name = 'Non-Color'
            tex_b.extension = 'REPEAT'
            links.new(map_b.outputs['Vector'], tex_b.inputs['Vector'])

            bump_n = nodes.new('ShaderNodeBump')
            bump_n.inputs['Strength'].default_value = bump_scale
            bump_n.inputs['Distance'].default_value = 0.00012
            links.new(bevel_node.outputs['Normal'], bump_n.inputs['Normal'])  # bevel feeds into bump
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
        global_scale=0.001,   # OBJs in mm → meters
    )
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    return new_objs[0] if new_objs else None


# ── Lighting ──────────────────────────────────────────────────────────────────

def setup_lights(cx, cy, cz, scene_scale):
    sc = scene_scale

    # Key: grazing incidence — broader for clean cinematic falloff (no noisy breakup without denoiser)
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*1.4, cy + sc*0.3, cz + sc*0.8))
    key = bpy.context.object
    key.data.energy = 135
    key.data.color  = (0.94, 0.96, 1.00)
    key.data.size   = sc * 0.12  # broader = smooth Alexa-style falloff, not harsh CG spotlight
    key.data.shape  = 'SQUARE'
    _track_to(key, (cx, cy, cz))

    # Fill: emergency recovery only — cavities must fall into shadow
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*1.0, cy - sc*1.0, cz + sc*0.5))
    fill = bpy.context.object
    fill.data.energy = 1.5
    fill.data.color  = (0.60, 0.75, 1.00)
    fill.data.size   = sc * 0.50
    _track_to(fill, (cx, cy, cz))

    # Rim 1: warm amber — traces anterior silhouette
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.5, cy + sc*1.4, cz + sc*0.3))
    rim1 = bpy.context.object
    rim1.data.energy = 150
    rim1.data.color  = (1.00, 0.50, 0.25)  # fiery amber
    rim1.data.size   = sc * 0.02
    _track_to(rim1, (cx, cy, cz))

    # Rim 2: cool kicker from opposite side — peels dark organs away from void
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*0.4, cy - sc*1.4, cz - sc*0.2))
    rim2 = bpy.context.object
    rim2.data.energy = 90
    rim2.data.color  = (0.40, 0.70, 1.00)  # clean cyan/blue
    rim2.data.size   = sc * 0.02
    _track_to(rim2, (cx, cy, cz))


def _track_to(obj, target_xyz):
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


# ── Camera ────────────────────────────────────────────────────────────────────

def setup_camera(size, fov_deg=20):  # 20° = cinematic lens compression
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam_obj = bpy.context.scene.camera = bpy.data.objects['Camera']
    cam_obj.data.type      = 'PERSP'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle     = math.radians(fov_deg)
    cam_obj.data.clip_start = 0.001
    cam_obj.data.clip_end   = 100.0
    # DoF: f/4.5 — midground sharp, subtle fore/background bokeh
    cam_obj.data.dof.use_dof        = True
    cam_obj.data.dof.aperture_fstop = 4.5
    # Asymmetric shift — breaks clinical centered composition
    cam_obj.data.shift_x = -0.05
    cam_obj.data.shift_y =  0.02
    return cam_obj


def point_camera(cam_obj, position, target):
    cam_obj.location = position
    _track_to(cam_obj, target)
    # Auto-focus: distance to scene centre so organs at centre are sharp
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(position, target)))
    cam_obj.data.dof.focus_distance = dist
    bpy.context.view_layer.update()


# ── Simple render (flat EEVEE, no compositor, no DoF) ────────────────────────

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
    cam_obj.data.dof.use_dof = False   # no DoF for simple reference render
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 4
    for obj, simple_mat, _ in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(simple_mat)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    # Restore
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
        print("Run v7 first: python scripts/render_pair_totalseg_v7_bumpmap.py"
              f" --subject {args.subject} --spp 1 --angles 1")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"[v11-cinematic]  subject={args.subject}  spp={args.spp}  size={args.size}px")
    print(f"SSS + VolumeAbsorption + DoF + NegFill + Volumetrics + Bloom")
    print(f"{'='*65}")

    # ── Load CT metadata for camera/scene scale ───────────────────────────────
    import nibabel as nib
    ct_img   = nib.load(str(Path(args.dataset) / args.subject / "ct.nii.gz"))
    shape    = ct_img.shape[:3]
    zooms    = ct_img.header.get_zooms()[:3]
    nx,ny,nz = shape
    sx,sy,sz = zooms
    # All positions in meters (OBJs imported with global_scale=0.001)
    cx, cy, cz  = nx*sx/2/1000, ny*sy/2/1000, nz*sz/2/1000
    radius      = max(nx*sx, ny*sy, nz*sz) * 0.9 / 1000
    scene_scale = max(nx*sx, ny*sy, nz*sz) / 1000

    # ── Camera angle sweep ────────────────────────────────────────────────────
    if args.angles == 1:
        offsets = [-20]
    elif args.angles == 2:
        offsets = [-55, 15]
    else:
        offsets = [-55, -20, 15]  # asymmetric — cinematic, not clinical grid

    def cam_pos_at_angle(theta_deg):
        t      = math.radians(theta_deg)
        dx_rel = radius * 0.6
        dy_rel = radius * 1.15    # +Y = posterior side → camera looks anteriorly at organs
        return [cx + dx_rel*math.cos(t) - dy_rel*math.sin(t),
                cy + dx_rel*math.sin(t) + dy_rel*math.cos(t),
                cz + radius*0.45]

    # ── Build scene ───────────────────────────────────────────────────────────
    reset_scene()
    setup_render(args.spp, args.size, args.device)
    cam_obj = setup_camera(args.size)
    setup_lights(cx, cy, cz, scene_scale)
    add_negative_fill_planes(cx, cy, cz, scene_scale)
    setup_compositor(bpy.context.scene)

    # ── Import meshes and assign materials ────────────────────────────────────
    print("\n[1/3] Importing meshes ...")
    objs_with_mats = []
    loaded = 0

    for row in TISSUES:
        (seg_name, simple_hex, base_rgb, roughness, ior,
         sss_weight, sss_scale_mm, sss_radius,
         coat_weight, coat_roughness,
         bump_type, bump_scale, uv_mode) = row

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

    # ── Render each angle ─────────────────────────────────────────────────────
    print("\n[2/3] Rendering angles ...")
    angle_rows = []

    for theta in offsets:
        label = f"{theta:+.0f}°"
        print(f"\n--- Angle {label} ---")
        cam_position = cam_pos_at_angle(theta)
        point_camera(cam_obj, cam_position, (cx, cy, cz))

        # Simple (EEVEE, no compositor, no DoF)
        simple_path = out_dir / f"simple_v12_{label}.png"
        render_simple(objs_with_mats, cam_obj, simple_path, args.size)
        restore_gt_materials(objs_with_mats)
        print(f"  Simple → {simple_path.name}")

        # GT (Cycles, full cinematic pipeline)
        gt_path = out_dir / f"gt_v12_spp{args.spp}_{label}.png"
        bpy.context.scene.render.filepath = str(gt_path)
        bpy.ops.render.render(write_still=True)
        print(f"  GT     → {gt_path.name}")

        angle_rows.append((simple_path, gt_path, label))

    # ── Assemble paired grid ──────────────────────────────────────────────────
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

    grid_path = pair_out / f"{args.subject}_v12_cinematic_spp{args.spp}.png"
    save_numpy_as_png(grid, grid_path)
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
