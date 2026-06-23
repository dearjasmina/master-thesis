"""
generate_training_dataset.py — Production ML Dataset Generator (v20 quality)

Generates 4-channel training pairs per subject using the v20 cinematic renderer.
All rendering code is identical to v20 — only the camera loop and output structure differ.

Output per sample:
  render.exr     Multi-pass EXR — 11 passes scene-linear (Image, Depth, Normal,
                 IndexOB, DiffDir, DiffInd, DiffCol, GlossDir, GlossInd,
                 VolumeDir, VolumeInd)
  rgb_preview.png AgX tone-mapped PNG — human inspection only, do not train on
  seg.png        flat semantic segmentation (EEVEE, colour-coded by tissue)
  meta.json      camera K + extrinsics, ray fields, subject photometric constants

Camera manifold: 20 full-orbit views — azimuth [0,72,144,216,288] × elevation [-50,0,30,60]
Full 360° azimuth coverage at 4 elevations (bottom-up/equatorial/mid/overhead).
Per-subject deterministic jitter (seeded by MD5 of subject name):
  Geometric (per-view):   azimuth ±2.5°, elevation ±2.0°, distance ±6%, FOV 20–24°
  Photometric (per-subj): key energy ±5%, color temperature ±1%, exposure ±0.04
  Photometric constants are identical across all 15 views for multi-view consistency.

Output structure:
  {output_dir}/
    {subject}_v{N:02d}_az{az:+.0f}_el{el:+.0f}/
      rgb.png  seg.png  depth.png  normals.png  meta.json
    {subject}_summary.json

Dataset split by subject prefix: collect all sample dirs, group by first token of folder
name (e.g. "s0050"), then split groups 80/10/10.

Run:
    blender --background --python scripts/generate_training_dataset.py -- \\
        --subject s0050 --spp 256 --size 512 --device GPU
"""

import bpy
import sys
import os
import math
import json
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
    ap.add_argument("--subject",    default="s0050")
    ap.add_argument("--dataset",
        default="/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
    ap.add_argument("--output_dir", default="data/training_dataset")
    ap.add_argument("--spp",    type=int, default=384)
    ap.add_argument("--size",   type=int, default=1024)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    # Fraction of the frame the organ bounding-sphere should fill (camera framing).
    # 0.7 = prominent with margin; lower = more zoomed out.
    ap.add_argument("--fill",   type=float, default=0.7)
    return ap.parse_args(argv)


# ── Tissue definitions — identical to v20 ────────────────────────────────────
# (name, simple_hex, base_rgb, rough, ior,
#  sss_weight, sss_scale_mm, sss_radius_rgb,
#  coat_weight, coat_roughness, bump_type, bump_scale)
TISSUES = [
    ("autochthon_left",   "#4A3E3D", [0.04, 0.025, 0.02], 0.55, 1.40, 0.01, 1.0, (2.0,1.4,1.0),  0.05, 0.20, "fibrous",  0.15),
    ("autochthon_right",  "#4A3E3D", [0.04, 0.025, 0.02], 0.55, 1.40, 0.01, 1.0, (2.0,1.4,1.0),  0.05, 0.20, "fibrous",  0.15),
    ("lung_lower_lobe_left",  "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_lower_lobe_right", "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_left",  "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_right", "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("vertebrae_T12", "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L1",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L2",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L3",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L4",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L5",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("heart",      "#8A2A2A", [0.13, 0.03, 0.025], 0.35, 1.40, 0.04, 1.8, (1.9,1.4,1.0),  0.22, 0.06, "lobular", 0.35),
    ("esophagus",  "#9E6464", [0.18, 0.09, 0.08],  0.40, 1.40, 0.02, 1.5, (1.5,1.2,1.0),  0.12, 0.10, "vessel",  0.20),
    ("liver",      "#5C2018", [0.10, 0.02, 0.015], 0.24, 1.38, 0.05, 1.8, (1.9,1.4,1.0),  0.25, 0.04, "lobular", 0.40),
    ("stomach",    "#9E916B", [0.22, 0.18, 0.12],  0.34, 1.40, 0.04, 2.0, (1.5,1.1,1.0),  0.18, 0.08, "wrinkled",0.45),
    ("gallbladder","#3A5E35", [0.03, 0.08, 0.02],  0.20, 1.40, 0.05, 1.8, (0.4,2.2,0.6),  0.35, 0.04, "lobular", 0.30),
    ("spleen",     "#523050", [0.09, 0.02, 0.05],  0.25, 1.40, 0.04, 1.8, (1.8,1.3,1.0),  0.25, 0.04, "lobular", 0.40),
    ("kidney_right","#4A1E28", [0.08, 0.02, 0.03], 0.24, 1.42, 0.04, 1.8, (1.7,1.3,1.0),  0.28, 0.05, "lobular", 0.45),
    ("kidney_left", "#4A1E28", [0.08, 0.02, 0.03], 0.24, 1.42, 0.04, 1.8, (1.7,1.3,1.0),  0.28, 0.05, "lobular", 0.45),
    ("pancreas",   "#B09170", [0.24, 0.19, 0.14],  0.42, 1.40, 0.03, 1.5, (1.5,1.2,1.0),  0.15, 0.10, "lobular", 0.40),
    ("duodenum",   "#A38470", [0.21, 0.16, 0.13],  0.38, 1.40, 0.02, 1.8, (1.5,1.2,1.0),  0.18, 0.08, "wrinkled",0.40),
    ("small_bowel","#A38470", [0.21, 0.16, 0.13],  0.38, 1.40, 0.02, 1.8, (1.5,1.2,1.0),  0.18, 0.08, "wrinkled",0.40),
    ("colon",      "#8F6E5C", [0.18, 0.13, 0.10],  0.38, 1.40, 0.02, 1.8, (1.4,1.1,1.0),  0.18, 0.08, "wrinkled",0.40),
    ("urinary_bladder","#6E758A",[0.10,0.11,0.14], 0.32, 1.40, 0.02, 1.5, (1.4,1.2,1.0),  0.20, 0.06, "smooth",  0.20),
    ("aorta",                       "#A31414", [0.28,0.02,0.01], 0.15, 1.38, 0.04, 1.0, (2.5,1.6,1.0), 0.35, 0.03, "vessel", 0.20),
    ("inferior_vena_cava",          "#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.30, 0.03, "vessel", 0.15),
    ("portal_vein_and_splenic_vein","#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.28, 0.03, "vessel", 0.15),
    ("superior_vena_cava",          "#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.30, 0.03, "vessel", 0.15),
]

TEX_DIR = Path("data/renders/textures")


# ── Scene helpers — identical to v20 ─────────────────────────────────────────

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

    if device == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = 'GPU'
        print(f"[GPU] CUDA devices: {[d.name for d in prefs.devices if d.use]}")
    else:
        scene.cycles.device = 'CPU'
    scene.render.film_transparent = False

    scene.cycles.max_bounces             = 12
    scene.cycles.diffuse_bounces         = 2
    scene.cycles.glossy_bounces          = 4
    scene.cycles.transmission_bounces    = 6
    scene.cycles.volume_bounces          = 1
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.blur_glossy             = 0.2

    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width      = 0.5

    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = 'PNG'
    scene.frame_current = 1  # ensures OutputFile uses 0001 suffix consistently

    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    wt = scene.world.node_tree
    wt.nodes.clear()
    wout = wt.nodes.new('ShaderNodeOutputWorld')

    bg = wt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value    = (0.001, 0.001, 0.001, 1)
    bg.inputs['Strength'].default_value = 0.01
    wt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    vol = wt.nodes.new('ShaderNodeVolumeScatter')
    vol.inputs['Color'].default_value     = (0.72, 0.75, 0.82, 1)
    vol.inputs['Density'].default_value   = 0.0015
    vol.inputs['Anisotropy'].default_value = 0.25
    wt.links.new(vol.outputs['Volume'], wout.inputs['Volume'])

    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - Medium Contrast'
    except Exception:
        try:
            scene.view_settings.look = 'AgX - Medium High Contrast'
        except Exception:
            pass
    scene.view_settings.exposure = 0.0
    scene.unit_settings.system = 'METRIC'

    # Enable Cycles render passes — all land in render.exr (OPEN_EXR).
    # Computed at zero extra render cost alongside the GT RGB pass.
    vl = scene.view_layers[0]
    vl.use_pass_z               = True  # metric depth (metres)
    vl.use_pass_normal          = True  # world-space normals [-1,1]
    vl.use_pass_object_index    = True  # tissue IDs (obj.pass_index)
    vl.use_pass_diffuse_direct  = True  # direct illumination
    vl.use_pass_diffuse_indirect = True  # indirect / GI
    vl.use_pass_diffuse_color   = True  # albedo (material colour, no lighting)
    vl.use_pass_glossy_direct   = True  # specular direct
    vl.use_pass_glossy_indirect = True  # specular indirect
    try:
        vl.use_pass_volume_direct   = True  # atmospheric scatter direct
        vl.use_pass_volume_indirect = True  # atmospheric scatter indirect
    except AttributeError:
        pass  # removed in Blender 5.x — VolumeDir/VolumeInd won't appear in EXR


def setup_compositor(scene):
    """
    Applies subtle RGB post-processing (glare + CA) for the preview PNG only.
    Fails silently if scene.node_tree is unavailable (Blender 5.x).
    Training data comes from the scene-linear MULTILAYER EXR, not the PNG.
    """
    try:
        scene.use_nodes = True
        tree = getattr(scene, 'node_tree', None)
        if tree is None:
            return
        tree.nodes.clear()

        rl  = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('CompositorNodeComposite')

        glare = tree.nodes.new('CompositorNodeGlare')
        glare.glare_type = 'FOG_GLOW'
        glare.quality    = 'HIGH'
        glare.threshold  = 0.88
        glare.size       = 5
        glare.mix        = -0.97

        lens = tree.nodes.new('CompositorNodeLensdist')
        lens.inputs['Distortion'].default_value = 0.0
        lens.inputs['Dispersion'].default_value = 0.008

        tree.links.new(rl.outputs['Image'],    glare.inputs['Image'])
        tree.links.new(glare.outputs['Image'], lens.inputs['Image'])
        tree.links.new(lens.outputs['Image'],  out.inputs['Image'])

    except Exception as e:
        print(f"[compositor] warning: {e}")


def teardown_compositor(scene):
    try:
        scene.use_nodes = False
    except Exception:
        pass


# ── Depth and normals pass materials (EEVEE emission, no compositor needed) ───

def make_depth_material():
    """
    Outputs raw camera-space Z depth (metres) via emission → save as EXR.
    No normalisation — training code loads the float EXR and normalises using
    cam_dist / scene_scale from meta.json for cross-view metric consistency.
    Per-frame normalisation (Blender Normalize node) was intentionally avoided
    because it destroys the common metric scale across views and subjects.
    """
    mat = bpy.data.materials.new(name='__depth_pass__')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    cam = nodes.new('ShaderNodeCameraData')
    em  = nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0

    links.new(cam.outputs['View Z Depth'], em.inputs['Color'])
    links.new(em.outputs['Emission'],      out.inputs['Surface'])
    return mat


def make_normals_material():
    """
    World-space surface normals → RGB in [0,1].  Mapping: (N + 1) / 2.
    Background pixels have alpha=0 (film_transparent=True) → black when read as RGB.
    """
    mat = bpy.data.materials.new(name='__normals_pass__')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    geo = nodes.new('ShaderNodeNewGeometry')

    add = nodes.new('ShaderNodeVectorMath')
    add.operation = 'ADD'
    add.inputs[1].default_value = (1.0, 1.0, 1.0)

    mul = nodes.new('ShaderNodeVectorMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)

    em = nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0

    links.new(geo.outputs['Normal'],  add.inputs[0])
    links.new(add.outputs['Vector'],  mul.inputs[0])
    links.new(mul.outputs['Vector'],  em.inputs['Color'])
    links.new(em.outputs['Emission'], out.inputs['Surface'])
    return mat


def make_segid_material():
    """
    Outputs object pass_index as a float via emission → save as EXR.
    Each tissue mesh has obj.pass_index set to a 1-based integer during loading.
    Background pixels = 0.  Training code rounds float values to get class IDs.
    RGB colour-coded seg.png is kept separately for visual inspection.
    """
    mat = bpy.data.materials.new(name='__segid_pass__')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial')
    info = nodes.new('ShaderNodeObjectInfo')
    em   = nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.0

    links.new(info.outputs['Object Index'], em.inputs['Color'])
    links.new(em.outputs['Emission'],       out.inputs['Surface'])
    return mat


def render_pass(objs_with_mats, pass_mat, cam_obj, out_path, fmt='PNG'):
    """
    Render a geometry pass (depth, normals, segid) using EEVEE + emission material.
    fmt='OPEN_EXR' for linear float data (depth, normals, segid).
    fmt='PNG'      for 8-bit previews (not currently used for geo passes).
    EXR saves scene-linear values before any colour management — correct for data.
    """
    scene = bpy.context.scene
    orig_engine      = scene.render.engine
    orig_exposure    = scene.view_settings.exposure
    orig_transparent = scene.render.film_transparent
    orig_dof         = cam_obj.data.dof.use_dof
    orig_fmt         = scene.render.image_settings.file_format
    orig_color_mode  = scene.render.image_settings.color_mode

    teardown_compositor(scene)
    cam_obj.data.dof.use_dof       = False
    scene.render.engine            = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 1
    scene.view_settings.exposure   = 0.0
    scene.render.film_transparent  = (fmt == 'PNG')  # transparent bg only for PNG
    scene.render.image_settings.file_format = fmt
    if fmt == 'OPEN_EXR':
        scene.render.image_settings.color_mode  = 'RGB'
        scene.render.image_settings.color_depth = '32'

    for obj, _, _ in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(pass_mat)

    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)

    scene.render.engine                     = orig_engine
    scene.view_settings.exposure            = orig_exposure
    scene.render.film_transparent           = orig_transparent
    cam_obj.data.dof.use_dof               = orig_dof
    scene.render.image_settings.file_format = orig_fmt
    scene.render.image_settings.color_mode  = orig_color_mode
    setup_compositor(scene)


# ── Negative fill planes — identical to v20 ───────────────────────────────────

def add_negative_fill_planes(cx, cy, cz, scene_scale):
    sc  = scene_scale
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


# ── Material creation — identical to v20 ─────────────────────────────────────

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

    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 0.0025
    mix_ao = nodes.new('ShaderNodeMix')
    mix_ao.data_type  = 'RGBA'
    mix_ao.blend_type = 'MULTIPLY'
    mix_ao.inputs['Factor'].default_value = 0.22
    mix_ao.inputs[6].default_value = (*base_rgb, 1.0)
    links.new(ao_node.outputs['Color'], mix_ao.inputs[7])

    hsv_node = nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Saturation'].default_value = 0.95
    links.new(mix_ao.outputs[2], hsv_node.inputs['Color'])

    uv_node = nodes.new('ShaderNodeTexCoord')

    _vascular = ("liver", "kidney", "spleen", "heart")
    if any(v in seg_name for v in _vascular):
        obj_map = nodes.new('ShaderNodeMapping')
        obj_map.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
        links.new(uv_node.outputs['Object'], obj_map.inputs['Vector'])

        noise_low = nodes.new('ShaderNodeTexNoise')
        noise_low.inputs['Scale'].default_value     = 4.0
        noise_low.inputs['Detail'].default_value    = 2.0
        noise_low.inputs['Roughness'].default_value = 0.5
        links.new(obj_map.outputs['Vector'], noise_low.inputs['Vector'])

        map_low = nodes.new('ShaderNodeMapRange')
        map_low.inputs['From Min'].default_value = 0.0
        map_low.inputs['From Max'].default_value = 1.0
        map_low.inputs['To Min'].default_value   = 0.88
        map_low.inputs['To Max'].default_value   = 1.12
        links.new(noise_low.outputs['Fac'], map_low.inputs['Value'])

        combine_low = nodes.new('ShaderNodeCombineColor')
        links.new(map_low.outputs['Result'], combine_low.inputs['Red'])
        links.new(map_low.outputs['Result'], combine_low.inputs['Green'])
        links.new(map_low.outputs['Result'], combine_low.inputs['Blue'])

        mix_low = nodes.new('ShaderNodeMix')
        mix_low.data_type  = 'RGBA'
        mix_low.blend_type = 'MULTIPLY'
        mix_low.inputs['Factor'].default_value = 1.0
        links.new(hsv_node.outputs['Color'],    mix_low.inputs[6])
        links.new(combine_low.outputs['Color'], mix_low.inputs[7])
        links.new(mix_low.outputs[2], principled.inputs['Base Color'])
    else:
        links.new(hsv_node.outputs['Color'], principled.inputs['Base Color'])

    principled.inputs['Roughness'].default_value = roughness

    _hollow = ("small_bowel", "colon", "duodenum", "stomach", "esophagus")
    if sss_weight > 0 and not any(h in seg_name for h in _hollow):
        principled.subsurface_method = 'RANDOM_WALK'
        principled.inputs['Subsurface Weight'].default_value = sss_weight
        principled.inputs['Subsurface Scale'].default_value  = sss_scale_mm * 0.001
        principled.inputs['Subsurface Radius'].default_value = sss_radius

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
    else:
        links.new(bevel_node.outputs['Normal'], principled.inputs['Normal'])

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ── Mesh import — identical to v20 ───────────────────────────────────────────

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


def organ_bbox_world(objs_with_mats):
    """
    World-space bounding box of all loaded organ meshes.
    Returns (center[3], sphere_radius, max_extent). The camera is framed on THIS
    (the actual organs) instead of the CT volume, so framing is consistent across
    subjects regardless of scan size / organ set / off-centring.
    """
    import mathutils
    bpy.context.view_layer.update()
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for obj, _, _ in objs_with_mats:
        mw = obj.matrix_world
        for corner in obj.bound_box:           # 8 local-space corners
            w = mw @ mathutils.Vector(corner)  # → world space
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
    radius = 0.5 * math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3)))  # bounding sphere
    extent = max(mx[i] - mn[i] for i in range(3))
    return center, radius, extent


# ── Lighting — identical to v20 ───────────────────────────────────────────────

def setup_lights(cx, cy, cz, scene_scale):
    sc = scene_scale

    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*1.4, cy + sc*0.3, cz + sc*0.8))
    key = bpy.context.object
    key.name        = "KeyLight"   # named so jitter logic targets only this light
    key.data.energy = 90
    key.data.color  = (1.00, 0.98, 0.96)
    key.data.size   = sc * 0.25
    key.data.shape  = 'SQUARE'
    _track_to(key, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*1.0, cy - sc*1.0, cz + sc*0.5))
    fill = bpy.context.object
    fill.data.energy = 3.0
    fill.data.color  = (0.96, 0.97, 1.00)
    fill.data.size   = sc * 0.50
    _track_to(fill, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.5, cy + sc*1.4, cz + sc*0.3))
    rim1 = bpy.context.object
    rim1.data.energy = 20
    rim1.data.color  = (1.00, 0.94, 0.88)
    rim1.data.size   = sc * 0.02
    _track_to(rim1, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*0.4, cy - sc*1.4, cz - sc*0.2))
    rim2 = bpy.context.object
    rim2.data.energy = 21
    rim2.data.color  = (0.95, 0.95, 1.00)
    rim2.data.size   = sc * 0.08
    _track_to(rim2, (cx, cy, cz))


def _track_to(obj, target_xyz):
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


# ── Camera — identical to v20 ────────────────────────────────────────────────

def setup_camera(size, fov_deg=18):
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
    cam_obj.data.shift_x = 0.0
    cam_obj.data.shift_y = 0.0
    return cam_obj


def point_camera(cam_obj, position, target):
    cam_obj.location = position
    _track_to(cam_obj, target)
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(position, target)))
    cam_obj.data.dof.focus_distance = dist * 0.94
    bpy.context.view_layer.update()


# ── Simple (segmentation) render — identical to v20 ──────────────────────────

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
    setup_compositor(scene)  # re-enable glare/CA for the GT render that follows


def restore_gt_materials(objs_with_mats):
    for obj, _, gt_mat in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(gt_mat)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    mesh_dir   = Path("data/meshes") / args.subject
    out_root   = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    subj_dir   = out_root / args.subject
    subj_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_dir.exists():
        print(f"ERROR: mesh_dir not found: {mesh_dir}")
        sys.exit(1)

    # CT is now only a FALLBACK for framing — the camera is framed on the organ
    # bounding box once meshes load (see below). So the source CT being absent
    # (e.g. deleted to save space) is fine; we frame purely from the meshes.
    cx = cy = cz = 0.0
    radius = scene_scale = 0.1
    ct_path = Path(args.dataset) / args.subject / "ct.nii.gz"
    if ct_path.exists():
        import nibabel as nib
        ct_img = nib.load(str(ct_path))
        nx, ny, nz = ct_img.shape[:3]
        sx, sy, sz = ct_img.header.get_zooms()[:3]
        cx, cy, cz  = float(nx*sx/2/1000), float(ny*sy/2/1000), float(nz*sz/2/1000)
        radius      = float(max(nx*sx, ny*sy, nz*sz) * 0.9 / 1000)
        scene_scale = float(max(nx*sx, ny*sy, nz*sz) / 1000)
    else:
        print(f"  [info] no CT at {ct_path} — framing purely from organ meshes")

    # 20-view full-orbit manifold — 5 azimuths evenly around 360° × 4 elevations.
    # -50° = true bottom-up; 0° = equatorial; 30° = mid; 60° = overhead.
    azimuths   = [0.0, 72.0, 144.0, 216.0, 288.0]
    elevations = [-50.0, 0.0, 30.0, 60.0]
    n_views    = len(azimuths) * len(elevations)

    print(f"\n{'='*65}")
    print(f"[dataset-gen] subject={args.subject}  spp={args.spp}  size={args.size}px")
    print(f"{n_views}-view full-orbit manifold + per-subject deterministic jitter")
    print(f"Output: {out_root.resolve()}")
    print(f"{'='*65}")

    # Deterministic per-subject seed — same subject always produces the same jitter,
    # so the dataset is reproducible and can be extended with new subjects consistently.
    import hashlib
    seed_int = int(hashlib.md5(args.subject.encode('utf-8')).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed_int)

    reset_scene()
    setup_render(args.spp, args.size, args.device)
    cam_obj = setup_camera(args.size)
    setup_compositor(bpy.context.scene)
    # NOTE: lights + negative-fill planes are placed AFTER the meshes load, so they
    # can be framed on the organ bounding box rather than the CT volume (see below).

    print("\n[1/3] Loading meshes and materials...")
    objs_with_mats = []
    tissue_id_map  = {}  # {seg_name: pass_index (1-based)}
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
        blender_obj.pass_index = loaded + 1  # 1-based; 0 = background in segid EXR
        objs_with_mats.append((blender_obj, simple_mat, gt_mat))
        tissue_id_map[seg_name] = loaded + 1
        loaded += 1
    print(f"  {loaded} tissues loaded")

    # ── Frame the camera/lights on the ORGAN bounding box (not the CT volume) ──
    # The CT-volume framing doesn't transfer across subjects (whole-body scans,
    # off-centre / sparse organ sets → tiny or clipped shots). Deriving the orbit
    # centre + radius + scale from the actual loaded meshes makes framing consistent
    # for every subject. Falls back to the CT estimate if nothing loaded.
    if loaded > 0:
        (cx, cy, cz), radius, extent = organ_bbox_world(objs_with_mats)
        scene_scale = float(extent)
        print(f"  organ bbox: center=({cx:.3f},{cy:.3f},{cz:.3f}) "
              f"sphere_r={radius:.3f} extent={extent:.3f}")
    else:
        print("  [warn] no meshes loaded — falling back to CT-volume framing")

    # Lights + negative-fill placed relative to the organ bbox (moved here from
    # before mesh loading so their scale/position track the organs).
    setup_lights(cx, cy, cz, scene_scale)
    add_negative_fill_planes(cx, cy, cz, scene_scale)

    # Cache baseline light properties — per-view jitter is applied relative to these
    # so accumulated drift across views doesn't happen.
    light_baselines = {}
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            light_baselines[obj.name] = {
                'energy': obj.data.energy,
                'color':  list(obj.data.color),
            }

    # Write the tissue→ID mapping once per subject (training code uses this to
    # convert segid.exr float values to semantic class labels).
    tissue_ids_path = subj_dir / "tissue_ids.json"
    with open(tissue_ids_path, "w") as f:
        json.dump({"background_id": 0, "tissues": tissue_id_map}, f, indent=2)

    print("\n[2/3] Rendering 20-view manifold...")
    generated_samples = []
    view_id = 0

    # Per-subject photometric constants — drawn ONCE before the view loop so that
    # all 15 views share the same lighting conditions.  Multi-view consistency
    # requires photometric variation to be between subjects, not between views.
    subj_key_energy  = float(rng.uniform(0.95, 1.05))
    subj_exposure    = float(rng.uniform(-0.04, 0.04))
    subj_color_drift = {
        name: (float(rng.uniform(0.99, 1.01)), float(rng.uniform(0.99, 1.01)))
        for name in light_baselines
    }
    for name, props in light_baselines.items():
        l_obj = bpy.data.objects.get(name)
        if l_obj and l_obj.type == 'LIGHT':
            if name == "KeyLight":
                l_obj.data.energy = props['energy'] * subj_key_energy
            else:
                l_obj.data.energy = props['energy']
            r_scale, g_scale = subj_color_drift[name]
            l_obj.data.color[0] = props['color'][0] * r_scale
            l_obj.data.color[1] = props['color'][1] * g_scale
            l_obj.data.color[2] = props['color'][2]
    bpy.context.scene.view_settings.exposure = subj_exposure

    for az_nom in azimuths:
        for el_nom in elevations:
            view_id += 1

            # Per-view jitter — prevents network from memorising exact camera positions.
            # Jitter is drawn from the same per-subject RNG so the sequence is
            # reproducible: view 1 of s0050 always has the same jitter.
            az_j   = float(az_nom  + rng.uniform(-2.5,  2.5))
            el_j   = float(el_nom  + rng.uniform(-2.0,  2.0))
            fov_j  = float(rng.uniform(20.0, 24.0))
            # Distance frames the organ bounding sphere to fill ~args.fill of the
            # view at this FOV: a sphere of radius R at distance D subtends a
            # half-angle arcsin(R/D); set that to 0.5*fill*FOV. Robust to subject /
            # organ-set / scan size. Small jitter keeps the camera from being identical.
            base_dist = radius / math.sin(0.5 * args.fill * math.radians(fov_j))
            dist_j = float(base_dist * rng.uniform(0.97, 1.03))

            theta = math.radians(az_j)
            phi   = math.radians(el_j)

            # Camera position — standard spherical coords for a true full orbit.
            # X/Y plane: azimuth rotates around the organ in 360°.
            # Z: elevation lifts the camera above the equator.
            dx = dist_j * math.cos(phi) * math.cos(theta)
            dy = dist_j * math.cos(phi) * math.sin(theta)
            dz = dist_j * math.sin(phi)
            cam_pos = [cx + dx, cy + dy, cz + dz]

            # Small look-at offset keeps organs from being artificially dead-centred
            target_pos = [
                cx + rng.uniform(-scene_scale * 0.02, scene_scale * 0.02),
                cy + rng.uniform(-scene_scale * 0.02, scene_scale * 0.02),
                cz + rng.uniform(-scene_scale * 0.02, scene_scale * 0.02),
            ]

            label    = f"v{view_id:02d}_az{az_nom:+.0f}_el{el_nom:+.0f}"
            view_dir = subj_dir / label
            view_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- View [{view_id:02d}/{n_views}] {label} "
                  f"(az_actual={az_j:.1f}° el_actual={el_j:.1f}°) ---")

            point_camera(cam_obj, cam_pos, target_pos)
            cam_obj.data.angle = math.radians(fov_j)
            cam_dist = float(math.sqrt(sum((a - b)**2 for a, b in zip(cam_pos, target_pos))))

            # Camera intrinsics (K) and extrinsics (cam-to-world 4×4).
            # Convention: Blender −Z forward, Y up.  For OpenCV/COLMAP: flip Y and Z.
            fov_rad  = cam_obj.data.angle
            focal_px = float((args.size / 2) / math.tan(fov_rad / 2))
            half     = float(args.size / 2)
            K_matrix = [[focal_px, 0.0, half],
                        [0.0, focal_px, half],
                        [0.0, 0.0, 1.0]]
            M = cam_obj.matrix_world
            cam_to_world = [[float(M[r][c]) for c in range(4)] for r in range(4)]

            # Camera ray fields — allow training code to reconstruct pixel rays without
            # re-deriving them from K and cam_to_world.
            # Convention: Blender −Z forward, +Y up; rays point from camera into scene.
            R3 = [[float(M[r][c]) for c in range(3)] for r in range(3)]
            principal_ray_world = [float(-M[r][2]) for r in range(3)]  # −Z column of R
            def _corner_ray(px, py):
                # Pixel centre → camera-space direction → world-space normalised.
                dc = [(px - half) / focal_px, -(py - half) / focal_px, -1.0]
                dw = [sum(R3[i][j] * dc[j] for j in range(3)) for i in range(3)]
                n  = math.sqrt(sum(v * v for v in dw))
                return [float(v / n) for v in dw]
            s = float(args.size)
            frustum_corners_world = {
                "top_left":     _corner_ray(0.5,       0.5),
                "top_right":    _corner_ray(s - 0.5,   0.5),
                "bottom_left":  _corner_ray(0.5,       s - 0.5),
                "bottom_right": _corner_ray(s - 0.5,   s - 0.5),
            }

            # --- Render seg (EEVEE) → seg.png ---
            seg_path = view_dir / "seg.png"
            render_simple(objs_with_mats, cam_obj, seg_path, args.size)
            restore_gt_materials(objs_with_mats)
            print(f"  seg   → {seg_path}")

            # --- Render GT (Cycles) → render.exr (MULTILAYER) + rgb_preview.png ---
            #
            # MULTILAYER EXR contains (scene-linear, 32-bit float, no tone mapping):
            #   Image    — combined RGB (training ground truth)
            #   Depth    — metric Z depth (metres)
            #   Normal   — world-space normals [-1,1]
            #   IndexOB  — tissue pass_index (see tissue_ids.json; round to int)
            #   DiffDir  — diffuse direct light
            #   DiffInd  — diffuse indirect / GI
            #   DiffCol  — diffuse albedo (material colour, no lighting)
            #   GlossDir  — specular / glossy direct
            #   GlossInd  — specular / glossy indirect
            #   VolumeDir — volumetric scatter direct (world atmosphere)
            #   VolumeInd — volumetric scatter indirect
            #
            # rgb_preview.png = AgX tone-mapped PNG, for human inspection only.
            # Do NOT train on the PNG — it has baked-in tone mapping.
            scene = bpy.context.scene
            scene.render.image_settings.file_format = 'OPEN_EXR'
            scene.render.image_settings.color_depth = '32'
            exr_path = view_dir / "render.exr"
            scene.render.filepath = str(exr_path)
            bpy.ops.render.render(write_still=True)
            print(f"  render → {exr_path}")

            # Save tone-mapped PNG from the still-live render buffer (no 2nd Cycles render).
            render_img = bpy.data.images.get('Render Result')
            preview_path = view_dir / "rgb_preview.png"
            try:
                if render_img:
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_mode  = 'RGB'
                    render_img.save_render(filepath=str(preview_path), scene=scene)
                    print(f"  preview → {preview_path}")
            except Exception as e:
                print(f"  [warn] preview PNG failed: {e}")

            # Restore PNG as the default format so the seg render on the next view works.
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode  = 'RGB'

            # --- Write per-sample metadata ---
            meta = {
                "subject":           args.subject,
                "view_id":           view_id,
                "azimuth_nominal":   az_nom,
                "elevation_nominal": el_nom,
                "azimuth_actual":    round(az_j, 3),
                "elevation_actual":  round(el_j, 3),
                "camera_pos":        [round(v, 6) for v in cam_pos],
                "target_pos":        [round(v, 6) for v in target_pos],
                "camera_distance":   round(cam_dist, 6),
                "fov_deg":           round(fov_j, 3),
                # Intrinsic K — 3×3, units: pixels.  principal point = (size/2, size/2).
                "K":                 K_matrix,
                # Extrinsic cam-to-world 4×4 (Blender: −Z forward, Y up).
                # To convert to OpenCV convention: flip columns 1 and 2 of rotation part.
                "cam_to_world":      cam_to_world,
                # Ray fields — pre-computed for convenience; derivable from K + cam_to_world.
                # All rays are unit vectors pointing from camera into the scene.
                # Convention: Blender −Z forward, +Y up.
                "ray_origin":              [round(v, 6) for v in cam_pos],
                "principal_ray_world":     [round(v, 6) for v in principal_ray_world],
                "frustum_corners_world":   {k: [round(v, 6) for v in vs]
                                            for k, vs in frustum_corners_world.items()},
                "ray_convention":          "blender_neg_z_forward_y_up",
                # Subject-level photometric constants (identical for all 15 views of this subject).
                "subj_key_energy_factor": round(subj_key_energy, 4),
                "subj_exposure":          round(subj_exposure, 4),
                "organ_center":      [round(cx, 6), round(cy, 6), round(cz, 6)],
                "scene_scale":       round(scene_scale, 6),
                "radius":            round(radius, 6),
                # Reference for normalising the Depth pass in render.exr.
                # Normalise: depth_norm = depth_raw / depth_metric_ref → [0, 1] approx.
                "depth_metric_ref":  round(cam_dist + scene_scale, 6),
                "spp":               args.spp,
                "resolution":        args.size,
                "renderer":          "cycles_v20",
                "files": {
                    "render":      "render.exr",       # MULTILAYER EXR — all passes, scene-linear
                    "rgb_preview": "rgb_preview.png",  # AgX tone-mapped, visualization only
                    "seg":         "seg.png",           # RGB semantic label, conditioning input
                },
                # Channels inside render.exr (read with OpenEXR / imageio / Blender):
                "render_exr_passes": {
                    "Image":    "combined RGB, scene-linear float — use for training",
                    "Depth":    "metric Z depth (metres); normalise with depth_metric_ref",
                    "Normal":   "world-space normals, range [-1,1] per channel",
                    "IndexOB":  "tissue pass_index float; round to int, see tissue_ids.json",
                    "DiffDir":  "diffuse direct illumination",
                    "DiffInd":  "diffuse indirect / global illumination",
                    "DiffCol":  "diffuse albedo (material colour, lighting-free)",
                    "GlossDir":  "specular direct",
                    "GlossInd":  "specular indirect",
                    "VolumeDir": "volumetric scatter direct (world atmosphere)",
                    "VolumeInd": "volumetric scatter indirect",
                },
            }
            with open(view_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            generated_samples.append({
                "sample_dir": label,
                "view_id":    view_id,
                "azimuth_nominal":   az_nom,
                "elevation_nominal": el_nom,
            })

    # --- Subject-level summary ---
    print("\n[3/3] Writing subject summary...")
    summary = {
        "subject":          args.subject,
        "total_views":      len(generated_samples),
        "spp":              args.spp,
        "resolution":       args.size,
        "rng_seed":         seed_int,
        "camera_manifold": {
            "azimuths":   azimuths,
            "elevations": elevations,
        },
        "samples": generated_samples,
    }
    summary_path = subj_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSubject summary → {summary_path}")
    print(f"Generated {len(generated_samples)}/{n_views} samples for {args.subject}.")
    print("Done.")


if __name__ == "__main__":
    main()
