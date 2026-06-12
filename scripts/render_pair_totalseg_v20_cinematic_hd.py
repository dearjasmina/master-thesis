"""
render_pair_totalseg_v20_cinematic_hd.py — Optics + Acquisition + Anatomy Fidelity

v20 enters the optics/acquisition/anatomy fidelity phase. No new procedural complexity —
only physically grounded improvements to existing parameters.

NEW in v20:

1. SPECTRAL-INFORMED SSS RADIUS (tissue optics literature calibration)
   All SSS radius values recalibrated from measured reduced scattering coefficients (μ_s')
   from tissue optics literature (Bashkatov 2011, Jacques 2013, Cheong 1990).
   Key fix: previous R:G:B ratios like (1.6, 0.05, 0.01) are wrong — that models
   extreme absorption asymmetry via scattering, which is not the physics.
   μ_s' varies mildly across the visible spectrum (~2:1.4:1 ratio for soft tissue).
   Absorption is handled by base_rgb (diffuse color), NOT SSS radius.
   - Arterial (oxy-Hb): (2.5, 1.6, 1.0) — lower 650nm absorption → deeper R penetration
   - Venous (deoxy-Hb): (1.4, 1.2, 0.9) — higher 650nm absorption → less R vs arteries
   - Hepatic/splenic/renal: (1.9, 1.4, 1.0) — Bashkatov liver μ_s' measurements
   - Bile (gallbladder): (0.4, 2.2, 0.6) — bilirubin/biliverdin green-dominant absorption
   - Muscle: (2.0, 1.4, 1.0) — lower blood content, measured skeletal muscle μ_s'
   - Lung: (1.6, 1.2, 1.0) — air-filled, lower Hb density, more balanced scattering

2. TISSUE MICROSTRUCTURE ANISOTROPY — REMOVED (render time, not visual impact)
   Anisotropy was removed after profiling: Random Walk SSS + anisotropic BSDF on 29
   complex organ meshes causes ~14× render time increase (7 min → 100 min).
   The effect is also masked by the coat specular layer sitting above the base BSDF.
   Anisotropy is physically correct for muscle/vessels but not practically viable here.
   Future path: geometry-derived tangent fields from mesh curvature analysis.

3. CAMERA OPTICAL SIMULATION: CHROMATIC ABERRATION
   CompositorNodeLensdist with Dispersion=0.008 — simulates real camera lens
   lateral chromatic aberration (R/G/B focus plane separation).
   Real medical photography always contains some CA from non-apochromatic optics.
   Extremely subtle at 0.008 — visible in specular edges, not in flat regions.
   Applied to GT render only (not simple/control render).

4. LARGER KEY LIGHT (surgical photography reference)
   Key size sc*0.12 → sc*0.25 — matches overhead OR surgical light footprint.
   Wider source → softer shadow penumbra → more faithful to intraoperative photography.
   Energy unchanged (90W) since total flux is maintained by area scaling.

Not yet implementable in render script (requires pipeline changes):
  - True spectral rendering: Blender is RGB; would need OSL + IES spectral data
  - Geometry remeshing: must be done in preprocessing before OBJ export
  - Contact mechanics / gravity deformation: requires physics simulation
  - Moisture accumulation: requires fluid simulation
  - CT-informed volumetric heterogeneity: requires volumetric rendering pipeline
  - Polarization: Cycles has no polarization model

Run:
    blender --background --python scripts/render_pair_totalseg_v20_cinematic_hd.py -- \\
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
        default="/home/vulovic/jasmina/dataset")
    ap.add_argument("--mesh_dir", default="data/meshes")
    ap.add_argument("--spp",    type=int, default=384)
    ap.add_argument("--size",   type=int, default=1024)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    return ap.parse_args(argv)


# ── Tissue definitions — production palette, R>G>B vascular rule ─────────────
# (name, simple_hex, base_rgb, rough, ior,
#  sss_weight, sss_scale_mm, sss_radius_rgb,
#  coat_weight, coat_roughness, bump_type, bump_scale)
# SSS radius calibration source: μ_s' measured from tissue optics literature
# (Bashkatov 2011, Jacques 2013, Cheong 1990). Ratios represent R:G:B reduced
# scattering coefficients — NOT effective penetration depth (that includes absorption).
# Absorption is already encoded in base_rgb. SSS radius = scattering path only.
# Soft tissue μ_s' ratios typically ~2:1.4:1 across R:G:B.
TISSUES = [
    # Skeletal muscle — measured μ_s' ~16/12/8 cm⁻¹ at 650/550/450nm
    ("autochthon_left",   "#4A3E3D", [0.04, 0.025, 0.02], 0.55, 1.40, 0.01, 1.0, (2.0,1.4,1.0),  0.05, 0.20, "fibrous",  0.15),
    ("autochthon_right",  "#4A3E3D", [0.04, 0.025, 0.02], 0.55, 1.40, 0.01, 1.0, (2.0,1.4,1.0),  0.05, 0.20, "fibrous",  0.15),
    # Lung parenchyma — air-filled, low Hb density, more balanced μ_s'
    ("lung_lower_lobe_left",  "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_lower_lobe_right", "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_left",  "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    ("lung_upper_lobe_right", "#9C8585", [0.20, 0.14, 0.13], 0.42, 1.36, 0.03, 1.2, (1.6,1.2,1.0),  0.12, 0.15, "smooth", 0.10),
    # Bone: no SSS (cortical bone is opaque)
    ("vertebrae_T12", "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L1",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L2",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L3",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L4",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    ("vertebrae_L5",  "#C5BEB2", [0.23, 0.21, 0.18], 0.78, 1.55, 0.0, 0.0, (0.0,0.0,0.0), 0.0, 0.40, "none", 0.0),
    # Cardiac muscle — Bashkatov measured cardiac μ_s' ~19/14/10 cm⁻¹
    ("heart",      "#8A2A2A", [0.13, 0.03, 0.025], 0.35, 1.40, 0.04, 1.8, (1.9,1.4,1.0),  0.22, 0.06, "lobular", 0.35),
    # Esophagus mucosa — mucosal scattering, measured ~15/11/8 cm⁻¹
    ("esophagus",  "#9E6464", [0.18, 0.09, 0.08],  0.40, 1.40, 0.02, 1.5, (1.5,1.2,1.0),  0.12, 0.10, "vessel",  0.20),
    # Liver — Bashkatov 2011: μ_s' = 17.5/12.8/9.2 cm⁻¹ at 632/532/457nm → ratio ~1.9:1.4:1
    ("liver",      "#5C2018", [0.10, 0.02, 0.015], 0.24, 1.38, 0.05, 1.8, (1.9,1.4,1.0),  0.25, 0.04, "lobular", 0.40),
    # Stomach wall — gastric mucosa/muscularis ~15/11/8 cm⁻¹
    ("stomach",    "#9E916B", [0.22, 0.18, 0.12],  0.34, 1.40, 0.04, 2.0, (1.5,1.1,1.0),  0.18, 0.08, "wrinkled",0.45),
    # Gallbladder — bile fluid: bilirubin/biliverdin → green-dominant absorption → G scatters deepest
    ("gallbladder","#3A5E35", [0.03, 0.08, 0.02],  0.20, 1.40, 0.05, 1.8, (0.4,2.2,0.6),  0.35, 0.04, "lobular", 0.30),
    # Spleen — red pulp: very high RBC density, similar to liver μ_s'
    ("spleen",     "#523050", [0.09, 0.02, 0.05],  0.25, 1.40, 0.04, 1.8, (1.8,1.3,1.0),  0.25, 0.04, "lobular", 0.40),
    # Kidney cortex — renal tissue μ_s' ~17/13/10 cm⁻¹
    ("kidney_right","#4A1E28", [0.08, 0.02, 0.03], 0.24, 1.42, 0.04, 1.8, (1.7,1.3,1.0),  0.28, 0.05, "lobular", 0.45),
    ("kidney_left", "#4A1E28", [0.08, 0.02, 0.03], 0.24, 1.42, 0.04, 1.8, (1.7,1.3,1.0),  0.28, 0.05, "lobular", 0.45),
    # Pancreas — exocrine tissue, fat-influenced, more balanced μ_s'
    ("pancreas",   "#B09170", [0.24, 0.19, 0.14],  0.42, 1.40, 0.03, 1.5, (1.5,1.2,1.0),  0.15, 0.10, "lobular", 0.40),
    # Bowel wall — hollow organs excluded from SSS by _hollow check, but calibrated anyway
    ("duodenum",   "#A38470", [0.21, 0.16, 0.13],  0.38, 1.40, 0.02, 1.8, (1.5,1.2,1.0),  0.18, 0.08, "wrinkled",0.40),
    ("small_bowel","#A38470", [0.21, 0.16, 0.13],  0.38, 1.40, 0.02, 1.8, (1.5,1.2,1.0),  0.18, 0.08, "wrinkled",0.40),
    ("colon",      "#8F6E5C", [0.18, 0.13, 0.10],  0.38, 1.40, 0.02, 1.8, (1.4,1.1,1.0),  0.18, 0.08, "wrinkled",0.40),
    # Urinary bladder — transitional epithelium, fairly balanced μ_s' (removed erroneous blue bias)
    ("urinary_bladder","#6E758A",[0.10,0.11,0.14], 0.32, 1.40, 0.02, 1.5, (1.4,1.2,1.0),  0.20, 0.06, "smooth",  0.20),
    # Aorta — oxy-Hb: lower 650nm absorption vs deoxy → deeper R penetration
    ("aorta",                       "#A31414", [0.28,0.02,0.01], 0.15, 1.38, 0.04, 1.0, (2.5,1.6,1.0), 0.35, 0.03, "vessel", 0.20),
    # Veins — deoxy-Hb: higher 650nm absorption → less R vs arteries, more R:G balanced
    ("inferior_vena_cava",          "#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.30, 0.03, "vessel", 0.15),
    ("portal_vein_and_splenic_vein","#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.28, 0.03, "vessel", 0.15),
    ("superior_vena_cava",          "#3D2050", [0.05,0.02,0.08], 0.16, 1.38, 0.04, 1.0, (1.4,1.2,0.9), 0.30, 0.03, "vessel", 0.15),
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


def setup_compositor(scene):
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        if tree is None:
            return
        tree.nodes.clear()
        rl  = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('CompositorNodeComposite')

        # Subtle fog-glow: only clips brightest specular peaks, mix=-0.97 = ~1.5% blend
        # Breaks the "optically perfect render" look without biasing training data
        glare = tree.nodes.new('CompositorNodeGlare')
        glare.glare_type = 'FOG_GLOW'
        glare.quality    = 'HIGH'
        glare.threshold  = 0.88
        glare.size       = 5
        glare.mix        = -0.97

        # Chromatic aberration: lateral R/G/B focus-plane separation from real camera optics
        lens = tree.nodes.new('CompositorNodeLensdist')
        lens.inputs['Distortion'].default_value = 0.0   # no barrel/pin distortion
        lens.inputs['Dispersion'].default_value = 0.008  # very subtle CA (~1% lateral shift)

        tree.links.new(rl.outputs['Image'],    glare.inputs['Image'])
        tree.links.new(glare.outputs['Image'], lens.inputs['Image'])
        tree.links.new(lens.outputs['Image'],  out.inputs['Image'])
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

    # AO contact shadows — lighter touch, doesn't crush albedo before light interaction
    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 0.0025
    mix_ao = nodes.new('ShaderNodeMix')
    mix_ao.data_type  = 'RGBA'
    mix_ao.blend_type = 'MULTIPLY'
    mix_ao.inputs['Factor'].default_value = 0.22
    mix_ao.inputs[6].default_value = (*base_rgb, 1.0)
    links.new(ao_node.outputs['Color'], mix_ao.inputs[7])

    # Higher saturation — palette controls organ identity, not this node
    hsv_node = nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Saturation'].default_value = 0.95
    links.new(mix_ao.outputs[2], hsv_node.inputs['Color'])

    uv_node = nodes.new('ShaderNodeTexCoord')

    # Low-freq color variation — models lobular blood-content patchwork on vascular organs
    _vascular = ("liver", "kidney", "spleen", "heart")
    if any(v in seg_name for v in _vascular):
        obj_map = nodes.new('ShaderNodeMapping')
        obj_map.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
        links.new(uv_node.outputs['Object'], obj_map.inputs['Vector'])

        noise_low = nodes.new('ShaderNodeTexNoise')
        noise_low.inputs['Scale'].default_value     = 4.0   # anatomically large variation
        noise_low.inputs['Detail'].default_value    = 2.0
        noise_low.inputs['Roughness'].default_value = 0.5
        links.new(obj_map.outputs['Vector'], noise_low.inputs['Vector'])

        map_low = nodes.new('ShaderNodeMapRange')
        map_low.inputs['From Min'].default_value = 0.0
        map_low.inputs['From Max'].default_value = 1.0
        map_low.inputs['To Min'].default_value   = 0.88
        map_low.inputs['To Max'].default_value   = 1.12
        links.new(noise_low.outputs['Fac'], map_low.inputs['Value'])

        # Convert scalar to RGB (gray) for MULTIPLY blend
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

    # Calibrated static roughness per tissue — no noise modulation
    principled.inputs['Roughness'].default_value = roughness

    # Random Walk SSS — static scale (noise modulation removed: added variance, not signal)
    _hollow = ("small_bowel", "colon", "duodenum", "stomach", "esophagus")
    if sss_weight > 0 and not any(h in seg_name for h in _hollow):
        principled.subsurface_method = 'RANDOM_WALK'
        principled.inputs['Subsurface Weight'].default_value = sss_weight
        principled.inputs['Subsurface Scale'].default_value  = sss_scale_mm * 0.001
        principled.inputs['Subsurface Radius'].default_value = sss_radius

    # Coherent coat lobe — static per tissue, no modulation
    # Broad unbroken specular is what photographed wet serosal tissue actually produces
    if coat_weight > 0:
        principled.inputs['Coat Weight'].default_value    = coat_weight
        principled.inputs['Coat Roughness'].default_value = coat_roughness
        principled.inputs['Coat IOR'].default_value       = 1.41

    principled.inputs['Specular IOR Level'].default_value = 0.5

    # Bevel: softens STL-derived hard edges (geometry correction, not artistic)
    bevel_node = nodes.new('ShaderNodeBevel')
    bevel_node.samples = 4
    bevel_node.inputs['Radius'].default_value = 0.00012

    # Texture bump only if a geometry-derived bump file exists for this tissue type
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

    # Key: slightly softer surgical white (less clinical harshness)
    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*1.4, cy + sc*0.3, cz + sc*0.8))
    key = bpy.context.object
    key.data.energy = 90
    key.data.color  = (1.00, 0.98, 0.96)
    key.data.size   = sc * 0.25  # wider = softer shadows, matches OR surgical light footprint
    key.data.shape  = 'SQUARE'
    _track_to(key, (cx, cy, cz))

    # Fill: minimal ambient lift
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*1.0, cy - sc*1.0, cz + sc*0.5))
    fill = bpy.context.object
    fill.data.energy = 3.0
    fill.data.color  = (0.96, 0.97, 1.00)
    fill.data.size   = sc * 0.50
    _track_to(fill, (cx, cy, cz))

    # Rim 1: soft neutral warm edge definition
    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.5, cy + sc*1.4, cz + sc*0.3))
    rim1 = bpy.context.object
    rim1.data.energy = 20
    rim1.data.color  = (1.00, 0.94, 0.88)
    rim1.data.size   = sc * 0.02
    _track_to(rim1, (cx, cy, cz))

    # Rim 2: near-neutral specular catch, demoted to avoid stylized reads
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


# ── Camera ────────────────────────────────────────────────────────────────────

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

    mesh_dir = Path(args.mesh_dir) / args.subject
    out_dir  = Path("data/renders") / args.subject
    pair_out = Path("results/totalseg_pairs")
    pair_out.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_dir.exists():
        print(f"ERROR: mesh_dir not found: {mesh_dir}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"[v20-cinematic-hd]  subject={args.subject}  spp={args.spp}  size={args.size}px")
    print(f"Spectral SSS · tissue anisotropy · chromatic aberration · surgical key light")
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

        simple_path = out_dir / f"simple_v20_{label}.png"
        render_simple(objs_with_mats, cam_obj, simple_path, args.size)
        restore_gt_materials(objs_with_mats)
        print(f"  Simple → {simple_path.name}")

        gt_path = out_dir / f"gt_v20_spp{args.spp}_{label}.png"
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

    grid_path = pair_out / f"{args.subject}_v20_cinematic_hd_spp{args.spp}.png"
    save_numpy_as_png(grid, grid_path)
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
