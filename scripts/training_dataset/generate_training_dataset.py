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

# render_pass() re-enables the compositor after a G-buffer pass but has no access to
# main()'s feature dict. The dataset is always generated training-safe (no chromatic
# aberration or glare — both screen-space and positional, so a convolutional generator
# cannot learn them and they act as label noise), so a module-level constant is correct
# here and cannot drift from the per-run flags.
_FEAT_FOR_PASSES = {"training_safe": True, "env": True}

import bpy
import sys
import os
import math
import json
import argparse
import importlib.util
import numpy as np
from pathlib import Path

# ── Shading comes from the render script, NOT a copy of it ───────────────────
# This file used to duplicate v20's tissue table, materials, lighting and world.
# That is how the dataset silently stayed on v20 shading while the pair renderer
# advanced to v25: two copies of the same logic, only one of them maintained.
# Importing the render script makes it the single source of truth, so the dataset
# and the pair preview can never disagree again.
#
# Its main() is guarded by __name__, so importing runs only the module-level table
# construction (TISSUES, REFERENCE_LOOK, apply_reference_look) — which is what we want.
_V_PATH = Path(__file__).resolve().parent.parent / "render_pair_totalseg_v25_textured.py"
_spec = importlib.util.spec_from_file_location("v25", _V_PATH)
V = importlib.util.module_from_spec(_spec)
_argv_backup = sys.argv
sys.argv = ["v25"]                       # v25 parses argv only inside main(); be safe
_spec.loader.exec_module(V)
sys.argv = _argv_backup


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
    # 0.8 validated on s1340; raise toward 0.85 for tighter, lower for more margin.
    ap.add_argument("--fill",   type=float, default=0.8)
    # Reference organ scale (metres) at which the base light energies are correct.
    # Exposure is held constant across organ sizes via energy ∝ (scene_scale/ref)².
    # Lower → brighter; raise if renders are overexposed, lower if too dark.
    # Reference scale for v25's size-invariant exposure. v25 recalibrated this from
    # 0.40 to 0.78 m: the 0.40 constant was set against the CT extent, and once framing
    # moved to the (smaller) organ extent the lights sat closer AND got scaled up,
    # compounding to roughly 4x too bright.
    ap.add_argument("--light-ref", type=float, default=0.78,
                    help="UNUSED since the v25 port. v25.setup_lights does its own "
                         "size-invariant exposure normalised at 0.78 m internally, so "
                         "this no longer affects anything. Kept only so existing "
                         "run scripts that pass it do not break.")
    # ── v25 shading controls (forwarded to the imported render script) ──
    ap.add_argument("--tex_dir",    default="data/renders/textures/tissue",
                    help="synthesised PBR maps from synth_tissue_textures.py")
    ap.add_argument("--tex_mm",     type=float, default=80.0,
                    help="real-world tile size of those maps; MUST match the --tex_mm "
                         "used to generate them")
    ap.add_argument("--tex_tint",   type=float, default=0.0)
    ap.add_argument("--saturation", type=float, default=1.05)
    ap.add_argument("--exposure",   type=float, default=-1.00)
    ap.add_argument("--sss_scale",  type=float, default=0.90)
    ap.add_argument("--albedo",     type=float, default=1.00)
    ap.add_argument("--smooth",     type=int,   default=0,
                    help="render-time Laplacian smoothing. 0 is correct once meshes are "
                         "re-extracted with the float32 + relaxation 0.3 fix; raise only "
                         "if the marching-cubes staircase reappears.")
    ap.add_argument("--smooth_factor", type=float, default=0.5)
    # Read by v25.setup_render / setup_camera.
    ap.add_argument("--fstop",   type=float, default=11.0,
                    help="v20 used 6.3, which defocused most of the field")
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--look",    default="AgX - Medium Contrast")
    ap.add_argument("--walls", action="store_true",
                    help="negative-fill side planes. Off by default in v25 — with the "
                         "organ-bbox framing they sit in shot and light up as a backdrop.")
    ap.add_argument("--no-vessels",   dest="vessels",   action="store_false")
    ap.add_argument("--no-perfusion", dest="perfusion", action="store_false")
    ap.add_argument("--no-micro",     dest="micro",     action="store_false")
    return ap.parse_args(argv)


def v25_features(args):
    """Feature dict in the shape v25.make_material()/setup_world()/setup_lights() expect."""
    return dict(
        sss_fix=True, micro=args.micro, vessels=args.vessels, perfusion=args.perfusion,
        env=True, denoise=True, tone_fix=True, detail=1.0, vessel_gain=1.0,
        saturation=args.saturation, sss_method="RANDOM_WALK",
        training_safe=True,          # no chromatic aberration / glare: screen-space and
                                     # positional, so a conv generator cannot learn them
        legacy_bump=False,
        tex_dir=args.tex_dir, tex_tint=args.tex_tint, tex_mm=args.tex_mm,
    )


# Tissue definitions, materials, lighting, camera and compositor all live in
# render_pair_totalseg_v25_textured.py and are used via the `V` import above.
# They were duplicated here until an audit found eight diverged render settings —
# denoising, three bounce counts, blur_glossy, the pixel filter, firefly clamp and
# aperture — so the dataset was being generated with v20 shading while the previews
# used v25. Deleting the copies is what stops that recurring.

TEX_DIR = Path("data/renders/textures")


# ── Scene helpers — identical to v20 ─────────────────────────────────────────

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def setup_render(args, feat):
    """Cycles/tone setup delegated to v25, then the dataset's own G-buffer passes.

    This used to be a COPY of v20's setup_render, and an audit against v25 showed eight
    shading-critical settings had silently diverged:

        denoising            OFF          -> ON (OIDN, ACCURATE, albedo+normal guides)
        diffuse_bounces      2            -> 4    (inter-organ red bounce)
        transmission_bounces 6            -> 8
        volume_bounces       1            -> 2
        blur_glossy          0.2          -> 0.02 (0.2 smears the wet-film glints)
        pixel filter         BOX 0.5      -> BLACKMAN_HARRIS 1.5
        sample_clamp_indirect absent      -> 4.0  (firefly control)
        aperture             f/6.3        -> f/11

    Any one of those makes the dataset differ from the previews the look was tuned on,
    so this now CALLS v25 rather than re-stating it. Only the passes below are
    dataset-specific: v25 renders a single beauty image, the dataset also needs the
    depth / normal / index side-channels.
    """
    V.setup_render(args, feat)
    scene = bpy.context.scene

    # Enable ONLY the G-buffer passes the training pipeline conditions on. These land
    # in render.exr (OPEN_EXR_MULTILAYER) at zero extra render cost. Disabling a pass
    # does NOT change the rendered image — passes are side-channels, not part of the
    # shading. The diffuse/glossy/volume light-DECOMPOSITION passes are intentionally
    # OFF: they aren't G-buffers, the architecture never uses them, and they would
    # ~2–3× the EXR size across 24k frames. Re-enable per-experiment if ever needed.
    vl = scene.view_layers[0]
    vl.use_pass_z            = True   # metric depth (metres)        — G-buffer
    vl.use_pass_normal       = True   # world-space normals [-1,1]   — G-buffer
    vl.use_pass_object_index = True   # tissue IDs (obj.pass_index)  — G-buffer
    vl.use_pass_diffuse_direct   = False
    vl.use_pass_diffuse_indirect = False
    vl.use_pass_diffuse_color    = False
    vl.use_pass_glossy_direct    = False
    vl.use_pass_glossy_indirect  = False
    try:
        vl.use_pass_volume_direct   = False
        vl.use_pass_volume_indirect = False
    except AttributeError:
        pass  # not present in Blender 5.x


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

    V.teardown_compositor(scene)
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
    V.setup_compositor(scene, _FEAT_FOR_PASSES)


# ── Negative fill planes — identical to v20 ───────────────────────────────────

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

def _track_to(obj, target_xyz):
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


# ── Camera — identical to v20 ────────────────────────────────────────────────

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

    feat = v25_features(args)
    reset_scene()
    setup_render(args, feat)
    cam_obj = V.setup_camera(args.fstop)
    V.setup_compositor(bpy.context.scene, feat)
    # NOTE: lights + negative-fill planes are placed AFTER the meshes load, so they
    # can be framed on the organ bounding box rather than the CT volume (see below).

    print("\n[1/3] Loading meshes and materials...")

    # v25's per-organ appearance table is absolute; --albedo / --sss_scale are global
    # corrections applied the same way the pair renderer applies them.
    for _t in V.TISSUES:
        _t["base"]   = [c * args.albedo for c in _t["base"]]
        _t["sss_mm"] = _t["sss_mm"] * args.sss_scale

    # PASS 1 — geometry only. Materials cannot be built yet: v25 caps its procedural
    # noise octaves against the image-plane sampling rate (mm/px), and that depends on
    # the camera distance, which depends on the organ bounding box, which needs the
    # meshes loaded. Building materials first would silently use MM_PER_PX = 0 and
    # disable the Nyquist guard that stops sub-pixel octaves aliasing into moire.
    loaded_pairs = []          # (blender_obj, tissue_dict)
    tissue_id_map = {}
    loaded = 0
    for t in V.TISSUES:
        seg_name = t["name"]
        obj_path = mesh_dir / f"{seg_name}_uv.obj"
        if not obj_path.exists():
            obj_path = mesh_dir / f"{seg_name}.obj"
        if not obj_path.exists():
            continue
        blender_obj = V.import_obj(obj_path)
        if blender_obj is None:
            continue
        if args.smooth > 0:
            V.destaircase(blender_obj, args.smooth, args.smooth_factor)
        blender_obj.pass_index = loaded + 1   # 1-based; 0 = background in segid EXR
        loaded_pairs.append((blender_obj, t))
        tissue_id_map[seg_name] = loaded + 1
        loaded += 1
    print(f"  {loaded} tissues loaded")

    # ── Frame the camera/lights on the ORGAN bounding box (not the CT volume) ──
    # The CT-volume framing doesn't transfer across subjects (whole-body scans,
    # off-centre / sparse organ sets -> tiny or clipped shots). Deriving the orbit
    # centre + radius + scale from the actual loaded meshes makes framing consistent
    # for every subject. Falls back to the CT estimate if nothing loaded.
    if loaded > 0:
        (cx, cy, cz), radius, extent = organ_bbox_world(
            [(o, None, None) for o, _ in loaded_pairs])
        scene_scale = float(extent)
        print(f"  organ bbox: center=({cx:.3f},{cy:.3f},{cz:.3f}) "
              f"sphere_r={radius:.3f} extent={extent:.3f}")
    else:
        print("  [warn] no meshes loaded — falling back to CT-volume framing")

    # Sampling rate at the subject, from the real camera geometry. v25's _safe_detail
    # reads this global to cap octaves so none lands below ~2 px.
    orbit_dist = radius / math.sin(0.5 * args.fill * math.radians(20.0))
    _frame_m = 2.0 * orbit_dist * math.tan(math.radians(20.0) / 2.0)
    V.MM_PER_PX = (_frame_m * 1000.0) / float(args.size)
    print(f"  sampling  : {V.MM_PER_PX:.3f} mm/px "
          f"(frame {_frame_m*100:.1f} cm at {args.size} px)")

    # PASS 2 — materials, now that MM_PER_PX is known.
    objs_with_mats = []
    n_tex = 0
    for blender_obj, t in loaded_pairs:
        simple_mat = V.setup_simple_material(t["name"], t["hex"])
        gt_mat     = V.make_material(t, feat)
        blender_obj.data.materials.clear()
        blender_obj.data.materials.append(gt_mat)
        objs_with_mats.append((blender_obj, simple_mat, gt_mat))
        if "albedo" in V.tissue_textures(t["name"], args.tex_dir):
            n_tex += 1
    print(f"  {n_tex}/{loaded} organs using synthesised textures from {args.tex_dir}")
    if n_tex == 0:
        print("  [warn] NO textures found — run scripts/synth_tissue_textures.py first, "
              "or the dataset falls back to procedural shading and will not match v25")

    # Lights: proportional rig at organ scale, energy-compensated so EXPOSURE is
    # constant regardless of organ size (irradiance held fixed). Tune with --light-ref.
    # v25's rig: same three-point geometry, but a much stronger fill and an added
    # warm cavity bounce. That soft wraparound is what reads as translucency in the
    # reference renders — a backlit test showed liver is opaque at any plausible SSS
    # radius, so the effect is lighting, not subsurface transport.
    V.setup_lights(cx, cy, cz, scene_scale, feat, 90.0)
    V.setup_world(bpy.context.scene, feat)
    # Negative-fill planes MUST sit outside the camera orbit, else they occlude the
    # organs at some azimuths (the "fully black" views). Frame them on the orbit radius
    # (max camera distance, at the widest FOV), not the small organ scale.
    if args.walls:
        V.add_fill_planes(cx, cy, cz, scene_scale, feat, cam_radius=orbit_dist)

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
    # Jitter is RELATIVE to the base exposure. Assigning subj_exposure alone was
    # harmless while the base was 0.0, but v25 renders at -1.0 EV — overwriting it
    # here would silently discard that and produce a dataset ~1 stop brighter than
    # every preview we tuned against.
    bpy.context.scene.view_settings.exposure = args.exposure + subj_exposure

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

            V.point_camera(cam_obj, cam_pos, target_pos)
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
            V.render_simple(objs_with_mats, cam_obj, seg_path, feat)
            V.restore_gt_materials(objs_with_mats)
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
            # render.exr = plain single-layer OPEN_EXR holding the scene-linear combined
            # RGBA (the GT image). NOTE: this Blender (5.x/6.0) removed OPEN_EXR_MULTILAYER
            # from the format enum, so the G-buffers are written as SEPARATE files below
            # via the EEVEE pass materials (version-independent, no compositor needed).
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

            # --- G-buffer passes → separate single-layer EXRs (depth/normals/segid) ---
            # Fast EEVEE emission renders from the same camera. render_pass restores the
            # engine/format afterwards; we restore the GT materials before the next view.
            render_pass(objs_with_mats, make_depth_material(),   cam_obj, view_dir / "depth.exr",   fmt='OPEN_EXR')
            render_pass(objs_with_mats, make_normals_material(), cam_obj, view_dir / "normals.exr", fmt='OPEN_EXR')
            render_pass(objs_with_mats, make_segid_material(),   cam_obj, view_dir / "segid.exr",   fmt='OPEN_EXR')
            V.restore_gt_materials(objs_with_mats)
            print(f"  gbuffers → depth.exr normals.exr segid.exr")

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
