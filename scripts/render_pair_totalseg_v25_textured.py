"""
render_pair_totalseg_v25_textured.py — framing, palette and lighting pass over v23

v23 got the geometry and the procedural machinery right. v24 addresses what was still
wrong when the whole scene was viewed together.

1. CAMERA FRAMED FROM THE ORGANS, NOT THE SCAN.
   v20-v23 derived camera distance from max(nx*sx, ny*sy, nz*sz) — the CT field of
   view. That varies enormously between subjects (a whole-body scan and a tight
   abdominal scan can contain identical organs), so the same anatomy rendered
   zoomed-out on one subject and cropped on another. v24 imports the meshes first,
   takes the union world bounding box, and frames from that. --frame controls the crop.
   Side effect: nibabel and the CT file are no longer needed at all.

2. COOLER, LESS VIBRANT PALETTE.
   saturation 1.30 -> 1.05, and every base colour darkened/desaturated. The liver in
   particular read as bright red; it is now deep burgundy (0.145, 0.032, 0.028).

3. VESSELS PULLED BACK ON THE GUT.
   small_bowel/duodenum vessel 0.95 -> 0.42 at 9 mm, colon 0.70 -> 0.34 at 13 mm. The
   v23 values read as ink drawn on the surface rather than vasculature under serosa.

4. MATERIALS ACTUALLY DIFFER FROM EACH OTHER.
   Everything rendered as the same wet plastic. The coat range is now wide enough to
   separate them: bowel serosa 0.60 @ 0.030 (wettest), liver 0.62 @ 0.022 (hard wet
   capsule), lung 0.20 @ 0.150 (soft, air-filled, NOT lacquered), pancreas 0.16 @ 0.230
   (matte, lobulated), bone 0.02 @ 0.480 (matte).

5. SIZE-INVARIANT EXPOSURE + SOFT WRAPAROUND LIGHT.
   Light positions always scaled with the scene but energies were absolute watts, and
   irradiance goes as 1/d^2 — so exposure drifted between subjects, and shifted again
   when (1) changed the framing. Energies now scale with scene_scale^2, normalised at
   0.40 m so the defaults are numerically unchanged for a typical torso.

   The fill and environment are also much stronger (env 0.45 -> 0.85, fill 3 W -> 15 W
   at a 2.2x larger source). This is what produces the "translucent" quality of the
   reference renders. It is NOT subsurface transmission: a backlit test renders liver
   as an opaque silhouette at both 0.3x and 0.9x SSS scale, because the transport mean
   free path is ~1-3 mm against a 190 mm organ. What reads as translucency is a soft
   shading terminator, which is a lighting property. Chasing it with SSS radius does
   nothing except blur away the albedo detail.

────────────────────────────────────────────────────────────────────────────────
THE BLOCKER IS THE MESH — verified by rendering, not inferred
────────────────────────────────────────────────────────────────────────────────
The wavy corduroy in v20-v22 is marching-cubes staircase in the GEOMETRY. Proof: the
same liver mesh rendered with a DEFAULT Principled material — no bump, no noise, no
textures, no custom shader of any kind — still shows the ripples. Nothing in this file
can cause an artefact that appears without this file.

extract_meshes.py contours a BINARY mask:

    data = (img.get_fdata() > 0.5).astype(np.uint8)      # 0/1 only
    grid.cell_data_to_point_data().contour(isosurfaces=[0.5])

Iso-surfacing a binary field always staircases; cell_data_to_point_data is only a
2x2x2 box average, and .smooth(n_iter=30, relaxation_factor=0.05) is far too weak to
repair it (Bade et al. use lambda = 0.5).

Two fixes, both measured:

1. STOPGAP, here, no re-extraction: --smooth applies a Laplacian Smooth modifier and
   clears the OBJ's custom split normals first (without that the baked normals keep
   shading the ripple no matter what the vertices do). At 12-20 iterations the ripples
   are visually gone; 50 works but starts melting real anatomical detail.

2. PERMANENT, in extract_meshes.py: Gaussian-blur the mask before contouring, so the
   iso-surface is smooth by construction. Validated on a synthetic ellipsoid where the
   true volume is known:

       current  binary + smooth(30, 0.05)   volume 101.6 %   |meanCurv| p90 0.153
       fixed    gaussian(sigma=1.0) + smooth(20, 0.3)  97.2 %   |meanCurv| p90 0.068

   i.e. 55 % less surface ripple, both within one voxel of the true volume. Side-by-side
   renders of the two are unambiguous. See scripts/training_dataset/extract_meshes_smooth_patch.py.

Three earlier diagnoses of this artefact were WRONG. Recorded so the ground is not
covered twice:
  a) sub-pixel noise octaves (Detail 5 on a 1.0 mm field). A real aliasing bug, and
     _safe_detail still fixes it, but not this artefact.
  b) 8-bit quantisation banding in data/renders/textures/bump_*.png. Those files ARE
     bad — fibrous and vessel are pure linear gradients (plane-fit R^2 0.964 / 0.988)
     that tile into a sawtooth, and the rest have wide quantisation plateaus — which is
     why the legacy UV bump chain is off by default here. But they are not the ripple
     source: the ripple survives with no material at all.
  c) "the mesh is fine, mean dihedral is only 3.4 degrees". Wrong conclusion from a bad
     metric: shading responds to COHERENT normal variation, and a 2-3 degree ripple
     organised into bands is glaring even though its amplitude is sub-pixel.

Methodological note: every one of those wrong calls came from an ad-hoc scalar metric
computed on a noisy render. Render an image and look at it.

────────────────────────────────────────────────────────────────────────────────
REAL TEXTURE MAPS VIA BOX PROJECTION
────────────────────────────────────────────────────────────────────────────────
The reference renders you found beat this one mainly because they use photographic
albedo/normal/roughness maps, not procedural noise. Procedural fields can supply
variation but not the specific structure of real tissue — hepatic capsule vessels,
lung anthracosis, serosal fat streaks all have characteristic shapes noise cannot
invent.

Marching-cubes meshes have no usable UV layout, so v23 uses BOX (triplanar)
projection, which needs none: the texture is projected down all three axes in object
space and blended at the seams. Drop files here and they are picked up automatically:

    data/renders/textures/tissue/{tissue}_albedo.png
                                 {tissue}_normal.png    (optional, Non-Color)
                                 {tissue}_rough.png     (optional, Non-Color)

e.g. liver_albedo.png, lung_upper_lobe_left_albedo.png. Anything missing falls back to
the procedural chain, so this is incremental — texture the liver first and see.
--tex_mm sets the real-world tile size (default 120 mm, i.e. the texture repeats every
12 cm of tissue). Missing textures are reported at load so you can see what was used.

────────────────────────────────────────────────────────────────────────────────
v21 introduced the right machinery with wrong numbers and rendered worse than v20:
banded moire instead of texture, no specular, no visible vessels, and an orange cast.
v22 keeps v21's architecture and fixes the four measured errors. See "WHAT v21 GOT
WRONG" below — that section is the useful part of this file.

v20 reads as painted latex. This version fixes the four things that actually cause that. Everything is a deterministic function of object-space
position, so the render stays reproducible per subject (required by the training pipeline).

────────────────────────────────────────────────────────────────────────────────
ROOT CAUSE 1 — the noise-scale bug (why v16–v19 "proved" procedural detail useless)
────────────────────────────────────────────────────────────────────────────────
OBJs are in millimetres and imported with global_scale=0.001, so object-space
coordinates are in METRES and an organ spans ~0.15–0.35 units.

A Noise Texture with Scale = s produces features of wavelength 1/s object units:

    wavelength_mm = 1000 / scale

v16 used scale 2.5 / 85 and called them "macro / micro". Those are 400 mm and 12 mm
features — i.e. one blob across the whole torso, and a coarse patch. There was never
any micro-detail in the render. v19 then removed all noise on the grounds that it
"added variance, not signal", but what it removed was a low-frequency blotch field,
not surface microstructure. The conclusion was correct about the field that existed
and wrong about the technique.

v21 specifies every procedural feature by physical wavelength in mm via _mm_scale(),
so the numbers in the tissue table are anatomically readable:

    macro lobulation   ~25 mm      capsule wrinkle   ~6 mm
    serosal grain      ~1.6 mm     specular glint    ~1.0 mm

At 1024 px over a ~0.35 m field of view the sampling rate is ~0.34 mm/px, so 1.0 mm
is ~3 px — visible, above Nyquist, and the finest scale worth rendering here.

────────────────────────────────────────────────────────────────────────────────
ROOT CAUSE 2 — Subsurface Weight 0.04–0.05 is opaque paint, not flesh
────────────────────────────────────────────────────────────────────────────────
At weight 0.05 the shader is 95 % Lambertian diffuse. Flesh at these scales is
dominated by subsurface transport — that is the entire difference between "matte
plastic in an organ shape" and meat. v21 raises parenchymal organs to 0.80–0.95 and
adds two parameters v20 never set:

  · Subsurface Anisotropy = 0.75–0.85. Soft tissue has Henyey–Greenstein g ≈ 0.9
    (strongly forward-scattering). Blender defaulted this to 0 = isotropic, which is
    physically wrong and visually flat.
  · Subsurface IOR — attempted, but this socket does NOT exist in Blender 5.x
    (verified against bpy 5.0.1: the Principled inputs are Subsurface Weight/Radius/
    Scale/Anisotropy only). The _set() call is a silent no-op and is kept only so the
    code still works on 4.x, where the socket does exist.

Radii are renormalised so the largest channel is 1.0 and Subsurface Scale carries the
actual transport depth in mm — so the table now reads as physical depth, not a ratio.
Liver δ ≈ 1/sqrt(3·μ_a·(μ_a+μ_s')) ≈ 0.8–1.2 mm at 630 nm (Bashkatov 2011), hence 3 mm
red / 2.2 mm green / 1.6 mm blue with scale 3.0.

────────────────────────────────────────────────────────────────────────────────
ROOT CAUSE 3 — one smeared highlight instead of a broken wet-film specular
────────────────────────────────────────────────────────────────────────────────
Wet serosa produces a sharp near-mirror reflection SHATTERED into hundreds of small
glints by sub-millimetre relief. v20 has the sharpness (coat roughness 0.04) but a
perfectly smooth coat normal, so it produces the plastic-wrap streak visible in
s0050_v20.

v21 drives the Coat Normal with its own high-frequency bump chain, separate from the
base normal. The diffuse/SSS shading stays coherent (v18's correct concern) while the
specular lobe alone is broken up. v17 failed at this because it perturbed at 12 mm
with strength 0.15 — wrong scale, wrong layer.

────────────────────────────────────────────────────────────────────────────────
ROOT CAUSE 4 — AgX desaturation is baked into the training target
────────────────────────────────────────────────────────────────────────────────
scripts/training/dataset.py:203 trains on rgb_preview.png by default, so the view
transform is not a preview concern — it IS the ground truth. AgX's inset transform
desaturates saturated darks hard, which is why base_rgb [0.10, 0.02, 0.015] (deep
maroon) renders as dusty salmon. v21 exposes --look/--view_transform/--saturation and
defaults to AgX Punchy with a 1.25 pre-compensation, and raises base albedos, which
were tuned to survive the old transform.

────────────────────────────────────────────────────────────────────────────────
ALSO NEW
────────────────────────────────────────────────────────────────────────────────
  · Surface vasculature (Voronoi distance-to-edge, two octaves) on parenchymal organs.
    Single strongest "this is real tissue" cue after SSS. Anthracotic mottling on lung.
  · Perfusion field: multi-octave value AND hue variation (congested purple ↔ perfused
    red), replacing v20's invisible ±12 % single-octave value noise.
  · Environment: dim vertical gradient (warm below / cool above) + a warm cavity-bounce
    light, replacing the pure-black void. Black surroundings are why the silhouettes
    read as cut-outs. Negative-fill planes become very dark warm walls that bounce
    instead of absorb.
  · OpenImageDenoise ON with albedo+normal guides and ACCURATE prefilter. At 384 spp
    with heavy SSS the grain is a CG tell; the guided denoiser preserves the micro
    detail this version adds.
  · Blackman-Harris 1.5 reconstruction filter (v20's BOX 0.5 is aliased and harsh),
    f/11 instead of f/6.3, diffuse_bounces 2 → 4 for inter-organ red bounce,
    blur_glossy 0.2 → 0.02 so glints stay crisp.

Every block above is behind a flag so you can bisect what actually helps:
    --no-sss-fix --no-micro --no-vessels --no-env --no-denoise --no-tone-fix

Known deviation left in deliberately: the aorta is still rendered arterial red. Real
aortic adventitia is pale tan-white — only the lumen is red. Changing it would break
the vascular colour coding your earlier versions were judged on. Set AORTA_REALISTIC
= True below if you want the anatomically correct version.

────────────────────────────────────────────────────────────────────────────────
CAVEAT FOR TRAINING (read before regenerating the dataset)
────────────────────────────────────────────────────────────────────────────────
The vascular network and micro-relief are functions of object-space position, which is
NOT recoverable from the (seg, depth, normals, segid) input stack in screen space. A
convolutional generator cannot invent view-consistent vessels from those inputs — it
will regress them to a blur and your L1/perceptual numbers may get WORSE even though
the GT is more realistic. Two ways out, if you go this route:
  (a) add a 3-channel object-space-position G-buffer to generate_training_dataset.py,
      which makes the detail learnable, or
  (b) keep --no-vessels for the dataset and use full v21 for thesis figures.
Decide this before committing 1228 subjects × 20 views of render time.

────────────────────────────────────────────────────────────────────────────────
WHAT v21 GOT WRONG  (all four verified against the render, not guessed)
────────────────────────────────────────────────────────────────────────────────
1. SUB-PIXEL OCTAVES -> MOIRE, not texture.
   Blender's noise `Detail = N` stacks N octaves at halving wavelength. v21 used
   Detail 5.0 on the 1.5 mm and 1.0 mm fields, so the finest octaves were 1.5/32 and
   1.0/32 mm — about 0.2 px at this camera. Sub-pixel noise aliases into the regular
   banding v21 produced. The base wavelengths were right; the octave count was not.
   Fix: _safe_detail() derives the octave budget from the actual mm/px of the shot, so
   the finest octave never lands below ~2 px. This is computed from the camera, not
   guessed, and adapts automatically to --size and FOV.

2. SUBSURFACE BLUR ERASED THE ALBEDO DETAIL.
   Liver ran sss = 0.92 at 3.0 mm scale. Random-walk transport is a ~3 mm blur kernel
   on base colour, and the vessel lines were ~1.6 mm wide — painted finer than the
   blur applied over them, so they vanished. Real liver penetration depth is ~1 mm,
   so 3.0 mm was too deep anyway. Fix: SSS scales roughly halved, vessel lines roughly
   doubled in width, and vessels now also drive the normal, which SSS does not blur.

3. THE SPECULAR WAS SPREAD UNTIL IT DIED.
   Coat-normal breakup at strength 0.30/0.35 scattered the lobe so widely that no
   single glint stayed bright enough to read, and OIDN then removed the remainder as
   noise. Fix: strength down to 0.12/0.10, wavelengths up to 6 mm/3 mm — fewer, larger,
   brighter glints. Broken specular needs to stay bright to read as wet.

4. FOUR BRIGHTENINGS STACKED.
   albedo x3, saturation 1.25, AgX Punchy, and random-walk multiple scattering (which
   brightens and saturates ABOVE the input albedo) all at once -> orange. Measured,
   v20's liver is RGB (196,116,104); real liver in surgical photography is
   (110-150, 55-75, 55-70). v20 was already ~1.5 stops hot and v21 added to it.
   Fix: default exposure -1.3 EV, saturation back to 1.0, AgX Medium Contrast, and
   albedos pulled down — SSS supplies the brightness that the albedo used to fake.

Run:
    blender --background --python scripts/render_pair_totalseg_v25_textured.py -- \\
        --subject s0050 --spp 384 --size 1024 --angles 1 --device GPU

Bisect a single change:
    ... -- --subject s0050 --angles 1 --no-vessels --no-micro
"""

import bpy
import sys
import os
import math
import hashlib
import argparse
import numpy as np
from pathlib import Path


# Image-plane sampling rate in mm/px, computed in main() from the actual camera and
# used by _safe_detail() to cap the octave budget. Global because every noise node in
# every material needs it and it is a single per-run constant.
MM_PER_PX = 0.0


# Set True for anatomically correct pale aortic adventitia (see docstring).
AORTA_REALISTIC = False


# ── Parse args ────────────────────────────────────────────────────────────────
def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject",  default="s0050")
    ap.add_argument("--dataset",  default="/home/vulovic/jasmina/dataset")
    ap.add_argument("--mesh_dir", default="data/meshes")
    ap.add_argument("--spp",    type=int, default=384)
    ap.add_argument("--size",   type=int, default=1024)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--tag",    default="v21", help="output filename tag")
    ap.add_argument("--gt_only", action="store_true",
                    help="skip the EEVEE simple/seg pass and render only the Cycles GT. "
                         "Required on headless servers with no /dev/dri access, where "
                         "EEVEE cannot create a GL context (libEGL EGL_BAD_MATCH). The "
                         "simple pass is unchanged from v20, so nothing is lost here.")

    # ── Photorealism layers — disable individually to bisect ──
    ap.add_argument("--no-sss-fix",  dest="sss_fix",  action="store_false",
                    help="keep v20's near-opaque Subsurface Weight ~0.05")
    ap.add_argument("--no-micro",    dest="micro",    action="store_false",
                    help="disable multi-scale micro-relief and coat specular breakup")
    ap.add_argument("--no-vessels",  dest="vessels",  action="store_false",
                    help="disable surface vasculature and lung anthracosis")
    ap.add_argument("--no-perfusion",dest="perfusion",action="store_false",
                    help="disable multi-octave colour/hue variation")
    ap.add_argument("--no-env",      dest="env",      action="store_false",
                    help="keep v20's pure-black world and black negative fill")
    ap.add_argument("--no-denoise",  dest="denoise",  action="store_false")
    ap.add_argument("--no-tone-fix", dest="tone_fix", action="store_false",
                    help="revert to AgX Medium Contrast at saturation 0.95")

    # ── Continuous knobs ──
    ap.add_argument("--detail",     type=float, default=1.0,
                    help="global multiplier on all micro-relief amplitudes")
    ap.add_argument("--vessel_gain",type=float, default=1.0,
                    help="global multiplier on vessel visibility")
    ap.add_argument("--saturation", type=float, default=1.05,
                    help="pre-compensation for AgX inset desaturation. v23 shipped 1.3, "
                         "which reads too vibrant against the reference renders — the "
                         "targets are cooler and more muted. 1.05 with the darker v24 "
                         "palette lands closer; raise toward 1.3 for a hotter look.")
    ap.add_argument("--exposure",   type=float, default=-1.00,
                    help="EV offset. v20/v21 rendered liver at RGB ~(196,116,104); "
                         "surgical photographs sit at (110-150, 55-75, 55-70), i.e. "
                         "about 1.3 stops darker. THE most impactful single knob here "
                         "— sweep it first: -0.8, -1.3, -1.8")
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--look",       default="AgX - Medium Contrast",
                    help="v21 used 'AgX - Punchy', which over-saturated the reds")
    ap.add_argument("--albedo",     type=float, default=1.00,
                    help="global gain on every tissue base colour. v21 raised albedos "
                         "~3x to fight AgX desaturation, then also enabled strong SSS "
                         "(which brightens ABOVE the input albedo) — double-counted.")
    ap.add_argument("--legacy_bump", action="store_true",
                    help="re-enable the 8-bit UV bump maps from v7-v22. They are the "
                         "source of the contour banding — see the docstring. Off by "
                         "default; the object-space procedural relief replaces them.")
    ap.add_argument("--walls", action="store_true",
                    help="add the dark negative-fill side planes. These were a v20 "
                         "device for a much closer camera; with v24's framing they sit "
                         "in shot and light up as a brown backdrop, and the environment "
                         "now supplies the wraparound they used to shape. Off by default.")
    ap.add_argument("--frame",      type=float, default=0.85,
                    help="fraction of the frame the organs should fill, 0-1. "
                         "0.85 leaves a small margin; 1.0 is edge-to-edge; 0.6 pulls "
                         "back. v23 and earlier framed off the CT field of view, so "
                         "zoom varied wildly between subjects with identical anatomy.")
    ap.add_argument("--smooth",     type=int,   default=20,
                    help="Laplacian smoothing iterations to kill the 1.5 mm "
                         "marching-cubes staircase. 0 disables (= v22 behaviour).")
    ap.add_argument("--smooth_factor", type=float, default=0.5,
                    help="lambda per iteration; Bade et al. use 0.5 for liver")
    ap.add_argument("--tex_dir",    default="data/renders/textures/tissue",
                    help="box-projected {tissue}_albedo/_normal/_rough maps")
    ap.add_argument("--tex_tint",   type=float, default=0.0,
                    help="0 = use the synthesised albedo as-is (it already carries the "
                         "organ palette); 1 = ignore it and use the table colour. "
                         "Values between blend the two.")
    ap.add_argument("--tex_mm",     type=float, default=80.0,
                    help="real-world tile size of those maps, in millimetres")
    ap.add_argument("--sss_scale",  type=float, default=0.90,
                    help="global gain on Subsurface Scale. v23's 0.55 suppressed the "
                         "translucency entirely — no light reached thin margins, so "
                         "organs read as opaque plastic. 0.90 restores the edge glow "
                         "that the reference renders show at the liver margin and "
                         "bowel wall. Too high and albedo detail blurs away again.")
    ap.add_argument("--fstop",      type=float, default=11.0, help="v20 used 6.3")
    ap.add_argument("--key_energy", type=float, default=90.0)
    ap.add_argument("--sss_method", default="RANDOM_WALK",
                    choices=["RANDOM_WALK", "RANDOM_WALK_SKIN", "BURLEY"],
                    help="RANDOM_WALK keeps the calibrated radii; SKIN derives them "
                         "from base colour (more saturated, less control)")

    # ── Dataset hygiene ──
    ap.add_argument("--training_safe", action="store_true",
                    help="disable screen-space post (chromatic aberration, fog glow). "
                         "These are positional effects a convolutional generator "
                         "cannot learn — leave them off for dataset generation.")
    return ap.parse_args(argv)


# ── Physical-scale helper — THE fix for the v16–v19 noise bug ────────────────

def _mm_scale(wavelength_mm):
    """Noise/Voronoi Scale that yields features of `wavelength_mm` millimetres.

    OBJECT SPACE IS MILLIMETRES, NOT METRES. This was wrong in v21/v22 and in every
    earlier version, and it is why no procedural detail has ever been visible.

    bpy.ops.wm.obj_import(global_scale=0.001) does NOT bake the scale into the vertex
    data — it sets the object's SCALE TRANSFORM to 0.001 and leaves local coordinates
    at their original millimetre values. Verified on s0050 liver:

        object.scale        = (0.001, 0.001, 0.001)
        local vertex coords = 155..340        <- millimetres
        world coords        = 0.155..0.340    <- metres

    ShaderNodeTexCoord's 'Object' output is LOCAL space, so it delivers millimetres.
    A texture Scale of s therefore gives features of 1/s MILLIMETRES:

        scale = 1 / wavelength_mm

    v21/v22 used 1000/wavelength_mm, i.e. every feature rendered at one millionth of
    the requested size — 28 mm lobulation came out at 0.028 mm, far below one pixel, so
    it averaged to flat grey. Rendering the raw Voronoi as emission shows uniform
    per-pixel noise instead of cells, which is what confirmed this.

    It also explains v16-v20: scales of 1.5-85 on millimetre coordinates are 0.67 mm
    down to 0.012 mm features. v19's conclusion that procedural noise "added variance,
    not signal" was literally correct — it was all sub-pixel.
    """
    return 1.0 / float(wavelength_mm)


def _safe_detail(wavelength_mm, mm_per_px, requested=4.0):
    """Octave budget that keeps the finest octave above the sampling limit.

    Blender's `Detail` adds octaves at halving wavelength, so the finest feature is
    wavelength / 2**Detail. Anything below ~2 px aliases into moire rather than
    rendering as texture — this is what wrecked v21. Solving for the octave count that
    lands the finest octave at 2 px:

        Detail_max = log2(wavelength_mm / (2 * mm_per_px))
    """
    if mm_per_px <= 0:
        mm_per_px = MM_PER_PX
    if mm_per_px <= 0:
        return requested
    budget = math.log2(max(wavelength_mm / (2.0 * mm_per_px), 1.0))
    return float(max(0.0, min(requested, budget)))


def _phase(name, octave=0):
    """Deterministic per-tissue seed, so adjacent organs don't share a noise phase.

    All OBJs live in the same absolute CT coordinate frame, so without this the
    procedural fields would run continuously across organ boundaries.
    """
    h = hashlib.md5(f"{name}:{octave}".encode()).hexdigest()
    return (int(h[:8], 16) % 100000) / 1000.0


# ── Tissue definitions ────────────────────────────────────────────────────────
#
# base            scattering albedo. Raised vs v20, which was tuned to survive AgX
#                 Base desaturation and reads washed-out once that is corrected.
# sss / sss_mm    Subsurface Weight, and Subsurface Scale in MILLIMETRES.
# sss_rgb         per-channel radius multiplier, renormalised so max == 1.0, so
#                 sss_mm is the true red-channel transport depth.
# sss_aniso       Henyey-Greenstein g. Soft tissue ≈ 0.9; Blender clamps near 0.9.
# micro           amplitude multiplier for the surface micro-relief chain.
# vessel          surface vasculature visibility, 0 = none.
# vessel_mm       spacing of the primary vascular arcade in mm.
# perfusion       amplitude of the blood-content colour/hue field.

_DEFAULTS = dict(
    rough=0.40, ior=1.40,
    sss=0.0, sss_mm=0.0, sss_rgb=(1.0, 1.0, 1.0), sss_aniso=0.80,
    coat=0.0, coat_rough=0.06, coat_tint=(1.00, 0.99, 0.95),
    sheen=0.0, sheen_rough=0.35,
    bump_type="none", bump_scale=0.0,
    micro=1.0, macro_mm=25.0, meso_mm=6.0, fine_mm=1.6,
    vessel=0.0, vessel_mm=14.0, vessel_col=(0.10, 0.020, 0.045),
    perfusion=0.17, hue_shift=0.010,
)


def T(name, hex_, base, **kw):
    d = dict(_DEFAULTS)
    d.update(name=name, hex=hex_, base=list(base))
    d.update(kw)
    return d


TISSUES = [
    # ── Skeletal muscle ──────────────────────────────────────────────────────
    # Dark red-brown with perimysial fibre striation. Fibrous bump is directional
    # in reality; the UV texture approximates it.
    T("autochthon_left",  "#4A3E3D", (0.22, 0.065, 0.055), rough=0.52,
      sss=0.55, sss_mm=1.6, sss_rgb=(1.00, 0.70, 0.50), sss_aniso=0.85,
      coat=0.10, coat_rough=0.260, sheen=0.06,
      bump_type="fibrous", bump_scale=0.15,
      micro=1.15, meso_mm=4.0, fine_mm=1.2,
      vessel=0.14, vessel_mm=24.0, perfusion=0.24),
    T("autochthon_right", "#4A3E3D", (0.22, 0.065, 0.055), rough=0.52,
      sss=0.55, sss_mm=1.6, sss_rgb=(1.00, 0.70, 0.50), sss_aniso=0.85,
      coat=0.10, coat_rough=0.260, sheen=0.06,
      bump_type="fibrous", bump_scale=0.15,
      micro=1.15, meso_mm=4.0, fine_mm=1.2,
      vessel=0.14, vessel_mm=24.0, perfusion=0.24),

    # ── Lung ─────────────────────────────────────────────────────────────────
    # Air-filled, so short transport depth and high albedo. The vessel channel here
    # carries ANTHRACOTIC MOTTLING — the black carbon deposits along interlobular
    # septa present in every adult lung. Highly distinctive and a strong realism cue;
    # v20's uniform pink-grey is what a neonatal lung looks like.
    *[T(n, "#9C8585", (0.44, 0.315, 0.305), rough=0.44,
        sss=0.70, sss_mm=1.1, sss_rgb=(1.00, 0.80, 0.68), sss_aniso=0.70,
        coat=0.20, coat_rough=0.150, sheen=0.05,
        bump_type="smooth", bump_scale=0.10,
        micro=0.85, macro_mm=30.0, meso_mm=7.0, fine_mm=2.0,
        vessel=0.10, vessel_mm=26.0, vessel_col=(0.055, 0.050, 0.050),
        perfusion=0.19, hue_shift=0.006)
      for n in ("lung_lower_lobe_left", "lung_lower_lobe_right",
                "lung_upper_lobe_left", "lung_upper_lobe_right")],

    # ── Vertebrae ────────────────────────────────────────────────────────────
    # v20 gave bone zero SSS and it renders as blown-out chalk. Cortical bone is
    # translucent to ~1-2 mm and fresh bone is off-white with a yellow cast, never
    # neutral white. Low SSS + warmer albedo removes the plaster-cast look.
    *[T(n, "#C5BEB2", (0.50, 0.455, 0.385), rough=0.62, ior=1.55,
        sss=0.22, sss_mm=0.9, sss_rgb=(1.00, 0.90, 0.76), sss_aniso=0.55,
        coat=0.02, coat_rough=0.480, sheen=0.03,
        micro=0.70, macro_mm=18.0, meso_mm=5.0, fine_mm=1.4,
        vessel=0.0, perfusion=0.12, hue_shift=0.004)
      for n in ("vertebrae_T12", "vertebrae_L1", "vertebrae_L2",
                "vertebrae_L3", "vertebrae_L4", "vertebrae_L5")],

    # ── Heart ────────────────────────────────────────────────────────────────
    # Epicardial fat streaks along the coronary grooves and a wet pericardial sheen.
    T("heart", "#8A2A2A", (0.34, 0.085, 0.070), rough=0.34,
      sss=0.88, sss_mm=2.2, sss_rgb=(1.00, 0.74, 0.53), sss_aniso=0.85,
      coat=0.55, coat_rough=0.032, sheen=0.05,
      bump_type="lobular", bump_scale=0.35,
      micro=1.10, macro_mm=22.0, meso_mm=5.5, fine_mm=1.5,
      vessel=0.45, vessel_mm=28.0, vessel_col=(0.12, 0.030, 0.055),
      perfusion=0.27, hue_shift=0.014),

    # ── Esophagus ────────────────────────────────────────────────────────────
    T("esophagus", "#9E6464", (0.36, 0.185, 0.165), rough=0.40,
      sss=0.60, sss_mm=1.4, sss_rgb=(1.00, 0.80, 0.67), sss_aniso=0.80,
      coat=0.44, coat_rough=0.060, sheen=0.05,
      bump_type="vessel", bump_scale=0.20,
      micro=1.0, meso_mm=4.5, fine_mm=1.3,
      vessel=0.10, vessel_mm=20.0, perfusion=0.19),

    # ── Liver ────────────────────────────────────────────────────────────────
    # Bashkatov 2011: μ_s' = 17.5/12.8/9.2 cm⁻¹ at 632/532/457 nm; with μ_a ≈ 2.5 cm⁻¹
    # the effective penetration is ~1 mm, so scale 3.0 mm with the radii below puts
    # red at 3 mm and blue at 1.6 mm. Glisson's capsule gives the wettest, sharpest
    # specular of any abdominal organ, hence the high coat weight and low coat roughness.
    T("liver", "#5C2018", (0.32, 0.085, 0.065), rough=0.26,
      sss=0.92, sss_mm=3.0, sss_rgb=(1.00, 0.74, 0.53), sss_aniso=0.85,
      coat=0.62, coat_rough=0.022, sheen=0.04,
      bump_type="lobular", bump_scale=0.40,
      micro=1.25, macro_mm=28.0, meso_mm=6.0, fine_mm=1.5,
      vessel=0.40, vessel_mm=34.0, vessel_col=(0.085, 0.018, 0.040),
      perfusion=0.34, hue_shift=0.016),

    # ── Stomach ──────────────────────────────────────────────────────────────
    T("stomach", "#9E916B", (0.44, 0.335, 0.235), rough=0.34,
      sss=0.62, sss_mm=1.8, sss_rgb=(1.00, 0.82, 0.70), sss_aniso=0.80,
      coat=0.60, coat_rough=0.030, sheen=0.06,
      bump_type="wrinkled", bump_scale=0.45,
      micro=1.20, macro_mm=20.0, meso_mm=4.5, fine_mm=1.3,
      vessel=0.10, vessel_mm=22.0, vessel_col=(0.14, 0.045, 0.055),
      perfusion=0.22, hue_shift=0.012),

    # ── Gallbladder ──────────────────────────────────────────────────────────
    # Thin translucent wall over green bile: deepest SSS relative to size, and the
    # green-dominant radius is genuine (bilirubin/biliverdin absorption), not stylistic.
    T("gallbladder", "#3A5E35", (0.105, 0.165, 0.075), rough=0.20,
      sss=0.95, sss_mm=3.4, sss_rgb=(0.42, 1.00, 0.58), sss_aniso=0.75,
      coat=0.66, coat_rough=0.020, sheen=0.03,
      bump_type="lobular", bump_scale=0.30,
      micro=0.85, macro_mm=16.0, meso_mm=4.0, fine_mm=1.4,
      vessel=0.12, vessel_mm=18.0, vessel_col=(0.05, 0.075, 0.030),
      perfusion=0.20, hue_shift=0.010),

    # ── Spleen ───────────────────────────────────────────────────────────────
    # Red pulp: the most purple of the parenchymal organs, and the most friable-
    # looking surface. Highest perfusion variance.
    T("spleen", "#523050", (0.245, 0.055, 0.080), rough=0.27,
      sss=0.90, sss_mm=2.6, sss_rgb=(1.00, 0.72, 0.56), sss_aniso=0.85,
      coat=0.58, coat_rough=0.026, sheen=0.04,
      bump_type="lobular", bump_scale=0.40,
      micro=1.20, macro_mm=24.0, meso_mm=5.5, fine_mm=1.5,
      vessel=0.30, vessel_mm=30.0, vessel_col=(0.075, 0.016, 0.050),
      perfusion=0.34, hue_shift=0.018),

    # ── Kidneys ──────────────────────────────────────────────────────────────
    *[T(n, "#4A1E28", (0.30, 0.100, 0.085), rough=0.26, ior=1.42,
        sss=0.88, sss_mm=2.4, sss_rgb=(1.00, 0.76, 0.59), sss_aniso=0.85,
        coat=0.58, coat_rough=0.028, sheen=0.04,
        bump_type="lobular", bump_scale=0.45,
        micro=1.15, macro_mm=20.0, meso_mm=5.0, fine_mm=1.4,
        vessel=0.35, vessel_mm=26.0, vessel_col=(0.10, 0.025, 0.045),
        perfusion=0.29, hue_shift=0.014)
      for n in ("kidney_right", "kidney_left")],

    # ── Pancreas ─────────────────────────────────────────────────────────────
    # The most obviously lobulated organ in the abdomen — coarse fat-separated
    # lobules, not a smooth surface. Short macro wavelength drives that.
    T("pancreas", "#B09170", (0.46, 0.345, 0.235), rough=0.42,
      sss=0.72, sss_mm=1.7, sss_rgb=(1.00, 0.84, 0.72), sss_aniso=0.78,
      coat=0.16, coat_rough=0.230, sheen=0.07,
      bump_type="lobular", bump_scale=0.40,
      micro=1.35, macro_mm=12.0, meso_mm=3.5, fine_mm=1.2,
      vessel=0.12, vessel_mm=18.0, vessel_col=(0.16, 0.070, 0.045),
      perfusion=0.26, hue_shift=0.010),

    # ── Bowel ────────────────────────────────────────────────────────────────
    # v20 excluded hollow organs from SSS entirely. A bowel wall is 2-4 mm of
    # translucent tissue over a lumen — it is one of the MOST translucent structures
    # in the field, especially at grazing angles. The exclusion is what makes these
    # read as tan plastic tubing. Serosal vessel arcades are their signature feature.
    *[T(n, "#A38470", (0.42, 0.305, 0.245), rough=0.36,
        sss=0.68, sss_mm=1.9, sss_rgb=(1.00, 0.82, 0.70), sss_aniso=0.82,
        coat=0.60, coat_rough=0.030, sheen=0.07,
        bump_type="wrinkled", bump_scale=0.40,
        micro=1.25, macro_mm=14.0, meso_mm=4.0, fine_mm=1.2,
        vessel=0.0,  vessel_mm=8.0, vessel_col=(0.15, 0.045, 0.050),
        perfusion=0.24, hue_shift=0.012)
      for n in ("duodenum", "small_bowel")],
    T("colon", "#8F6E5C", (0.38, 0.270, 0.205), rough=0.36,
      sss=0.66, sss_mm=1.9, sss_rgb=(1.00, 0.82, 0.70), sss_aniso=0.82,
      coat=0.60, coat_rough=0.030, sheen=0.07,
      bump_type="wrinkled", bump_scale=0.40,
      micro=1.25, macro_mm=16.0, meso_mm=4.5, fine_mm=1.3,
      vessel=0.0,  vessel_mm=9.0, vessel_col=(0.15, 0.045, 0.050),
      perfusion=0.24, hue_shift=0.012),

    # ── Urinary bladder ──────────────────────────────────────────────────────
    T("urinary_bladder", "#6E758A", (0.22, 0.235, 0.290), rough=0.32,
      sss=0.64, sss_mm=1.6, sss_rgb=(1.00, 0.88, 0.80), sss_aniso=0.78,
      coat=0.38, coat_rough=0.070, sheen=0.05,
      bump_type="smooth", bump_scale=0.20,
      micro=0.95, macro_mm=18.0, meso_mm=5.0, fine_mm=1.5,
      vessel=0.12, vessel_mm=20.0, vessel_col=(0.13, 0.055, 0.075),
      perfusion=0.17, hue_shift=0.008),

    # ── Aorta ────────────────────────────────────────────────────────────────
    # Oxy-Hb scatters deepest in red. See AORTA_REALISTIC in the docstring: the real
    # adventitial surface is pale tan-white, not arterial red.
    T("aorta", "#A31414",
      (0.42, 0.305, 0.265) if AORTA_REALISTIC else (0.40, 0.055, 0.038),
      rough=0.17,
      sss=0.80, sss_mm=1.3, sss_rgb=(1.00, 0.64, 0.40), sss_aniso=0.85,
      coat=0.58, coat_rough=0.022, sheen=0.04,
      bump_type="vessel", bump_scale=0.20,
      micro=0.90, macro_mm=14.0, meso_mm=3.5, fine_mm=1.1,
      vessel=0.08, vessel_mm=16.0, vessel_col=(0.14, 0.040, 0.040),
      perfusion=0.15, hue_shift=0.008),

    # ── Veins ────────────────────────────────────────────────────────────────
    # Deoxy-Hb absorbs more at 650 nm, so red penetrates less than in arteries —
    # the R:G:B radius ratio is genuinely flatter here, not a stylistic choice.
    *[T(n, "#3D2050", (0.145, 0.075, 0.170), rough=0.18,
        sss=0.78, sss_mm=1.2, sss_rgb=(1.00, 0.86, 0.66), sss_aniso=0.82,
        coat=0.34, coat_rough=0.028, sheen=0.04,
        bump_type="vessel", bump_scale=0.15,
        micro=0.90, macro_mm=14.0, meso_mm=3.5, fine_mm=1.1,
        vessel=0.08, vessel_mm=16.0, vessel_col=(0.06, 0.030, 0.085),
        perfusion=0.15, hue_shift=0.008)
      for n in ("inferior_vena_cava", "portal_vein_and_splenic_vein",
                "superior_vena_cava")],
]

# ── Reference-matched appearance ─────────────────────────────────────────────
#
# The table above holds physically-derived priors (tissue optics, transport depths).
# This layer holds APPEARANCE, matched by eye against reference renders of anatomical
# models, and is deliberately separate so the two never get confused: change this to
# chase a look, change the table above to change the physics.
#
# base        scattering albedo, linear. These are absolute — --albedo defaults to 1.0
#             now, so what is written here is what renders.
# perfusion   amplitude of the blood-content field. Values below ~0.4 are invisible
#             once subsurface scattering compresses them; 0.5-0.7 reads as tissue.
# vessel      surface vasculature coverage. Needs vessel_col to CONTRAST with base —
#             on a dark organ a dark vessel disappears no matter how high this goes.
# hue_shift   keep <= 0.02. At 0.04 the hue rotation pushes tissue into yellow-green.
#
# Verified per organ by rendering in isolation and comparing to reference; see the
# per-organ notes for what each was matched against.
REFERENCE_LOOK = {
    # Salmon-pink, strong irregular mottling, fine dark vessel tracery.
    # Was pure white plastic — the single worst offender in the v23 scene render.
    # Lung tracery must stay near-subliminal. At vessel 0.70 the single noise-contour
    # field reads as drawn-on squiggles rather than vasculature — there is only one
    # structural model here, so pushing its amplitude makes it look MORE procedural,
    # not more detailed. Low coverage, low contrast, broad territories.
    "lung": dict(sss=0.30, sheen=0.22, rough=0.62, base=(0.460, 0.240, 0.232), perfusion=0.60, vessel=0.22,
                 vessel_mm=30.0, vessel_col=(0.30, 0.135, 0.135), hue_shift=0.015),
    # Deep burgundy-brown, glossy capsule, large tonal zones, vessels subtle —
    # in the references the liver reads smooth and dark, not heavily veined.
    # Tuned by sweep: 0.26 red read as orange, and hue_shift 0.018 pushed patches into
    # visible yellow-green. Deeper and browner, with the hue drift almost off.
    "liver": dict(sss=0.95, sheen=0.02, rough=0.22, base=(0.108, 0.036, 0.028), perfusion=0.60, vessel=0.30,
                  vessel_mm=34.0, vessel_col=(0.070, 0.014, 0.026), hue_shift=0.005),
    "spleen": dict(base=(0.135, 0.030, 0.050), perfusion=0.55, vessel=0.26,
                   vessel_mm=30.0, vessel_col=(0.065, 0.013, 0.042), hue_shift=0.007),
    "kidney": dict(base=(0.185, 0.068, 0.046), perfusion=0.50, vessel=0.30,
                   vessel_mm=26.0, vessel_col=(0.090, 0.022, 0.038), hue_shift=0.008),
    # Myocardium red-pink; coronary vessels are the prominent feature.
    "heart": dict(base=(0.300, 0.085, 0.072), perfusion=0.50, vessel=0.65,
                  vessel_mm=26.0, vessel_col=(0.110, 0.025, 0.045), hue_shift=0.015),
    # Pale tan-pink serosa, smooth and glossy, sparse vessels.
    "stomach": dict(base=(0.500, 0.365, 0.305), perfusion=0.45, vessel=0.22,
                    vessel_mm=22.0, vessel_col=(0.22, 0.075, 0.070), hue_shift=0.012),
    # Pink with DENSE fine vessels — the defining feature of small-bowel serosa.
    # Tuned by sweep: at vessel_mm 11 with a light vessel_col the tracery washed out
    # entirely. 7 mm spacing and a much darker vessel colour is what makes it read.
    "bowel": dict(base=(0.520, 0.315, 0.288), perfusion=0.45, vessel=0.42,
                  vessel_mm=9.0, vessel_col=(0.16, 0.020, 0.022), hue_shift=0.012),
    # "bowel" does not substring-match "duodenum", so it needs its own entry —
    # without it the duodenum silently kept the un-tuned table values and rendered
    # cream while the rest of the small bowel rendered pink.
    "duodenum": dict(base=(0.520, 0.315, 0.288), perfusion=0.45, vessel=0.42,
                  vessel_mm=9.0, vessel_col=(0.16, 0.020, 0.022), hue_shift=0.012),
    "colon": dict(base=(0.440, 0.258, 0.248), perfusion=0.48, vessel=0.34,
                  vessel_mm=13.0, vessel_col=(0.17, 0.030, 0.040), hue_shift=0.014),
    # Tan-yellow and coarsely lobulated.
    "pancreas": dict(sss=0.40, sheen=0.14, rough=0.58, base=(0.480, 0.355, 0.238), perfusion=0.50, vessel=0.18,
                     vessel_mm=18.0, vessel_col=(0.24, 0.130, 0.060), hue_shift=0.012),
    # Bile green, but muted. The green-dominant SSS radius (bilirubin/biliverdin) at
    # sss 0.95 scatters green straight back out and reads as lime, so the weight and
    # the radius asymmetry both have to come down, not just the base colour.
    "gallbladder": dict(base=(0.065, 0.090, 0.058), perfusion=0.40, vessel=0.16,
                        vessel_mm=16.0, vessel_col=(0.040, 0.060, 0.030),
                        hue_shift=0.010, sss=0.55, sss_rgb=(0.70, 1.00, 0.78)),
    "esophagus": dict(base=(0.400, 0.262, 0.238), perfusion=0.40, vessel=0.20,
                      vessel_mm=20.0, vessel_col=(0.19, 0.070, 0.065), hue_shift=0.012),
    "urinary_bladder": dict(base=(0.34, 0.25, 0.24), perfusion=0.40, vessel=0.20,
                            vessel_mm=20.0, vessel_col=(0.17, 0.090, 0.100), hue_shift=0.010),
    # Muted brick rather than fire-engine red. See AORTA_REALISTIC — the true
    # adventitial surface is pale tan; this is a compromise with the colour coding.
    "aorta": dict(base=(0.340, 0.125, 0.105), perfusion=0.30, vessel=0.14,
                  vessel_mm=16.0, vessel_col=(0.20, 0.060, 0.055), hue_shift=0.008),
    "vena_cava": dict(base=(0.20, 0.11, 0.22), perfusion=0.30, vessel=0.12,
                      vessel_mm=16.0, vessel_col=(0.090, 0.045, 0.110), hue_shift=0.008),
    "portal": dict(base=(0.20, 0.11, 0.22), perfusion=0.30, vessel=0.12,
                   vessel_mm=16.0, vessel_col=(0.090, 0.045, 0.110), hue_shift=0.008),
    # Off-white with a yellow cast, never neutral chalk.
    "vertebrae": dict(sss=0.10, sheen=0.05, rough=0.74, base=(0.62, 0.56, 0.46), perfusion=0.25, vessel=0.0,
                      hue_shift=0.006),
    # Dark red striated skeletal muscle.
    "autochthon": dict(sss=0.35, sheen=0.26, rough=0.66, base=(0.255, 0.082, 0.098), perfusion=0.50, vessel=0.22,
                       vessel_mm=24.0, vessel_col=(0.12, 0.030, 0.030), hue_shift=0.014),
}


def apply_reference_look():
    """Overlay REFERENCE_LOOK onto TISSUES by longest-matching key."""
    for t in TISSUES:
        key = None
        for k in REFERENCE_LOOK:
            if k in t["name"] and (key is None or len(k) > len(key)):
                key = k
        if key is None:
            print(f"  [warn] no REFERENCE_LOOK entry matches '{t['name']}' — "
                  f"it will render with un-tuned table values")
            continue
        for field, val in REFERENCE_LOOK[key].items():
            t[field] = list(val) if isinstance(val, tuple) and field == "base" else val


apply_reference_look()


TEX_DIR = Path("data/renders/textures")


# ── Version-safe node plumbing ────────────────────────────────────────────────
# Targets Blender 5.x (the server build — see the OPEN_EXR_MULTILAYER note in
# generate_training_dataset.py) but degrades quietly on 4.x, where a few Principled
# v2 sockets are named differently or absent.

def _set(node, socket, value):
    """Set an input socket if it exists. Returns True if it took."""
    try:
        node.inputs[socket].default_value = value
        return True
    except Exception:
        return False


def _link_to(links, out_socket, node, socket):
    try:
        links.new(out_socket, node.inputs[socket])
        return True
    except Exception:
        return False


def _noise(nodes, links, vec_out, wavelength_mm, seed,
           detail=4.0, roughness=0.5, mm_per_px=0.0):
    """Fractal noise with features of `wavelength_mm` millimetres.

    Uses 4D noise so `seed` decorrelates octaves and tissues without needing a
    Mapping node per octave. See _mm_scale() for why the scale is 1000/mm.
    """
    n = nodes.new('ShaderNodeTexNoise')
    try:
        n.noise_dimensions = '4D'
    except Exception:
        pass
    _set(n, 'Scale',     _mm_scale(wavelength_mm))
    _set(n, 'Detail',    _safe_detail(wavelength_mm, mm_per_px, detail))
    _set(n, 'Roughness', roughness)
    _set(n, 'W',         seed)
    if vec_out is not None:
        _link_to(links, vec_out, n, 'Vector')
    return n


def _voronoi_edges(nodes, links, vec_out, wavelength_mm, seed):
    """Voronoi distance-to-edge — a reticular crack field.

    Distance → 0 on cell boundaries, so thresholding it low yields a branching
    network that reads convincingly as surface vasculature (liver, bowel serosa)
    or as interlobular septa (lung anthracosis).
    """
    v = nodes.new('ShaderNodeTexVoronoi')
    try:
        v.voronoi_dimensions = '4D'
    except Exception:
        pass
    try:
        v.feature = 'DISTANCE_TO_EDGE'
    except Exception:
        pass
    _set(v, 'Scale',      _mm_scale(wavelength_mm))
    _set(v, 'Randomness', 1.0)
    _set(v, 'W',          seed)
    if vec_out is not None:
        _link_to(links, vec_out, v, 'Vector')
    return v


def _map_range(nodes, links, value_out, to_min, to_max,
               from_min=0.0, from_max=1.0):
    m = nodes.new('ShaderNodeMapRange')
    _set(m, 'From Min', from_min)
    _set(m, 'From Max', from_max)
    _set(m, 'To Min',   to_min)
    _set(m, 'To Max',   to_max)
    if value_out is not None:
        _link_to(links, value_out, m, 'Value')
    return m


def _gray(nodes, links, value_out):
    """Scalar → grey RGB, for MULTIPLY-blending a scalar field into a colour."""
    c = nodes.new('ShaderNodeCombineColor')
    for ch in ('Red', 'Green', 'Blue'):
        _link_to(links, value_out, c, ch)
    return c


def _mix_rgb(nodes, links, blend, factor, a_out=None, b_out=None,
             a_val=None, b_val=None):
    """ShaderNodeMix in RGBA mode. Sockets: 6 = A, 7 = B, output 2 = Result."""
    m = nodes.new('ShaderNodeMix')
    m.data_type  = 'RGBA'
    m.blend_type = blend
    if isinstance(factor, float):
        m.inputs['Factor'].default_value = factor
    else:
        links.new(factor, m.inputs['Factor'])
    if a_val is not None:
        m.inputs[6].default_value = (*a_val, 1.0)
    if b_val is not None:
        m.inputs[7].default_value = (*b_val, 1.0)
    if a_out is not None:
        links.new(a_out, m.inputs[6])
    if b_out is not None:
        links.new(b_out, m.inputs[7])
    return m


def _bump(nodes, links, height_out, normal_out, strength, distance_m):
    """Chainable bump. `distance_m` is the physical relief height in metres."""
    b = nodes.new('ShaderNodeBump')
    _set(b, 'Strength', strength)
    _set(b, 'Distance', distance_m)
    if height_out is not None:
        _link_to(links, height_out, b, 'Height')
    if normal_out is not None:
        _link_to(links, normal_out, b, 'Normal')
    return b


# ── Procedural field builders ─────────────────────────────────────────────────

def build_perfusion(nodes, links, vec_out, t, color_out):
    """Multi-octave blood-content field: value AND hue.

    v20 varied value only, one octave, ±12 % at a 250 mm wavelength — i.e. a single
    gradient across the whole organ, invisible. Real parenchyma varies over 10-40 mm
    (congested regions read purple, well-perfused regions red), so this drives hue
    as well, which is what actually reads as "living tissue" rather than "tinted wax".
    """
    name = t["name"]
    amp  = t["perfusion"]

    n_lo = _noise(nodes, links, vec_out, 42.0, _phase(name, 1), detail=2.0)
    n_hi = _noise(nodes, links, vec_out, 13.0, _phase(name, 2), detail=4.0)

    # Weighted octave sum: 0.70 low + 0.30 high, biased back to unity mean.
    mix_oct = _mix_rgb(nodes, links, 'MIX', 0.30,
                       a_out=n_lo.outputs['Fac'], b_out=n_hi.outputs['Fac'])

    sep = nodes.new('ShaderNodeSeparateColor')
    links.new(mix_oct.outputs[2], sep.inputs['Color'])

    # Map from the noise's ACTUAL spread, not from 0..1. Blender's Noise `Fac` is
    # roughly gaussian about 0.5 and almost never reaches the extremes, so remapping
    # 0..1 -> the output range leaves nearly every pixel bunched in the middle and the
    # amplitude parameter does almost nothing. Measured on liver: raising `perfusion`
    # from 0.34 to 0.80 moved the output std only 0.044 -> 0.050.
    # Remapping from 0.35..0.65 lets the distribution fill the range.
    val = _map_range(nodes, links, sep.outputs['Red'], 1.0 - amp, 1.0 + amp,
                     from_min=0.35, from_max=0.65)
    out = _mix_rgb(nodes, links, 'MULTIPLY', 1.0,
                   a_out=color_out,
                   b_out=_gray(nodes, links, val.outputs['Result']).outputs['Color'])

    # Hue drift. 0.5 is neutral on the Hue/Saturation node.
    hs = t["hue_shift"]
    if hs > 0:
        hue = _map_range(nodes, links, n_lo.outputs['Fac'], 0.5 - hs, 0.5 + hs,
                         from_min=0.35, from_max=0.65)
        hsv = nodes.new('ShaderNodeHueSaturation')
        links.new(hue.outputs['Result'], hsv.inputs['Hue'])
        links.new(out.outputs[2], hsv.inputs['Color'])
        return hsv.outputs['Color']

    return out.outputs[2]


def build_vessels(nodes, links, vec_out, t, color_out, gain):
    """Sparse branching vasculature from noise iso-contours.

    v23's first attempt used Voronoi DISTANCE_TO_EDGE. That is a crackle/mud-flat
    pattern: a space-filling tessellation of uniform-width lines with uniform spacing.
    Vasculature is none of those things — it branches, its calibre varies continuously,
    and it covers a surface unevenly with large bare areas between arcades.

    This builds vessels as the |noise - 0.5| iso-contour instead:

      · the contour of a smooth field is sinuous and branches naturally
      · line width is inversely proportional to the local gradient, so calibre varies
        on its own — no extra machinery
      · a second, much lower-frequency mask gates where vessels appear at all, so the
        surface has bare regions rather than uniform coverage

    `vessel` in the tissue table is now coverage, not line darkness, and several organs
    are set to 0: small bowel serosa carries straight vasa recta perpendicular to the
    mesenteric border, which is a different pattern entirely and is not attempted here.

    Returns (colour_out, height_out); height drives a shallow relief bump.
    """
    name = t["name"]
    vis  = min(1.0, t["vessel"] * gain)
    if vis <= 0.0:
        return color_out, None
    vcol = t["vessel_col"]

    def _contour(wl_mm, width, seed):
        """Thin band along the 0.5 iso-contour of a noise field."""
        n = _noise(nodes, links, vec_out, wl_mm, seed, detail=2.5)
        off = nodes.new('ShaderNodeMath'); off.operation = 'SUBTRACT'
        links.new(n.outputs['Fac'], off.inputs[0]); _set(off, 1, 0.5)
        ab = nodes.new('ShaderNodeMath'); ab.operation = 'ABSOLUTE'
        links.new(off.outputs['Value'], ab.inputs[0])
        # near the contour -> 1, away -> 0
        return _map_range(nodes, links, ab.outputs['Value'], 1.0, 0.0,
                          from_min=0.0, from_max=width).outputs['Result']

    big   = _contour(t["vessel_mm"] * 2.2,  0.055, _phase(name, 3))
    small = _contour(t["vessel_mm"] * 0.85, 0.030, _phase(name, 4))

    # Union of the two calibres, small ones fainter.
    sm = nodes.new('ShaderNodeMath'); sm.operation = 'MULTIPLY'
    links.new(small, sm.inputs[0]); _set(sm, 1, 0.55)
    un = nodes.new('ShaderNodeMath'); un.operation = 'MAXIMUM'
    links.new(big, un.inputs[0]); links.new(sm.outputs['Value'], un.inputs[1])

    # Coverage gate: large bare regions between vascular territories.
    gate_n = _noise(nodes, links, vec_out, t["vessel_mm"] * 5.0, _phase(name, 12),
                    detail=1.5)
    gate = _map_range(nodes, links, gate_n.outputs['Fac'], 0.15, 1.0,
                      from_min=0.34, from_max=0.60)
    gated = nodes.new('ShaderNodeMath'); gated.operation = 'MULTIPLY'
    links.new(un.outputs['Value'], gated.inputs[0])
    links.new(gate.outputs['Result'], gated.inputs[1])

    amt = nodes.new('ShaderNodeMath'); amt.operation = 'MULTIPLY'
    links.new(gated.outputs['Value'], amt.inputs[0]); _set(amt, 1, vis)

    # Factor 1 on a vessel -> vessel colour. (Mix returns A at 0, B at 1.)
    tinted = _mix_rgb(nodes, links, 'MIX', amt.outputs['Value'],
                      a_out=color_out, b_val=tuple(vcol))
    return tinted.outputs[2], amt.outputs['Value']


def build_micro_normals(nodes, links, vec_out, t, base_normal_out, detail_gain):
    """Three-octave surface relief, plus a separate high-frequency coat normal.

    Amplitudes are physical heights in metres:
      macro ~0.6 mm  lobulation and capsular tension
      meso  ~0.18 mm capsule wrinkle
      fine  ~0.05 mm serosal grain

    The coat chain is deliberately NOT shared with the base normal. Perturbing the
    base normal at these frequencies is what made v17 look noisy — it corrupts the
    diffuse and SSS terms, which should stay smooth. Perturbing only the coat normal
    breaks the specular lobe into glints while leaving the shading coherent, which is
    exactly what a wet serosal film does in a photograph.

    Returns (base_normal_out, coat_normal_out).
    """
    name = t["name"]
    amp  = t["micro"] * detail_gain

    n_macro = _noise(nodes, links, vec_out, t["macro_mm"], _phase(name, 5),
                     detail=3.0, roughness=0.55)
    n_meso  = _noise(nodes, links, vec_out, t["meso_mm"],  _phase(name, 6),
                     detail=4.0, roughness=0.50)
    n_fine  = _noise(nodes, links, vec_out, t["fine_mm"],  _phase(name, 7),
                     detail=3.0, roughness=0.55)

    b = _bump(nodes, links, n_macro.outputs['Fac'], base_normal_out,
              0.12 * amp, 0.00060)
    b = _bump(nodes, links, n_meso.outputs['Fac'],  b.outputs['Normal'],
              0.18 * amp, 0.00018)
    b = _bump(nodes, links, n_fine.outputs['Fac'],  b.outputs['Normal'],
              0.22 * amp, 0.00005)
    base_out = b.outputs['Normal']

    # Coat film relief. Heights here are microns — enough to steer a mirror-sharp
    # lobe, far too small to affect the diffuse shading.
    # v21 used 3.0/1.0 mm at strength 0.30/0.35 and the highlight disappeared: the lobe
    # was spread so wide that no glint stayed bright enough to read, and OIDN then
    # cleaned up the remainder as noise. Wet tissue needs FEWER and BRIGHTER glints.
    c_mid  = _noise(nodes, links, vec_out, 6.0, _phase(name, 8), detail=3.0)
    c_fine = _noise(nodes, links, vec_out, 3.0, _phase(name, 9), detail=3.0)

    c = _bump(nodes, links, c_mid.outputs['Fac'],  base_out,
              0.12 * amp, 0.000050)
    c = _bump(nodes, links, c_fine.outputs['Fac'], c.outputs['Normal'],
              0.10 * amp, 0.000015)

    return base_out, c.outputs['Normal']


# ── Box-projected texture maps ────────────────────────────────────────────────

def _box_tex(nodes, links, obj_vec, path, tex_mm, non_color):
    """Image texture in BOX (triplanar) projection.

    Marching-cubes meshes have no meaningful UV layout, so flat projection is useless
    and the *_uv.obj unwraps are arbitrary. Box projection needs no UVs at all: the
    image is projected along all three object axes and cross-faded at the seams, which
    is the standard solution for scanned and iso-surface geometry.

    Object coordinates are metres, so a Mapping scale of 1000/tex_mm makes the texture
    repeat every tex_mm millimetres of real tissue.
    """
    m = nodes.new('ShaderNodeMapping')
    sc = _mm_scale(tex_mm)
    _set(m, 'Scale', (sc, sc, sc))
    _link_to(links, obj_vec, m, 'Vector')

    tex = nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(path))
    if non_color:
        tex.image.colorspace_settings.name = 'Non-Color'
    try:
        tex.projection       = 'BOX'
        tex.projection_blend = 0.35      # cross-fade width at the axis seams
    except Exception:
        pass
    tex.extension = 'REPEAT'
    _link_to(links, m.outputs['Vector'], tex, 'Vector')
    return tex


def tissue_textures(tissue_name, tex_dir):
    """Find albedo/normal/rough for a tissue, falling back to its class.

    Exact name wins, then the longest REFERENCE_LOOK key contained in the name — so a
    single `lung_*` set serves all four lobes and `vena_cava_*` serves both cavae,
    instead of needing 29 near-identical texture sets on disk.
    """
    d = Path(tex_dir)
    names = [tissue_name]
    cls = None
    for k in REFERENCE_LOOK:
        if k in tissue_name and (cls is None or len(k) > len(cls)):
            cls = k
    if cls and cls != tissue_name:
        names.append(cls)

    found = {}
    for kind in ("albedo", "normal", "rough"):
        for nm in names:
            hit = None
            for ext in (".png", ".jpg", ".exr"):
                cand = d / f"{nm}_{kind}{ext}"
                if cand.exists():
                    hit = cand
                    break
            if hit:
                found[kind] = hit
                break
    return found


# ── Material creation ─────────────────────────────────────────────────────────

def make_material(t, feat):
    """Build the ground-truth shader for one tissue.

    `feat` carries the CLI feature flags so each layer can be bypassed for bisection.
    """
    name = t["name"]
    mat  = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    output     = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    _set(principled, 'IOR', t["ior"])

    tc      = nodes.new('ShaderNodeTexCoord')
    obj_vec = tc.outputs['Object']   # metres — see _mm_scale()

    # ── Base colour ──────────────────────────────────────────────────────────
    # AO contact darkening. Kept lighter than v20 (0.22 → 0.15): with real SSS in
    # place, crushing the albedo before light transport double-darkens the creases.
    ao = nodes.new('ShaderNodeAmbientOcclusion')
    _set(ao, 'Distance', 0.0025)
    col = _mix_rgb(nodes, links, 'MULTIPLY', 0.15,
                   a_val=tuple(t["base"]), b_out=ao.outputs['Color'])
    color_out = col.outputs[2]

    # Saturation pre-compensation for the AgX inset transform (see docstring §4).
    hsv = nodes.new('ShaderNodeHueSaturation')
    _set(hsv, 'Saturation', feat["saturation"])
    links.new(color_out, hsv.inputs['Color'])
    color_out = hsv.outputs['Color']

    # Synthesised albedo REPLACES the flat base colour rather than multiplying into it.
    # v24 mixed MULTIPLY at fac 0.85, i.e. result = A*0.15 + A*B*0.85 with A = texture:
    # texture-dominant and washed, and the palette barely survived. The maps from
    # synth_tissue_textures.py are already generated in each organ's palette, so they
    # are the base colour. --tex_tint blends back toward the table colour if a subject
    # needs nudging.
    tex = tissue_textures(t["name"], feat["tex_dir"])
    has_albedo = "albedo" in tex
    if has_albedo:
        ta = _box_tex(nodes, links, obj_vec, tex["albedo"], feat["tex_mm"], False)
        if feat["tex_tint"] > 0.0:
            blend = _mix_rgb(nodes, links, 'MIX', feat["tex_tint"],
                             a_out=ta.outputs['Color'], b_out=color_out)
            color_out = blend.outputs[2]
        else:
            color_out = ta.outputs['Color']

    # The texture already contains the vasculature and the blood-content variation, so
    # running the procedural layers on top would double them — two unrelated vessel
    # networks superimposed, which reads worse than either alone.
    if feat["perfusion"] and t["perfusion"] > 0 and not has_albedo:
        color_out = build_perfusion(nodes, links, obj_vec, t, color_out)

    vessel_height = None
    if feat["vessels"] and t["vessel"] > 0 and not has_albedo:
        color_out, vessel_height = build_vessels(
            nodes, links, obj_vec, t, color_out, feat["vessel_gain"])

    links.new(color_out, principled.inputs['Base Color'])

    # ── Roughness ────────────────────────────────────────────────────────────
    # Slight patchiness at the capsule-wrinkle scale. Real serosal surfaces are not
    # uniformly wet — exposed ridges dry, recesses pool. Kept narrow (±20 %) so it
    # reads as surface variation, not as a second texture.
    if feat["micro"]:
        n_r = _noise(nodes, links, obj_vec, t["meso_mm"] * 1.6, _phase(name, 10),
                     detail=3.0)
        r_mod = _map_range(nodes, links, n_r.outputs['Fac'],
                           t["rough"] * 0.80, t["rough"] * 1.20)
        links.new(r_mod.outputs['Result'], principled.inputs['Roughness'])
    else:
        _set(principled, 'Roughness', t["rough"])

    # ── Subsurface scattering ────────────────────────────────────────────────
    # v20 excluded hollow organs entirely; v21 does not (see the bowel note in the
    # tissue table — a 2-4 mm translucent wall is among the most translucent things
    # in the field, especially at grazing angles).
    sss_w = t["sss"] if feat["sss_fix"] else min(t["sss"], 0.05)
    if sss_w > 0 and t["sss_mm"] > 0:
        try:
            principled.subsurface_method = feat["sss_method"]
        except Exception:
            principled.subsurface_method = 'RANDOM_WALK'
        _set(principled, 'Subsurface Weight', sss_w)
        _set(principled, 'Subsurface Scale',  t["sss_mm"] * 0.001)   # mm → m
        _set(principled, 'Subsurface Radius', tuple(t["sss_rgb"]))
        # Absent on 4.0 and on the SKIN method — _set() no-ops if so.
        _set(principled, 'Subsurface Anisotropy', t["sss_aniso"])
        # No-op on Blender 5.x — the socket was removed. Harmless; kept for 4.x.
        _set(principled, 'Subsurface IOR', t["ior"])

    # ── Wet film ─────────────────────────────────────────────────────────────
    if t["coat"] > 0:
        _set(principled, 'Coat Weight',    t["coat"])
        _set(principled, 'Coat Roughness', t["coat_rough"])
        _set(principled, 'Coat IOR',       1.41)
        # Serous fluid is faintly straw-coloured, not water-clear.
        _set(principled, 'Coat Tint', (*t["coat_tint"], 1.0))

    # Sheen: grazing-angle brightening from the fibrous surface layer. Absent in v20,
    # and it is a large part of why muscle and serosa read as velvet rather than vinyl.
    if t["sheen"] > 0:
        _set(principled, 'Sheen Weight',    t["sheen"])
        _set(principled, 'Sheen Roughness', t["sheen_rough"])

    _set(principled, 'Specular IOR Level', 0.5)

    # ── Normals ──────────────────────────────────────────────────────────────
    # Bevel first: softens the staircase edges left by marching cubes on 1.5 mm CT.
    bevel = nodes.new('ShaderNodeBevel')
    bevel.samples = 4
    _set(bevel, 'Radius', 0.00012)
    normal_out = bevel.outputs['Normal']
    coat_normal_out = None

    if feat["micro"] and "normal" not in tex:
        normal_out, coat_normal_out = build_micro_normals(
            nodes, links, obj_vec, t, normal_out, feat["detail"])

    # Vessels sit slightly proud of the capsule — shallow, but it stops them reading
    # as a printed decal at grazing angles.
    if vessel_height is not None:
        # Normals are NOT blurred by subsurface transport, so relief carries the
        # vessel read even where the albedo tint gets washed out. Hence the increase.
        vb = _bump(nodes, links, vessel_height, normal_out, 0.55, 0.00035)
        normal_out = vb.outputs['Normal']

    # Legacy UV bump from data/renders/textures, if the file exists. Retained from v20
    # but demoted: it depends on the quality of the *_uv.obj unwrap, whereas the
    # object-space chain above does not.
    if feat["legacy_bump"] and t["bump_type"] != "none" and t["bump_scale"] > 0:
        bump_path = TEX_DIR / f"bump_{t['bump_type']}.png"
        if bump_path.exists():
            map_b = nodes.new('ShaderNodeMapping')
            _set(map_b, 'Scale', (3.5, 3.5, 3.5))
            _link_to(links, tc.outputs['UV'], map_b, 'Vector')

            tex_b = nodes.new('ShaderNodeTexImage')
            tex_b.image = bpy.data.images.load(str(bump_path))
            tex_b.image.colorspace_settings.name = 'Non-Color'
            tex_b.extension = 'REPEAT'
            _link_to(links, map_b.outputs['Vector'], tex_b, 'Vector')

            # Halved vs v20 — the object-space chain now carries most of the relief.
            tb = _bump(nodes, links, tex_b.outputs['Color'], normal_out,
                       t["bump_scale"] * 0.55, 0.00022)
            normal_out = tb.outputs['Normal']

    # A real normal map carries structure procedural noise cannot invent, so it goes
    # last and overrides the procedural relief where present.
    if "normal" in tex:
        tn = _box_tex(nodes, links, obj_vec, tex["normal"], feat["tex_mm"], True)
        nm = nodes.new('ShaderNodeNormalMap')
        _set(nm, 'Strength', 0.8)
        _link_to(links, tn.outputs['Color'], nm, 'Color')
        normal_out = nm.outputs['Normal']
    if "rough" in tex:
        tr = _box_tex(nodes, links, obj_vec, tex["rough"], feat["tex_mm"], True)
        rr = _map_range(nodes, links, tr.outputs['Color'],
                        t["rough"] * 0.6, t["rough"] * 1.4)
        _link_to(links, rr.outputs['Result'], principled, 'Roughness')

    links.new(normal_out, principled.inputs['Normal'])
    if coat_normal_out is not None:
        _link_to(links, coat_normal_out, principled, 'Coat Normal')

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ── Scene helpers ─────────────────────────────────────────────────────────────

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def configure_gpu(device):
    """Enable a Cycles compute backend explicitly.

    scene.cycles.device = 'GPU' only expresses intent — if no compute device is ticked
    in preferences, Cycles falls back to CPU without saying so. v20 never set this, so
    it is worth checking whether your dataset generation was actually GPU-bound.

    OptiX is preferred over CUDA on the L4s: RT-core traversal plus a much better
    denoiser than the CPU OIDN path.
    """
    if device != 'GPU':
        return
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        chosen = None
        for backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
            try:
                prefs.compute_device_type = backend
            except Exception:
                continue
            for refresh in ('refresh_devices', 'get_devices'):
                try:
                    getattr(prefs, refresh)()
                    break
                except Exception:
                    continue
            if any(getattr(d, 'type', None) == backend for d in prefs.devices):
                chosen = backend
                break

        if chosen is None:
            print("  [warn] no GPU backend found — Cycles will render on CPU")
            return

        enabled = 0
        for d in prefs.devices:
            d.use = (getattr(d, 'type', None) == chosen)
            if d.use:
                enabled += 1
                print(f"  GPU: {d.name} [{d.type}]")
        print(f"  Cycles compute: {chosen} — {enabled} device(s)")
    except Exception as e:
        print(f"  [warn] GPU configuration failed ({e}); leaving Blender's default")


def setup_render(args, feat):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.spp
    configure_gpu(args.device)
    scene.cycles.device  = args.device
    scene.render.film_transparent = False

    # ── Denoising ────────────────────────────────────────────────────────────
    # v20 rendered 384 spp with denoising OFF. Random-walk SSS is a high-variance
    # estimator, so the residual chroma grain is a direct CG tell. OIDN with albedo
    # and normal guides removes it without eating the micro-relief added above —
    # unguided denoising would flatten exactly the detail this version exists for.
    if feat["denoise"]:
        scene.cycles.use_denoising = True
        for prop, val in (("denoiser", 'OPENIMAGEDENOISE'),
                          ("denoising_input_passes", 'RGB_ALBEDO_NORMAL'),
                          ("denoising_prefilter", 'ACCURATE'),
                          ("denoising_quality", 'HIGH'),
                          ("denoising_use_gpu", args.device == 'GPU')):
            try:
                setattr(scene.cycles, prop, val)
            except Exception:
                pass
    else:
        scene.cycles.use_denoising = False

    scene.cycles.max_bounces             = 12
    # 2 → 4: organs in a cavity are lit substantially by red light bouncing off each
    # other. Truncating at 2 is why v20's shadowed faces go dead grey instead of warm.
    scene.cycles.diffuse_bounces         = 4
    scene.cycles.glossy_bounces          = 4
    scene.cycles.transmission_bounces    = 8
    scene.cycles.volume_bounces          = 2
    scene.cycles.transparent_max_bounces = 12
    # 0.2 → 0.02: blur_glossy smears the sharp wet-film glints this version creates.
    scene.cycles.blur_glossy             = 0.02
    try:
        scene.cycles.sample_clamp_indirect = 4.0    # firefly control, keeps highlights
    except Exception:
        pass

    # BOX 0.5 is a nearest-neighbour-ish filter: aliased AND soft, the worst of both.
    # Blackman-Harris at 1.5 px is what a real sensor's reconstruction looks like.
    try:
        scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
        scene.cycles.filter_width      = 1.5
    except Exception:
        scene.cycles.pixel_filter_type = 'GAUSSIAN'
        scene.cycles.filter_width      = 1.5

    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.image_settings.file_format = 'PNG'

    setup_world(scene, feat)

    # ── View transform — this IS the training target, not a preview ──────────
    if feat["tone_fix"]:
        vt, look = args.view_transform, args.look
    else:
        vt, look = 'AgX', 'AgX - Medium Contrast'
    try:
        scene.view_settings.view_transform = vt
    except Exception:
        scene.view_settings.view_transform = 'AgX'
    for candidate in (look, 'AgX - Punchy', 'AgX - Medium High Contrast',
                      'AgX - Medium Contrast', 'None'):
        try:
            scene.view_settings.look = candidate
            break
        except Exception:
            continue
    scene.view_settings.exposure = args.exposure

    scene.unit_settings.system = 'METRIC'


def setup_world(scene, feat):
    """Environment.

    v20 used a near-black background (strength 0.01). With nothing around the subject
    every shadow terminates in pure black and the silhouette reads as a cut-out — one
    of the most reliable giveaways of a CG render. Real gross-pathology and
    intraoperative photographs always carry fill from drapes, walls and the operator.

    This builds a dim vertical gradient: warm red-brown below (bounce off blood and
    the surgical field) and cool neutral above (room and overhead lighting).
    """
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    wt = scene.world.node_tree
    wt.nodes.clear()
    wout = wt.nodes.new('ShaderNodeOutputWorld')
    bg   = wt.nodes.new('ShaderNodeBackground')

    if feat["env"]:
        tc  = wt.nodes.new('ShaderNodeTexCoord')
        sep = wt.nodes.new('ShaderNodeSeparateXYZ')
        wt.links.new(tc.outputs['Generated'], sep.inputs['Vector'])

        ramp = wt.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color    = (0.060, 0.022, 0.016, 1)  # warm below
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color    = (0.045, 0.050, 0.058, 1)  # cool above

        mr = wt.nodes.new('ShaderNodeMapRange')
        mr.inputs['From Min'].default_value = -1.0
        mr.inputs['From Max'].default_value =  1.0
        wt.links.new(sep.outputs['Z'], mr.inputs['Value'])
        wt.links.new(mr.outputs['Result'], ramp.inputs['Fac'])
        wt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
        bg.inputs['Strength'].default_value = 0.85
    else:
        bg.inputs['Color'].default_value    = (0.001, 0.001, 0.001, 1)
        bg.inputs['Strength'].default_value = 0.01

    wt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # Atmosphere. v20's (0.72, 0.75, 0.82) at 0.0015 lays a cool milky veil over the
    # field; halved and warmed so it reads as air in a lit room, not as fog.
    vol = wt.nodes.new('ShaderNodeVolumeScatter')
    vol.inputs['Color'].default_value      = (0.80, 0.76, 0.72, 1)
    vol.inputs['Density'].default_value    = 0.00025
    vol.inputs['Anisotropy'].default_value = 0.35
    wt.links.new(vol.outputs['Volume'], wout.inputs['Volume'])


def setup_compositor(scene, feat):
    """Post. Both effects here are screen-space and positional.

    A convolutional generator is translation-equivariant and cannot represent a
    function of image coordinates, so chromatic aberration and glare are unlearnable
    and act as label noise during training. --training_safe disables them; they are
    worth keeping for thesis figures.
    """
    try:
        scene.use_nodes = True
        tree = scene.node_tree
        if tree is None:
            return
        tree.nodes.clear()
        rl  = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('CompositorNodeComposite')

        if feat["training_safe"]:
            tree.links.new(rl.outputs['Image'], out.inputs['Image'])
            return

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
    except Exception:
        pass


def teardown_compositor(scene):
    try:
        scene.use_nodes = False
    except Exception:
        pass


# ── Negative fill / cavity walls ──────────────────────────────────────────────

def add_fill_planes(cx, cy, cz, scene_scale, feat, cam_radius=None):
    """v20 used pure black planes, which absorb everything that hits them.

    A perfectly black surround is not a negative fill, it is a light sink — it removes
    the wrap-around bounce that makes flesh look soft. These are still far darker than
    any tissue (so they still shape the form) but return a faint warm bounce.
    """
    # The walls must sit OUTSIDE the camera or they occlude it and the frame renders
    # black. Before v24, radius was derived from the same quantity as scene_scale, so
    # walls at 2.0 * scene_scale were always beyond the camera. v24 frames from the
    # organ bounding box, which makes the camera radius independent of scene_scale and
    # frequently much larger — so the wall distance has to track the camera, not the
    # subject.
    sc = scene_scale
    wall_d = max(sc * 2.0, (cam_radius or 0.0) * 1.8)
    span   = max(sc * 4.0, wall_d * 2.4)
    mat = bpy.data.materials.new("CavityWall")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        col = (0.030, 0.012, 0.010, 1) if feat["env"] else (0, 0, 0, 1)
        _set(bsdf, 'Base Color', col)
        _set(bsdf, 'Roughness',  0.9)
        _set(bsdf, 'Specular IOR Level', 0.1)

    def make_plane(loc, rot_euler, name):
        bpy.ops.mesh.primitive_plane_add(size=span, location=loc)
        p = bpy.context.object
        p.rotation_euler = rot_euler
        p.data.materials.append(mat)
        p.name = name

    make_plane((cx - wall_d, cy + sc*0.2, cz), (0, math.pi/2, 0), "Wall_Left")
    make_plane((cx + wall_d, cy + sc*0.2, cz), (0, math.pi/2, 0), "Wall_Right")


# ── Lighting ──────────────────────────────────────────────────────────────────

def _track_to(obj, target_xyz):
    import mathutils
    direction = mathutils.Vector(target_xyz) - mathutils.Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def setup_lights(cx, cy, cz, scene_scale, feat, key_energy=90.0):
    """Three-point rig plus a warm cavity bounce, with size-invariant exposure.

    Light POSITIONS have always scaled with the scene, but the ENERGIES were absolute
    watts. Irradiance falls off as 1/d^2, so a subject framed at half the distance came
    out four times brighter — which is why exposure drifted between subjects, and why
    it shifted again when v24 started framing from the organ bounding box instead of
    the CT field of view. Scaling energy by scene_scale^2 makes the exposure depend on
    the material and the tone map only. Normalised at 0.40 m so the defaults below are
    numerically unchanged for a typical torso.

    Fill and environment are deliberately generous. The reference renders read
    "translucent" not because light passes through 5 cm of liver — it does not, the
    transport mean free path is ~1-3 mm and a backlit test confirms both 0.3x and 0.9x
    SSS scale render an opaque silhouette — but because soft wraparound fill keeps the
    shading terminator from going dead. That softness is a lighting property, not a
    subsurface one.
    """
    sc = scene_scale
    # Normalisation point matters as much as the scaling law. Irradiance goes as
    # P/d^2 and d scales with scene_scale, so P ~ scene_scale^2 keeps exposure constant
    # — but the CONSTANT was calibrated against 0.40 m while v20's flat 90 W was
    # implicitly calibrated against the CT extent (~0.8 m). Since v24 switched
    # scene_scale to the smaller organ extent, the lights moved closer AND got scaled
    # up, compounding to roughly 4x too bright: the scene measured mean RGB 212/255
    # where it should sit near 130. 0.78 m restores v20's light level.
    _e = (scene_scale / 0.78) ** 2      # size-invariant exposure

    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*1.4, cy + sc*0.3, cz + sc*0.8))
    key = bpy.context.object
    key.data.energy = key_energy * _e
    key.data.color  = (1.00, 0.98, 0.96)
    key.data.size   = sc * 0.25
    key.data.shape  = 'SQUARE'
    _track_to(key, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*1.0, cy - sc*1.0, cz + sc*0.5))
    fill = bpy.context.object
    fill.data.energy = 15.0 * _e
    fill.data.color  = (0.96, 0.97, 1.00)
    fill.data.size   = sc * 1.10
    _track_to(fill, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx - sc*0.5, cy + sc*1.4, cz + sc*0.3))
    rim1 = bpy.context.object
    rim1.data.energy = 20 * _e
    rim1.data.color  = (1.00, 0.94, 0.88)
    rim1.data.size   = sc * 0.02
    _track_to(rim1, (cx, cy, cz))

    bpy.ops.object.light_add(type='AREA',
        location=(cx + sc*0.4, cy - sc*1.4, cz - sc*0.2))
    rim2 = bpy.context.object
    rim2.data.energy = 21 * _e
    rim2.data.color  = (0.95, 0.95, 1.00)
    rim2.data.size   = sc * 0.08
    _track_to(rim2, (cx, cy, cz))

    # NEW — cavity bounce. In any real open body cavity a large fraction of the fill
    # arrives as saturated red light that has already passed through or reflected off
    # blood and tissue. Without it the shadow side goes neutral grey, which is the
    # single most consistent difference between v20 and an intraoperative photograph.
    if feat["env"]:
        bpy.ops.object.light_add(type='AREA',
            location=(cx + sc*0.2, cy + sc*0.1, cz - sc*1.1))
        bounce = bpy.context.object
        bounce.data.energy = 30.0 * _e
        bounce.data.color  = (1.00, 0.34, 0.22)
        bounce.data.size   = sc * 1.20
        _track_to(bounce, (cx, cy, cz))


# ── Camera ────────────────────────────────────────────────────────────────────

def setup_camera(fstop, fov_deg=18):
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam_obj = bpy.context.scene.camera = bpy.data.objects['Camera']
    cam_obj.data.type      = 'PERSP'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle     = math.radians(fov_deg)
    cam_obj.data.clip_start = 0.001
    cam_obj.data.clip_end   = 100.0
    cam_obj.data.dof.use_dof = True
    # f/6.3 at this FOV put most of the field outside the focal plane — the v20 crop
    # is soft everywhere, which reads as CG depth-of-field rather than as a photograph.
    # Clinical and gross-pathology macro work stops down to f/8-f/16 for exactly this.
    cam_obj.data.dof.aperture_fstop = fstop
    cam_obj.data.shift_x = -0.05
    cam_obj.data.shift_y =  0.02
    return cam_obj


def point_camera(cam_obj, position, target):
    cam_obj.location = position
    _track_to(cam_obj, target)
    dist = math.sqrt(sum((a - b)**2 for a, b in zip(position, target)))
    cam_obj.data.dof.focus_distance = dist * 0.94
    bpy.context.view_layer.update()


# ── Mesh import ───────────────────────────────────────────────────────────────

def destaircase(obj, iterations, factor):
    """Remove the 1.5 mm marching-cubes ripple from geometry AND shading normals.

    Two separate things have to happen, and doing only the first is why this artefact
    looked shader-shaped:

    1. The OBJ ships explicit vertex normals, which Blender imports as CUSTOM SPLIT
       NORMALS. Those override anything the geometry says, so the staircase keeps
       shading even after the vertices move. They must be cleared first.
    2. Laplacian smoothing of the positions themselves. factor/iterations default to
       the values Bade et al. report for liver (lambda 0.5, ~20 iterations).

    Laplacian smoothing shrinks volume slightly. That is acceptable here — the
    segmentation boundary is already +/- one voxel — but it is the reason the proper
    fix in extract_meshes.py should use Taubin, which does not shrink.
    """
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception:
            pass
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
        if iterations > 0:
            mod = obj.modifiers.new(name="Destaircase", type='SMOOTH')
            mod.factor     = factor
            mod.iterations = iterations
        obj.select_set(False)
    except Exception as e:
        print(f"  [warn] smoothing failed on {obj.name}: {e}")


def import_obj(obj_path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis='Y',
        up_axis='Z',
        global_scale=0.001,   # mm → m. This is why object coords are metres.
    )
    new_objs = [o for o in bpy.data.objects if o.name not in before]
    return new_objs[0] if new_objs else None


# ── Simple render (flat EEVEE) — unchanged from v20 ───────────────────────────

def setup_simple_material(seg_name, simple_hex):
    mat = bpy.data.materials.new(name=f"{seg_name}_simple")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output  = nodes.new('ShaderNodeOutputMaterial')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    r = int(simple_hex[1:3], 16) / 255.0
    g = int(simple_hex[3:5], 16) / 255.0
    b = int(simple_hex[5:7], 16) / 255.0
    diffuse.inputs['Color'].default_value     = (r, g, b, 1.0)
    diffuse.inputs['Roughness'].default_value = 0.8
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return mat


def render_simple(objs_with_mats, cam_obj, out_path, feat):
    scene = bpy.context.scene
    orig_engine = scene.render.engine
    teardown_compositor(scene)
    cam_obj.data.dof.use_dof = False
    # Blender 5.x reverted the identifier to BLENDER_EEVEE; 4.2-4.5 used _NEXT.
    for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    try:
        scene.eevee.taa_render_samples = 4
    except Exception:
        pass
    for obj, simple_mat, _ in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(simple_mat)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    scene.render.engine = orig_engine
    cam_obj.data.dof.use_dof = True
    setup_compositor(scene, feat)


def restore_gt_materials(objs_with_mats):
    for obj, _, gt_mat in objs_with_mats:
        obj.data.materials.clear()
        obj.data.materials.append(gt_mat)


# ── Grid helpers ──────────────────────────────────────────────────────────────

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
    rgba[:, :, 3]  = 1.0
    rgba = np.flipud(rgba)
    img.pixels = rgba.flatten().tolist()
    img.filepath_raw = str(path)
    img.file_format  = 'PNG'
    img.save()
    bpy.data.images.remove(img)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    feat = dict(
        sss_fix       = args.sss_fix,
        micro         = args.micro,
        vessels       = args.vessels,
        perfusion     = args.perfusion,
        env           = args.env,
        denoise       = args.denoise,
        tone_fix      = args.tone_fix,
        detail        = args.detail,
        vessel_gain   = args.vessel_gain,
        saturation    = args.saturation if args.tone_fix else 0.95,
        sss_method    = args.sss_method,
        training_safe = args.training_safe,
        legacy_bump   = args.legacy_bump,
        tex_tint      = args.tex_tint,
        tex_dir       = args.tex_dir,
        tex_mm        = args.tex_mm,
    )

    # Global table corrections. Applied here rather than edited into the 29 rows so the
    # v21 -> v22 delta stays visible and tunable from the command line.
    for _t in TISSUES:
        _t["base"]   = [c * args.albedo for c in _t["base"]]
        _t["sss_mm"] = _t["sss_mm"] * args.sss_scale

    mesh_dir = Path(args.mesh_dir) / args.subject
    out_dir  = Path("data/renders") / args.subject
    pair_out = Path("results/totalseg_pairs")
    pair_out.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_dir.exists():
        print(f"ERROR: mesh_dir not found: {mesh_dir}")
        sys.exit(1)

    enabled = [k for k in ("sss_fix", "micro", "vessels", "perfusion", "env",
                           "denoise", "tone_fix") if feat[k]]
    print(f"\n{'='*72}")
    print(f"[{args.tag}-photoreal]  subject={args.subject}  spp={args.spp}  size={args.size}px")
    print(f"  layers    : {', '.join(enabled) if enabled else 'none (v20 baseline)'}")
    print(f"  detail    : {args.detail:.2f}   vessels ×{args.vessel_gain:.2f}"
          f"   saturation {feat['saturation']:.2f}")
    print(f"  corrections: albedo ×{args.albedo:.2f}  sss_scale ×{args.sss_scale:.2f}")
    print(f"  legacy 8-bit UV bump: {'ON (banding expected)' if args.legacy_bump else 'OFF'}")
    print(f"  geometry  : destaircase {args.smooth} iters @ lambda {args.smooth_factor:.2f}"
          f"   textures: {args.tex_dir} @ {args.tex_mm:.0f}mm tile")
    print(f"  tone      : {args.view_transform} / {args.look}  exposure {args.exposure:+.2f} EV")
    print(f"  sss       : {args.sss_method}   camera f/{args.fstop:g}")
    print(f"{'='*72}")

    # ── Frame from the ORGANS, not the CT volume ─────────────────────────────
    # v20-v23 derived the camera distance from max(nx*sx, ny*sy, nz*sz), i.e. the
    # scan's field of view. That varies enormously between subjects — a whole-body CT
    # and a tight abdominal CT can contain identical organs — so the same anatomy came
    # out zoomed-out on one subject and cropped on another. Framing from the union
    # bounding box of the loaded organ meshes makes the shot consistent by
    # construction, and it is what the training set actually wants.
    #
    # This forces a reorder: meshes must be imported BEFORE the camera exists, and the
    # materials must be built after MM_PER_PX is known (make_material caps its noise
    # octaves against it — see _safe_detail).
    reset_scene()
    setup_render(args, feat)

    print("\n[1/3] Importing meshes ...")
    loaded = []
    for t in TISSUES:
        obj_path = mesh_dir / f"{t['name']}_uv.obj"
        if not obj_path.exists():
            obj_path = mesh_dir / f"{t['name']}.obj"
        if not obj_path.exists():
            continue
        blender_obj = import_obj(obj_path)
        if blender_obj is None:
            continue
        destaircase(blender_obj, args.smooth, args.smooth_factor)
        loaded.append((blender_obj, t))

    if not loaded:
        print("ERROR: no meshes matched the tissue table.")
        sys.exit(1)

    import mathutils
    corners = [obj.matrix_world @ mathutils.Vector(c)
               for obj, _ in loaded for c in obj.bound_box]
    lo = mathutils.Vector((min(c[i] for c in corners) for i in range(3)))
    hi = mathutils.Vector((max(c[i] for c in corners) for i in range(3)))
    ctr = (lo + hi) * 0.5
    cx, cy, cz = ctr.x, ctr.y, ctr.z
    extent      = max(hi[i] - lo[i] for i in range(3))

    # Solve for the camera radius that makes the organs fill `--frame` of the frame.
    # cam_pos_at_angle() sits at (0.6r, 1.15r, 0.45r) from centre, so the camera
    # distance is |(0.6, 1.15, 0.45)| * r, and the frame covers 2*d*tan(fov/2) metres.
    #   frame_m = 2 * K * r * tan(fov/2)   with K = |(0.6, 1.15, 0.45)| = 1.3739
    # Wanting frame_m = extent / fill gives r = extent / (fill * 2 * K * tan(fov/2)).
    #
    # The first cut of this multiplied extent by --frame directly, which made the
    # radius SMALLER than the subject: at frame 0.62 a 56 cm torso was shot with a
    # 15 cm frame. --frame is now the fraction of the frame the organs occupy, which
    # is what the name implies and is bounded in (0, 1].
    _K   = math.sqrt(0.6**2 + 1.15**2 + 0.45**2)
    _fov = math.radians(18.0)
    fill = min(max(args.frame, 0.05), 1.0)
    radius      = extent / (fill * 2.0 * _K * math.tan(_fov / 2.0))
    scene_scale = extent
    print(f"  organ bbox : {extent*100:.1f} cm across, centre "
          f"({cx:.3f}, {cy:.3f}, {cz:.3f}) m")

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

    # Image-plane sampling rate at the subject, from the actual camera geometry.
    # Everything procedural is capped against this so no octave lands below ~2 px.
    global MM_PER_PX
    _cam_d    = math.dist(cam_pos_at_angle(offsets[0]), (cx, cy, cz))
    _frame_m  = 2.0 * _cam_d * math.tan(math.radians(18.0) / 2.0)
    MM_PER_PX = (_frame_m * 1000.0) / float(args.size)
    print(f"  sampling   : {MM_PER_PX:.3f} mm/px  "
          f"(frame {_frame_m*100:.1f} cm at {args.size} px)")

    cam_obj = setup_camera(args.fstop)
    setup_lights(cx, cy, cz, scene_scale, feat, args.key_energy)
    if args.walls:
        add_fill_planes(cx, cy, cz, scene_scale, feat, cam_radius=radius)
    setup_compositor(bpy.context.scene, feat)

    # Materials last — they depend on MM_PER_PX.
    objs_with_mats = []
    for blender_obj, t in loaded:
        simple_mat = setup_simple_material(t["name"], t["hex"])
        gt_mat     = make_material(t, feat)
        blender_obj.data.materials.clear()
        blender_obj.data.materials.append(gt_mat)
        objs_with_mats.append((blender_obj, simple_mat, gt_mat))
        _tx = tissue_textures(t["name"], args.tex_dir)
        _tag = ("tex:" + "+".join(sorted(_tx))) if _tx else "procedural"
        print(f"  {t['name']:<34} sss={t['sss']:.2f}@{t['sss_mm']:.1f}mm  "
              f"coat={t['coat']:.2f}  vessel={t['vessel']:.2f}  {_tag}")

    print(f"\n  {len(objs_with_mats)} tissues loaded")
    if not objs_with_mats:
        print("ERROR: no meshes matched the tissue table.")
        sys.exit(1)

    print("\n[2/3] Rendering angles ...")
    angle_rows = []

    for theta in offsets:
        label = f"{theta:+.0f}°"
        print(f"\n--- Angle {label} ---")
        point_camera(cam_obj, cam_pos_at_angle(theta), (cx, cy, cz))

        # The EEVEE simple/seg pass needs a GL context, which headless servers without
        # /dev/dri access cannot provide (libEGL "Permission denied" → EGL_BAD_MATCH).
        # Its output is identical to v20 anyway — same hex palette, same flat diffuse —
        # so --gt_only skips it and goes straight to Cycles, which is CPU/CUDA only.
        simple_path = None
        if not args.gt_only:
            simple_path = out_dir / f"simple_{args.tag}_{label}.png"
            render_simple(objs_with_mats, cam_obj, simple_path, feat)
            restore_gt_materials(objs_with_mats)
            print(f"  Simple → {simple_path.name}")

        gt_path = out_dir / f"gt_{args.tag}_spp{args.spp}_{label}.png"
        bpy.context.scene.render.filepath = str(gt_path)
        bpy.ops.render.render(write_still=True)
        print(f"  GT     → {gt_path.name}")

        angle_rows.append((simple_path, gt_path, label))

    print("\n[3/3] Assembling grid ...")
    gap, label_h = 15, 40
    sz = args.size

    # When --gt_only skipped the EEVEE pass, fall back to the v20 simple render for the
    # left column if one is already on disk — it is the same image either way.
    resolved = []
    for sp, gp, lbl in angle_rows:
        if sp is None:
            for cand in (out_dir / f"simple_v20_{lbl}.png",
                         out_dir / f"simple_v21_{lbl}.png"):
                if cand.exists():
                    sp = cand
                    break
        resolved.append((sp, gp, lbl))

    two_col = any(sp is not None and sp.exists() for sp, _, _ in resolved)
    width   = (sz*2 + gap) if two_col else sz
    n       = len(resolved)
    grid = np.zeros(((sz + label_h) * n + gap, width, 3), dtype=np.uint8)
    grid[:] = 10

    for i, (sp, gp, _lbl) in enumerate(resolved):
        y0 = i * (sz + label_h) + gap
        gt_x0 = (sz + gap) if two_col else 0
        if two_col and sp is not None and sp.exists():
            grid[y0+label_h : y0+label_h+sz, 0:sz] = load_png_as_numpy(sp)[:sz, :sz]
        if gp.exists():
            grid[y0+label_h : y0+label_h+sz, gt_x0:gt_x0+sz] = load_png_as_numpy(gp)[:sz, :sz]

    grid_path = pair_out / f"{args.subject}_{args.tag}_photoreal_spp{args.spp}.png"
    save_numpy_as_png(grid, grid_path)
    print(f"\nGrid → {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
