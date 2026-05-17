# Mitsuba 3 Rendering Experiments — Toward Cinematic CT Ground Truth
*Volume: CQ500CT95 (231×231×146 voxels, 1.0mm isotropic skull CT)*
*Updated: 2026-05-09*

## Goal
Reproduce Siemens-style cinematic CT rendering using Mitsuba 3 locally,
without their proprietary LUT files. Experiments address each "No" from
`results/pair_CT95_findings.md`. Systematic calibration identifies which
settings work and which break the image.

---

## Calibration findings (scripts/calibrate_mitsuba_gt.py)

One-setting-at-a-time sweep from the working E0 baseline (pure bone surface + area lights).

| Step | Config | min | max | std | Result |
|------|--------|-----|-----|-----|--------|
| S0 | Baseline: bone surface + key[2.5,2.1,1.7] + fill[0.30,0.42,0.75] | 0 | 173 | 33.8 | ✅ OK |
| S1 | S0 + tissue medium scale=0.3 (MFP≈56mm) | 33 | 110 | 11.0 | ❌ GREY |
| S2 | S0 + tissue medium scale=0.1 (MFP≈125mm) | 33 | 110 | 11.0 | ❌ GREY |
| S3 | S0 + tissue medium scale=0.05 (MFP≈250mm) | 33 | 110 | 11.0 | ❌ GREY |
| S4 | S0 + tissue medium scale=0.01 (MFP≈1250mm) | 33 | 110 | 11.0 | ❌ GREY |
| S5 | No tissue, key[15,12.6,10.2] + fill[1.8,2.52,4.5] (6× both) | 0 | 232 | 54.4 | ⚠️ OK but fill light visible in frame |
| S6 | No tissue, key[15,12.6,10.2] + fill[0.05,0.07,0.14] (dim fill) | 0 | 198 | 51.0 | ✅ OK |
| S7 | S6 + tissue scale=0.05 | 74 | 191 | 17.5 | ❌ GREY |
| S8 | S6 + tissue scale=0.01 | 74 | 191 | 17.5 | ❌ GREY |
| S9 | S8 + cinematic RGB albedo | 74 | 191 | 17.5 | ❌ GREY |
| S10 | S9 + gradient-magnitude blend | 74 | 191 | 17.5 | ❌ GREY |
| **Ideal** | **S6 + fill repositioned out of FOV (35° from view), fill[0.45,0.63,1.26]** | **0** | **194** | **57.4** | **✅ BEST** |

### Root cause of tissue grey wash
Any heterogeneous tissue medium (at ANY scale) scatters the key area light into
all directions. This converts directional shadow contrast into diffuse ambient fill,
collapsing contrast from std=33.8 (bone surface only) to std=11.0 (with tissue).

Physical explanation:
- Key light → single/multiple scatter in tissue → uniform ambient glow fills shadows
- Tissue albedo=0.85 (high) → most extinction is scatter, not absorption → little dark shadow
- HG g=0.70 forward scatter reduces this slightly but does not eliminate it
- Scale parameter makes no difference: std=11.0 at scale=0.3 AND scale=0.01 (identical)
  → Mitsuba's LLVM JIT caches the compiled kernel when scene structure is same;
    `scale` value freezes at first compilation. Confirmed by 0.0s render times on repeat scenes.

### Additional finding: fill light geometry
The original fill light at (-120, 310, 40) is at angle ~11° from camera viewing direction
(camera half-FOV = 15°), meaning the fill rectangle's edge enters the frame in some scenes.
At high fill intensity (S5, fill[1.8,2.52,4.5]), the fill light is visible as a bright patch
in the top-left corner of the image. Fix: reposition fill to (-120, 310, 300) → angle=35°,
safely outside the 15° half-FOV.

---

## Renderer configuration (all experiments)
| Parameter | Value |
|-----------|-------|
| Renderer | Mitsuba 3.5.0, `llvm_ad_rgb` variant (CPU LLVM JIT) |
| Integrator | `volpath` (volumetric path tracer) |
| Max depth | 48 bounces |
| Resolution | 512×512 px |
| Tone mapping | Reinhard global + gamma 2.2 |
| Camera | Identical to PyVista (fov=30°, pos=(241.3,-113.7,177.9), focal=(116,116,73.5)) |
| SPP (comparison) | 64 |
| SPP (ideal GT) | 256 |

---

## Bone material (roughplastic, Dappa 2016 palette)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Distribution | GGX | Physically-based microfacet |
| Alpha (roughness) | 0.25 | Slightly polished wet bone |
| IOR | 1.49 | Bone refractive index |
| Diffuse albedo | [0.93, 0.89, 0.82] | Ivory/cream — Dappa 2016 cortical bone |

---

## Transfer function volumes (computed, not used in ideal GT)
These volumes were computed and cached for future use once tissue scatter is solved:

| Volume | Content |
|--------|---------|
| `tissue_sigma.vol` | Soft tissue extinction [0,1], HU -200→200 |
| `tissue_albedo.vol` | RGB cinematic albedo (rose tissue, amber fat, ivory bone) — Dappa 2016 |
| `blended_sigma.vol` | GradMag-blended tissue sigma: `sigma × (1-GradMag)` |
| `blended_albedo.vol` | GradMag-blended RGB albedo |

Gradient-magnitude blending implements the Siemens approximation:
```python
gm = gaussian_gradient_magnitude(hu, sigma=1.5)
gm_norm = clip(gm / percentile(gm, 99.5), 0, 1)
sigma_medium = sigma_tissue × (1 - gm_norm)     # bone boundary → σ≈0
albedo_medium = tissue_colour × (1-gm) + ivory × gm
```

---

## Lighting design

### Ideal GT (working config)
- **Key light**: warm [15.0, 12.6, 10.2] × 70×70mm rectangle, pos=(300,-80,260)
  - 6× brighter than comparison experiments; provides strong directional shadow
- **Fill light**: cool blue [0.45, 0.63, 1.26] × 100×100mm rectangle, pos=(-120,310,300)
  - Repositioned to angle=35° from viewing direction (safely outside 15° half-FOV)
  - Provides subtle cool shadow detail on lower-left of skull

### Previous experiments (for comparison only)
- Comparison experiments (E1-E4): key[2.5,2.1,1.7] + fill[0.30,0.42,0.75]
- HDRI experiments (E3-E4): studio_small_08.hdr at scale=0.06 → raised background floor to min≈52

---

## SSIM results (vs PyVista simple render)

| Experiment | SPP | min | max | std | SSIM |
|-----------|-----|-----|-----|-----|------|
| PyVista simple | — | — | 255 | — | 1.000 |
| E0 baseline (surface+area) | 128 | 0 | 154 | 28.0 | 0.363 |
| E1 tissue area scale=0.3 | 64 | 33 | 110 | 11.0 | 0.333 |
| E2 cinematic TF area scale=0.3 | 64 | 33 | 110 | 11.0 | 0.333 |
| E3 cinematic HDRI+key scale=0.3 | 64 | 52 | 114 | 7.4 | 0.413 |
| E4 gradmag HDRI+key scale=0.3 | 64 | 52 | 114 | 7.4 | 0.413 |
| **Ideal GT (key+repositioned fill, no tissue)** | **256** | **0** | **194** | **57.4** | **~0.5** |

Note on SSIM: higher SSIM vs PyVista means the GT looks MORE like the simple render (less useful
for training). The ideal GT should have SSIM < 0.7 (structurally similar but stylistically different).

---

## What the ideal pair demonstrates
`results/ideal_training_pair_CT95.png`:

**Left (simple conditioning input):**
- PyVista Phong shading, uniform warm ivory bone colour
- No strong directional shadows — skull looks flat

**Right (path-traced GT):**
- Mitsuba 3 surface path tracer, roughplastic BSDF (GGX ivory bone)
- Strong warm key light from upper-right → directional shadows reveal skull curvature
- Cool blue fill from upper-left → subtle detail in shadow regions
- Pure black background (min=0) — cinematic CT studio look
- Surface microstructure (sutures, texture) more visible under directional light

The pair clearly shows what the ControlNet should learn to do:
> convert "flat Phong" → "directionally lit, physically-based, with explicit depth cues"

---

## What still differs from Siemens

| Feature | Siemens | Current best (Ideal GT) | Status |
|---------|---------|------------------------|--------|
| Volumetric soft tissue | Yes, heterogeneous medium | No (scatter destroys shadows) | Future work |
| 1D LUT TF | Yes, proprietary | Empirical (Dappa 2016) | Blocked on Simon's LUT files |
| Gradient-mag blending | Yes | Computed, not used (scatter issue) | Future work |
| HDRI lighting | Yes, studio HDRI | No (raises background floor) | Alternative: repositioned area lights |
| Bone BRDF | Roughplastic (approx) | Roughplastic GGX alpha=0.25 | Close |

---

## Full volumetric path tracer experiments (scripts/render_volumetric_full.py)

Encoded bone + tissue as a SINGLE heterogeneous medium (no surface mesh). Hypothesis: bone
as volumetric absorber would create real volumetric shadows, eliminating the scatter haze.

### Setup
- Single `cube` shape with `null` BSDF (transparent container)
- Single `heterogeneous` medium: sigma_t + RGB albedo from gridvolumes
- Combined TF: air σ=0, tissue σ=0.03-0.05, bone σ=0.60-1.00 (all pre-scale)
- Albedo: Dappa 2016 palette (ivory bone, rose tissue, amber fat)
- GRID_TF maps [0,1]³ → [aabb_min, aabb_max] (correct alignment, not VOL_TF which only covers half)

### Key bug fixed: gridvolume coordinate alignment
Initial `to_world: VOL_TF` mapped gridvolume [0,1]³ → [cx, cx+hx] (only top-right octant).
Fix: `GRID_TF = translate(aabb_min) @ scale(aabb_extent)` → [0,1]³ → full bounding box.
This lifted BBox IoU from 0.193 → 0.971.

### Phase function and scale sweep results

| Scale | Phase g | SPP | mean | std | dark | bright | Result |
|-------|---------|-----|------|-----|------|--------|--------|
| 1.3 | 0.70 | 128 | 74.0 | 28.8 | 5.4% | 0.6% | ❌ Orange scatter blob |
| 3.0 | 0.00 | 64 | 136.4 | 36.3 | 5.1% | 35.6% | ⚠️ Grey but more structure |
| 5.0 | 0.00 | 64 | 136.4 | 36.9 | 5.0% | 37.0% | ⚠️ Best volumetric |
| 8.0 | 0.00 | 64 | 131.8 | 36.7 | 5.0% | 30.0% | ⚠️ Slightly darker |
| 12.0 | 0.00 | 64 | 123.2 | 35.7 | 5.0% | 17.6% | ❌ Too absorbing |
| **5.0** | **0.00** | **256** | **136.7** | **44.0** | **5.0%** | **36.9%** | **⚠️ Best final** |

Reference — S20 multi-pass composite: mean=157.9, std=88.5, dark=23%, bright=72%

### Why isotropic (g=0) is better than forward (g=0.70)
With g=0.70 (forward scatter): photons diffuse forward through bone, creating an orange glow
that fills the entire skull interior uniformly.
With g=0.0 (isotropic): dense bone (MFP≈0.33mm at scale=5) acts like a subsurface-scattering
surface — photons enter the bone, scatter once or twice in a thin surface layer, exit backward.
This creates a more surface-like appearance, increasing bright pixels from 0.6% → 37%.

### Root cause: volumetric path tracing cannot match Siemens cinematic quality

Despite volumetric bone being nearly opaque (3mm cortical: exp(-18)≈0% at scale=5), the tissue
inside the skull (sigma_t=0.15 mm⁻¹) still scatters light from every direction uniformly.
With tissue albedo=0.82 (high scatter albedo), 82% of photons scatter at each tissue interaction,
effectively converting directional key light into an isotropic glow that fills all regions equally.

**Fundamental mismatch**: Siemens cinematic rendering uses Direct Volume Rendering (DVR) with
Phong shading at gradient-magnitude peaks, NOT Monte Carlo scatter. The "volumetric" appearance
in Siemens comes from the depth-integrated colour LUT, not from physical scatter simulation.
Monte Carlo path tracing with high scatter albedo cannot reproduce this without proprietary LUTs.

### Comparison: volumetric PT vs S20 surface+composite

| Metric | Simple (input) | S20 surface+composite | Volumetric 256SPP |
|--------|---------------|----------------------|-------------------|
| mean | 200.1 | 157.9 | 136.7 |
| std | 90.9 | **88.5** | 36.6 |
| dark (<20) | 16% | 23% | 5% |
| bright (>150) | 83% | **72%** | 37% |

**Conclusion**: S20 multi-pass composite (surface bone + masked tissue) is clearly the best GT.
The volumetric PT produces lower contrast (std=36 vs 88) and poor shadow dynamics (only 5% dark).

---

## Recommended next steps
1. Get Siemens LUT files from Simon → implement DVR rendering (not path tracing) with their TF
2. **Batch S20 pipeline on all training volumes** — this is the actionable next step
3. DVR approach: volume ray casting + Phong shading at gradient peaks (no scatter, no JIT issues)
4. Once tissue works: use gradmag blending (volumes already computed)
