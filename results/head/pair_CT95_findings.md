# One Pair Proof-of-Concept Findings — CQ500CT95
*Generated: 2026-05-09. Send with `results/pair_CT95_proof_of_concept.png`.*

## What was produced
A single paired training example from skull CT volume CQ500CT95:
- **Left (simple input):** PyVista isosurface render, HU=300 isosurface, Phong shading, warm bone colour
- **Right (path-traced GT):** Mitsuba 3 surface path tracer, roughplastic BRDF (bone), two-light setup (warm key + cool fill)

## Technical parameters
| Parameter | Value |
|-----------|-------|
| Volume | CQ500CT95, 231×231×146 voxels, 1.0mm isotropic |
| HU range | [−1000, 2853], bone (>200 HU) mean=920.6, std=472.0 |
| Renderer | Mitsuba 3.5.0, `llvm_ad_rgb` (CPU), `path` integrator |
| SPP | 128 (converged — surface path tracing, direct illumination dominant) |
| Material | `roughplastic` GGX, alpha=0.3, diffuse=[0.91, 0.86, 0.79] (warm bone) |
| Key light | Warm white area light (60×60mm), upper-right-front |
| Fill light | Cool blue area light (80×80mm), lower-left-back |
| Camera | Identical to PyVista: pos=(241.3, -113.7, 177.9), focal=(116,116,73.5), up=(0,0,1), fov=30° |
| Render time | 0.5s on Apple M-series CPU (LLVM JIT) |

## Alignment verification
- **BBox IoU = 0.980** (simple vs GT, threshold >0.80 = PASS)
- Simple bbox: rows 10–511, cols 0–506 (skull fills the frame)
- GT bbox: rows 0–511, cols 0–506 (same region)
- Camera parameters were computed identically for both renderers — no manual adjustment needed

## What the GT shows differently from simple render
- **Black background** (area lights only, no environment) vs black background (Phong) — similar intent
- **Soft shadows** from area lights reveal depth cues absent in Phong shading
- **Cool/warm colour contrast** (warm key light, cool fill) vs uniform warm bone colour
- **Roughplastic BRDF** (specular highlights + diffuse) vs Phong specular — more physically correct
- **Same skull geometry** — no hallucination, no shape change — geometry preserved ✓

## What this approach does NOT yet model (noted for thesis)
1. **No volumetric rendering of soft tissue** — only the bone isosurface is rendered. Siemens renders the full heterogeneous medium including muscle and organ interiors.
2. **No Siemens LUT applied** — we used an empirical bone TF. Once Simon Niedermayr provides the 1D LUT files (email 2024-03-19), these should be used instead.
3. **No gradient-magnitude blending** — Siemens uses `(1-GradMag)*VolPhase + GradMag*SurfPhase`. Our pure surface approach is a special case (GradMag≈1 everywhere at bone boundary) but misses the volumetric interior.
4. **No HDRI lighting** — we used area lights. Siemens uses studio HDRIs. Will need to replicate once we obtain their light environment.
5. **Pure surface rendering is not fully photorealistic GT** — it is a proof-of-concept that the pipeline (CT → mesh → Mitsuba → paired render) works. For training data generation, we need to align with Siemens' volumetric path-traced style.

## Next steps before full batch
1. Obtain the two Siemens LUT files from Simon → replace empirical TF
2. Confirm Siemens will contribute paired renders (or decide we generate all with Mitsuba)
3. Switch from surface-only to hybrid: bone surface + volumetric medium for soft tissue (Phase 2.1 TF calibration loop)
4. Add HDRI lighting matching Siemens' studio environment (Phase 3.2)
5. Run SSIM calibration: target >0.85 between PyVista structural silhouette and Mitsuba GT

## Conclusion
The end-to-end pipeline is functional locally:
> CT volume → preprocessed NIfTI → PyVista simple render → Mitsuba surface path-traced GT → geometrically aligned pair (IoU=0.980)

This confirms skull CT (CQ500) is a viable data modality. The pipeline can be extended to full volumetric rendering once TF calibration is complete. Batch generation across all 342 training volumes is feasible on the cluster.
