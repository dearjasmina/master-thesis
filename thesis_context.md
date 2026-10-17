# Thesis Context — Experimental Design and Evidence Map

*Written 2026-09-05. Load this when writing or revising any thesis chapter.*

This file records **why the experiments are shaped the way they are** and **where the
evidence lives**. It exists because the two-stage structure of the work is a deliberate
methodological choice that is not visible from the code alone — from the repo it just
looks like there are two renderers.

Supersedes `context.md` for thesis-writing purposes. `context.md` (dated 2026-05-17)
still describes the abandoned ControlNet/Mitsuba approach and should not be used as a
description of the current pipeline.

---

## 1. The two-stage experimental design

The work was deliberately run in **two stages, with a feasibility gate between them**.

### Why

Both halves of the pipeline are expensive and neither is cheap to redo:

- **Dataset generation takes days.** Cycles costs ~53 s per rendered view, ~1060 s per
  subject at 20 views; a full dataset pass is a multi-day, multi-GPU job.
- **Training takes long.** 4×L4, 1024², 150 epochs.

Fine-tuning the *appearance* of the ground truth before knowing whether the *end-to-end
approach works at all* would have risked spending that budget twice on a method that
might not converge. So the appearance model was deliberately frozen at a
"good-enough" first generation, the whole pipeline was run end to end, and only after
the approach was validated was time spent on making the renders look right.

This is the standard de-risking argument and it should be written up as a deliberate
design decision, **not** as a false start.

### Stage 1 — feasibility (renderer v20)

- **Renderer**: `scripts/render_pair_totalseg_v20_cinematic_hd.py`
- **Question**: does a G-buffer-conditioned pix2pixHD actually learn the Cycles
  forward-rendering function, at all, on real anatomy?
- **What was run**: full dataset generation, full training of **both** the G-buffer model
  and the RGB-only ablation, and the complete evaluation suite on both.
- **Outcome**: validated. Numbers in §3.

### Stage 2 — appearance (renderer v25)

- **Renderer**: `scripts/render_pair_totalseg_v25_textured.py`
- Once feasibility was established, time was spent on the ground-truth appearance, and
  the dataset was regenerated from scratch with the improved renderer.
- **Status as of 2026-09-05: training/evaluation still running. No Stage-2 numbers yet.**
  Jasmina will paste them when the run finishes. Do not invent or estimate them, and do
  not write "in progress" into the thesis — write the Stage-2 results section only once
  real numbers exist.

### What changed from v20 to v25

From the v25 docstring (it describes the cumulative v20→v25 pass):

1. **Camera framed from the organs, not the CT field of view.** v20–v23 derived camera
   distance from the scan extent, which varies enormously between subjects — identical
   anatomy rendered zoomed-out on one subject and cropped on another. v25 takes the union
   world bounding box of the imported meshes. Removes a large, purely nuisance source of
   appearance variance from the training targets.
2. **Cooler, less saturated palette.** Saturation 1.30 → 1.05; base colours darkened
   (liver in particular went from bright red to deep burgundy).
3. **Vessels pulled back on the gut.** Bowel/duodenum vessel strength 0.95 → 0.42,
   colon 0.70 → 0.34 — v23 values read as ink drawn on the surface rather than
   vasculature under serosa.
4. **Materials differentiated.** Everything previously rendered as the same wet plastic;
   the coat range was widened so tissues separate (bowel serosa wettest at 0.60 @ 0.030,
   through lung 0.20 @ 0.150, to bone 0.02 @ 0.480).
5. **Size-invariant exposure + stronger wraparound light.** Light energies now scale with
   `scene_scale²` so exposure no longer drifts between subjects of different size;
   environment 0.45 → 0.85 and fill 3 W → 15 W.

6. **Measurement-grounded texture maps** (see §2a — this is the substantive v25
   contribution, not just a tuning pass).

Separately, a **mesh-level fix** landed in this period: the "wavy corduroy" artefact in
v20–v22 was marching-cubes staircase in the geometry, not a shading bug (proved by
rendering the same mesh with a default Principled material).

**Corrected 2026-09-06 — the shipped fix is NOT a Gaussian blur.** `extract_meshes.py`
fixes it with a **`float32` cast** of the thresholded mask (was `uint8`) plus stronger
smoothing (20 iterations at relaxation 0.3; thin structures 12 at 0.15, citing Bade et
al.). With `uint8`, `cell_data_to_point_data()` truncates every 2×2×2 average back to
0/1 — 2 distinct values — so `contour()` still runs on a binary field. `float32` keeps
9 distinct values and the isosurface comes out smooth. Measured on a synthetic
ellipsoid (`extract_meshes_smooth_patch.py`):

| variant | levels | \|meanCurv\| p90 | volume |
|---|---|---|---|
| uint8 + smooth(30, 0.05) *(old)* | 2 | 0.153 | 101.6 % |
| float32 + smooth(30, 0.05) | 9 | 0.079 | ~99 % |
| **float32 + smooth(20, 0.3)** *(shipped)* | 9 | **0.065** | 97.3 % |
| float32 + gaussian(1.0) + smooth(20, 0.3) | 33055 | 0.068 | 97.2 % |

The Gaussian variant was tested and is **not better** than float32 alone, which is why
it was not adopted. The v25 docstring still describes the blur as the intended
permanent fix — that docstring is stale; trust `extract_meshes.py`.

---

## 2a. Texture grounding — where v25's appearance actually comes from

**This is the part of v25 worth writing up as a contribution.** Frame it as *what v25
adds*, not as *what v20 got wrong*.

### The problem it solves

TotalSegmentator gives **organ boundaries and nothing inside them**. A segmentation mask
has no subcapsular vasculature, no serosal fat streaks, no anthracotic speckle in lung —
and those are exactly the cues that make a rendered surface read as tissue rather than
painted latex. Procedural shading in the render script hit a ceiling because every organ
evaluated the same node graph: organs differed in *colour* but not in *kind*.

### The pipeline

```
HiP-CT volumes (Human Organ Atlas)
   └─ scripts/hoa_vessel_stats.py     measures vessel calibre statistics
        └─ data/hoa_stats/*.json
             └─ scripts/synth_tissue_textures.py   synthesises tileable PBR maps
                  └─ data/renders/textures/tissue/{tissue}_albedo|_normal|_rough.png
                       └─ v25 renderer, BOX (triplanar) projection
```

Triplanar projection is required because marching-cubes meshes **have no UV layout** —
the texture is projected down all three object-space axes and blended at seams
(`projection_blend = 0.35`). Missing textures fall back to the procedural chain, so it
degrades gracefully. 18 texture sets on disk cover the 29 tissues, via a class-name
fallback (longest `REFERENCE_LOOK` key contained in the tissue name), so one `lung` set
serves all four lobes.

⚠️ **`--tex_mm` default differs between the two entry points**: the v25 pair renderer
and `synth_tissue_textures.py` both document **120 mm**, but
`generate_training_dataset.py` passes **80 mm**. The synthesiser's docstring warns the
two MUST match, so the dataset renders the maps at a 1.5× tighter tile than the
previews they were tuned against. Worth confirming this was intentional before
describing tile size anywhere in the thesis.

### Source data — Human Organ Atlas (HiP-CT)

Downloaded by hand from <https://human-organ-atlas.esrf.fr> (portal ZIP links embed an
expiring session token, so this is **not** scriptable). Currently in `~/Downloads/`:

| Organ | Donor | Voxel size | Size |
|---|---|---|---|
| liver | LADAF-2020-27 | 101.0 µm | 630 MB |
| kidney-right | LADAF-2020-27 | 100.0 µm | 61 MB |
| lung-left | LADAF-2020-27 | 84.68 µm | 1.4 GB |
| colon | LADAF-2021-17 | 80.1 µm | 1.6 GB |
| heart | LADAF-2020-31 | 79.56 µm | 1.1 GB |

**Licensing: CC-BY-4.0. Each dataset has its own DOI that MUST be cited separately**,
plus the HOA paper (DOI `10.1126/sciadv.adz2240`, Sci. Adv. 12, eadz2240, 2026 — verify
author list before submission). Collect the five dataset DOIs from the portal.

Nice connection worth making in Chapter 2: **Niedermayr et al. \cite{niedermayr2024gaussians}
also use HiP-CT data** — they path-trace it directly at 95 s/frame. This thesis instead
*measures statistics from it* to ground synthesis. Same source, opposite use.

### What was measured, and how

`hoa_vessel_stats.py`: Hessian multiscale vesselness (Sato or Frangi) **with scale
selection** — the sigma maximising ridge response gives calibre directly. σ → radius
factor of **1.336, calibrated on synthetic tubes of known radius**, not taken from a
paper. Restricted to a **subcapsular shell** (`--depth_mm`), because that is the network
the renderer actually draws. Has a `--selftest` that builds a tree with known parameters
and checks recovery before spending a download.

Measured subcapsular vessel radii (mm):

| Organ | Donor | p50 | p95 | Filter | Detected frac |
|---|---|---|---|---|---|
| liver | LADAF-2020-27 | 0.55 | 2.4 | sato | 0.0175 |
| kidney | LADAF-2020-27 | 0.9 | 1.5 | sato | 0.0008 |
| lung | LADAF-2020-27 | 0.2 | 2.4 | sato | 0.0013 |
| heart | LADAF-2020-31 | 0.9 | 2.4 | frangi | 0.0042 |
| colon | LADAF-2021-17 | 0.9 | 1.5 | frangi | 0.0011 |

Heart and colon need **Frangi, not Sato**: their walls and trabeculae are *sheets*, and
Sato responds to sheets — it flagged 2.7% of heart volume as "vessel"; Frangi drops that
to 0.42%. That is a real methodological finding, worth a sentence.

Trunk radius is set as `start_w = p50 / 0.56`, because the tree tapers by **Murray's law**
(0.794) over ~5 levels so the typical drawn vessel is `start_w × 0.794^2.5`.

**The headline result of the measurement**: hand-picked values had been 3.4 mm (liver) and
4.5 mm (heart). Measured p50 is 0.55 and 0.9. That is a **4–6× overestimate** — the
quantitative form of "the veins look too prominent." Write it that way; it turns a
subjective art note into a measured correction.

### Colour does NOT come from HiP-CT

Phase-contrast tomography is **greyscale** — it can ground *structure* only. Organ colour
comes from the **Visible Korean** sectioned-anatomy images (Park et al., 2015,
*Int J Morphol* 33(4):1323–1332). Example: subcutaneous fat sampled at sRGB (205,152,80)
and (194,141,101) → linear ≈ (0.61, 0.32, 0.08) — green sits near *half* of red; adipose
is a strong yellow-orange, not a cream.

### Honest limitations — state these, do not hide them

- **Split probability and segment length are NOT measured.** The skeleton gives 7604
  branch points against 3620 endpoints, but a binary tree must have roughly one *more*
  endpoint than branch point, so the extracted topology is unusable. These stay hand-set.
- **bowel, stomach, pancreas and great-vessel walls are extrapolated**, not measured —
  no HOA volume was downloaded for them.
- **p50 is threshold-sensitive.** Loosening detection raises kidney 0.9 → 1.5 and lung
  0.2 → 0.55, and pushes p95 against the largest offered scale (a sign of over-growth).
  The table uses the conservative setting. **What is robust across every method and
  threshold tried is the order of magnitude: subcapsular vessels are 0.2–1.5 mm radius.**
- **Resolution mismatch.** HOA whole-organ scans are 19–25 µm; the render samples at
  ~0.65 mm/px, ~30× coarser. So the useful product is *statistics*, not the voxel data.
  The 101 µm tier is the right one to work with.
- Textures are **synthetic in placement, measured in statistics**. Say exactly this —
  do not claim the textures are extracted from real scans, because they are not.

### Determinism is preserved

Everything is seeded per organ class and is a deterministic function of object-space
position, so renders stay reproducible per subject — which the training pipeline requires
and which the whole cross-view consistency argument depends on.

**Thesis framing**: v20 is the *first-generation appearance model*, not a "worse" or
"broken" one. The v20→v25 delta is a contribution in its own right — it is a systematic,
measured account of what makes synthetic anatomy read as photorealistic.

---

## 2. The RGB-only vs G-buffer ablation

Run at **both** stages, and it is the direct answer to the "does the G-buffer earn its
keep?" research question.

- **G-buffer model** (`full1024` preset): 8-channel input —
  `seg_rgb` (3) + `depth` (1) + `normals` (3) + `segid` (1).
- **RGB-only baseline** (`rgb_1024` preset): 3-channel input — `seg_rgb` only.

`rgb_1024` derives from `full1024`, **not** from the 512 px `rgb_only` preset. This is
deliberate and is documented at length in `scripts/training/config.py`: deriving from
`full1024` changes exactly one variable (the input buffers), so the comparison isolates
the contribution of the geometry buffers instead of confounding it with a resolution
change.

Because the ablation was run at both stages, the Stage-2 result is a **replication** of
the Stage-1 finding under a different target appearance. That is worth stating explicitly
in the thesis — it is a stronger claim than a single ablation.

---

## 3. Stage 1 results (renderer v20) — real numbers

Test split: **96 held-out subjects × 20 views = 1920 images**. Paired over identical
`(subject, view)` keys. Hardware: NVIDIA L4.

### Image quality

| Metric | G-buffer | RGB-only | Paired Δ (95% CI) | Wilcoxon |
|---|---|---|---|---|
| PSNR ↑  | **33.357** | 30.992 | −2.365 dB [−2.417, −2.317] | p ≈ 0, significant |
| SSIM ↑  | **0.9587** | 0.9435 | −0.0152 [−0.0155, −0.0149] | p ≈ 0, significant |
| LPIPS ↓ | **0.0467** | 0.0628 | +0.0161 [+0.0158, +0.0165] | p ≈ 0, significant |
| FID ↓   | **8.21**   | 14.80  | — | — |

Wilcoxon signed-rank (no normality assumption); bootstrap CI on the paired difference.
**The G-buffer wins on every metric, significantly.**

FID caveat: computed by an in-repo torchvision Inception-v3 fallback and explicitly
flagged in `fid_kid.json` as *not* numerically comparable to published
torch-fidelity/TF numbers. Only use it to compare these two models against each other,
and say so in the thesis.

### Speed

| | ms/frame | FPS | Peak mem | Speed-up vs Cycles |
|---|---|---|---|---|
| G-buffer | 79.66 | 12.55 | 1607 MiB | **665.7×** |
| RGB-only | 76.39 | 13.09 | 1559 MiB | 694.2× |

- All at 1024², best over batch sizes 1/2/4/8 (throughput is essentially flat in batch
  size — it is compute-bound, not launch-bound).
- **Cycles reference: 53.025 s/frame**, parsed from `logs/gpu0.log` — median 1060 s per
  subject over 294 subjects ÷ 20 views. Measured, not guessed (`eval_speed.py` refuses to
  guess it).
- The G-buffer costs ~3.3 ms/frame more than RGB-only — a ~4% speed penalty for a
  2.4 dB PSNR gain. That trade is worth stating explicitly.

### Cross-view consistency

> **Recompute before reporting (fixed 2026-09-17); and the metric is LOW-POWER.**
> `eval_multiview.py` reprojected with `meta.json`'s K, omitting the lens shift and the
> emission-pass depth gain (§4b); it now uses `camera_model`. Measured on identical view
> pairs, the correction more than doubles the correspondences that survive the occlusion
> test (11 → 26 pairs) but moves the GT floor only 15.75 → 16.19 dB. So — correcting an
> earlier claim of mine — **the camera bug is NOT why the floor is low.** The floor is low
> because the same surface genuinely looks different from two angles (specular highlights
> move, plus SSS and depth of field), because the warp is nearest-neighbour forward, and
> because co-visibility is only ~4%. Consequences for the thesis: report only the GAP
> (model − GT floor), never the absolute PSNR; state the co-visible fraction and n; and
> treat a near-zero gap as "no measurable added inconsistency", not as proof of
> consistency. The v20 numbers below cannot be recomputed (v20 data deleted).

| | reproj PSNR (model) | reproj PSNR (GT floor) | Gap |
|---|---|---|---|
| G-buffer | 11.691 dB | 11.519 dB | **−0.172 dB** |
| RGB-only | 11.593 dB | 11.519 dB | −0.074 dB |

Both gaps are **negative**, i.e. the models are at or fractionally below the ground-truth
reprojection floor. The interpretation: the network adds **no measurable cross-view
inconsistency** beyond what the reprojection test itself incurs (interpolation, depth
quantisation, genuinely view-dependent shading). This is the empirical confirmation of
the determinism argument.

**Caveat that must be reported honestly**: n = 43 view pairs and mean co-visible fraction
is only **4.87%**. The camera manifold is 20 views over a full 360° azimuth at 4
elevations, so most view pairs barely overlap. The absolute reprojection PSNR (~11.5 dB)
is low for exactly this reason — it is a *floor-relative* measurement, and only the gap is
meaningful. Do not report the absolute number without the floor next to it.

### Per-tissue

Per-organ PSNR spans roughly **19.6 dB to 27.8 dB**. Error tracks organ size: the worst
organs are the ones occupying the smallest pixel fractions (e.g. `id_28` at 0.07% of
pixels → 19.6 dB), the best are the large ones (`id_4` at 1.7% → 27.8 dB). This is the
point of the per-tissue breakdown — a whole-image metric is dominated by whichever organ
has the most pixels and hides failure on every small structure.

Note the per-tissue CSVs use raw `pass_index` ids (`id_4`, `id_28`, …). Map them to organ
names via `tissue_ids.json` before putting them in a table.

### Training curves

`metrics_history` CSVs track train *and* val at epochs 0, 5, …, 60. The train-vs-val gap
is the overfitting signal, and the volume-level split means val/test are genuinely
held-out subjects, not held-out views of training subjects.

---

## 4. Evidence map

### In-repo — `EVALUATION/` (Stage 1)
| Path | Contents |
|---|---|
| `EVALUATION/test/metrics_test_gbuffer.csv` | 1920 per-view rows: subject, view, psnr, ssim, lpips |
| `EVALUATION/test/metrics_test_rgb.csv` | same, RGB-only baseline |
| `EVALUATION/test/comparison_test.png` | rendered comparison figure |
| `EVALUATION/eval/metrics_history–gbuffer.csv` | per-epoch train/val curves (0–60, step 5) |
| `EVALUATION/eval/metrics_history_rgb.csv` | same, RGB-only |
| `EVALUATION/eval/comparison_eval.png` | rendered curve figure |
| `EVALUATION/plot_test_comparison.py`, `plot_eval_comparison.py` | the plotting scripts |

### Outside the repo — `~/Downloads/old evaluation scripts/` (Stage 1, fuller)

> ⚠️ **This folder is in `Downloads`, outside the repo and not version-controlled.**
> It holds the only copy of the per-tissue, multiview, FID/KID and montage results.
> Move it into the repo before it gets lost.
>
> ⚠️ The **HiP-CT source volumes are also in `~/Downloads/`** (~4.8 GB across five
> organs, see §2a). They are re-downloadable from the HOA portal but only by hand.
> At minimum, keep `data/hoa_stats/*.json` — the measured statistics — in the repo,
> since that is what `synth_tissue_textures.py` actually consumes.

| Path | Contents |
|---|---|
| `v20_comparison/comparison.md`, `report.md` | the paired G-buffer vs RGB stats (§3) |
| `v20_gbuffer/`, `v20_rgb/` | full per-model eval: `report.md`, `metrics_test.csv`, `per_tissue_summary.csv`, `per_tissue_views.csv`, `multiview_summary.json`, `multiview_pairs.csv`, `fid_kid.json`, `speed.json`, `montage.png`, `grids/`, `worst/` |
| `gbuffer_speed.json`, `rgb_speed.json` | the speed/speed-up numbers |
| `montages/*.png` | per-subject qualitative montages (~30+ subjects) |
| `test_final/`, `rgb_infer_test/`, `eval/` | additional inference/eval outputs |

### Stage 2 — pending
Not yet produced. Will be generated by `scripts/evaluation/run_full_eval.sh` against the
v25-regenerated dataset.

---

## 4b. Texture is not learned — the position ablation (NEGATIVE RESULT, 2026-09-13)

**Symptom.** Inferred images match the GT closely in silhouette, shading, folds and
specular response. What is missing is the per-organ *surface texture* — vessels and
mottling are smoothed away. Visible in the `seg | fake | GT` triptychs: everything
derivable from geometry is reproduced; only the position-locked pattern is absent.

**Hypothesis tested.** v25 samples every organ texture through `ShaderNodeTexCoord`'s
*Object* output in BOX projection, so in the GT the pattern at a pixel is a deterministic
function of the object-space 3D point. The 8-channel input gives identity (`seg_rgb`,
`segid`), orientation (`normals`) and distance (`depth`) — but recovering the 3D point
additionally needs the camera pose, which was never an input. The network was therefore
asked to predict a function of a variable it cannot observe, whose best answer is the
average over all consistent positions: a smooth surface.

**What was built.** A `position` buffer (`dataset.py::unproject_position`), unprojected
from `depth.exr` using `meta.json`'s `K` and `cam_to_world`; object-space mm = world m ×
1000. 3 raw channels + 18 Fourier channels (sin/cos at 256/64/16 mm) = 21 ch, total input
29 ch. Preset `gbuf_pos1024`, identical to `full1024` in every other respect (same data,
losses, architecture, 61 epochs → `epoch_060.pt`). No dataset regeneration was needed.

**The buffer was WRONG — the pre-training verification was insufficient.** Those checks
compared coordinate *ranges* and a normal-vs-tangent angle whose 14° residual was
rationalised as noise; neither can detect an offset. The gold-standard check — distance
from every reconstructed organ pixel to the subject's own meshes
(`scripts/analysis/texture_learnability.py geometry`) — puts the trained buffer **7–30 mm
off the surface**. Two causes, neither recorded in `meta.json`:
1. **Lens shift.** v25 `setup_camera()` sets `shift_x=-0.05`, `shift_y=+0.02`, but
   `meta.json`'s K puts the principal point at (512, 512). The true point is
   (W/2 − sx·W, H/2 + sy·W) = **(563.2, 532.48)**. Omitting it displaces points ~25–30 mm
   laterally — fixed in the camera frame, hence a *different* world offset in every view.
   ICP offsets in camera coordinates matched the prediction view by view (27.2 vs 27.7 mm).
2. **Emission-pass gain.** depth/normals/segid are EEVEE emission renders, and colour
   management multiplies them on write: **R, G × 0.99213, B × 0.99213²** (same ratio in
   depth.exr and segid.exr). Depth reads 0.8% short: 11 mm at 1.35 m, 18 mm at 2.2 m. Fitting
   subjects at both distances separates a scale from an offset. Independent confirmation:
   gain-corrected normals decode to unit length (|N| median 0.9997) where raw ones do not
   (0.983–0.992).

With both corrections the reconstruction lies **0.63–0.90 mm** from the mesh (vertex-spacing
floor) on 8 views of s0895/s0050, against 7.5–25.5 mm uncorrected. Planar `View Z Depth` is
confirmed; ray distance fits worse.

**Consequence: the `gbuf_pos1024` run did not test the hypothesis.** Across 112 view pairs
(5.5 M true correspondences), the same surface point received coordinates that disagreed
by a median **30.5 mm** (p90 55 mm) — about 40% of the 80 mm texture tile; the 16 mm band
correlated r = 0.05 between views of one point. Corrected: **0.65 mm** median, 16 mm band
r = 0.95, detail registers (r ≥ 0.5) down to ~4 mm. No network could learn a texture from a
coordinate that inconsistent.

**Result.** Surfaces look *more* textured than `full1024`, but wrong in content: a
**regular periodic grid** on the organs instead of vessels. (TODO: val LPIPS at epoch 60
from `evaluate.py --track` against `full1024`'s 0.02174.)

**Why the grid — two separate artefacts.**
- **Screen-locked checkerboard — architectural, and present in `full1024` too.** Mean
  power spectra of predictions show a lattice of sharp peaks at a **32 px period and its
  harmonics** (16, 8, 4 px) — the generator's total upsampling stride, 2⁵, from stacked
  `ConvTranspose2d(k=3, s=2)` layers — in BOTH models and not in the GT. Share of
  high-frequency power at the lattice: GT 1.5%, `full1024` 4.9%, `gbuf_pos1024` 4.5%. Known
  fix: resize-then-convolve upsampling (Odena et al., 2016, "Deconvolution and
  Checkerboard Artifacts"). This is a limitation of the main model as well.
- **Not encoding leakage — tested and refuted.** `texture_learnability.py phase` on
  `gbuf_pos1024`: flipping the sign of the finest (16 mm) band changes the output by
  2.67/255 on average, while shifting the raw coordinate 256 mm with every band left
  identical changes it by 4.64/255 — ratio 0.6. The output is not driven by the band
  phase, so the grid is the checkerboard above, not the Fourier channels.

**Evaluation caveat — state this.** A perceptual metric can *reward* such artefacts: LPIPS
responds to the presence of fine detail, not whether it is the right detail. Never report
an LPIPS gain for `gbuf_pos1024` as "learned the vessels" without the images beside it.

**Where the vessels actually are.** The albedo tiles contain only soft mottling; the vessel
network lives **only in the normal map**, and appears in the GT as specular glints along
vessel ridges. The lights are world-fixed while the camera orbits, so glints are view-
dependent. The tangent-space normal map still perturbs shading on the UV-less meshes (the
glints are visible), but the exact tangent frame Blender uses there is unverified.

**The texture files are not reproducible.** `synth_tissue_textures.py` seeds with
`default_rng(seed + abs(hash(name)) % 100000)`, and Python salts `str` hashes per process,
so each run produces different vessel layouts despite the "deterministic" docstring. The
tiles are not in git. The copy on the server is the only record of what the dataset was
rendered with — archive it (with md5 sums) as part of the dataset.

**Other code using the wrong K:** `eval_multiview.py` reprojection, which explains its
implausible 11.5–15.4 dB GT floor. **The multi-view consistency numbers in every report are
invalid** until re-run with the corrected camera model. `evaluate.py`'s
`cross_view_reprojection_gate` likewise.

**Secondary factors:** loss weighting (L1 10 vs GAN 1, never tested); half-float depth
(0.5–2 mm steps — with the correct model still 0.65 mm median registration, so a limit on
sub-millimetre detail but not on vessel-network placement); depth of field ~1–2 px CoC at
f/11; `ngf=32`.

**Feasibility without regeneration — server results (2026-09-14),
`scripts/analysis/texture_learnability.py`:**
- *Textures:* server tiles last modified 2026-09-01 20:42, first dataset render 21:02, so
  they are the tiles the dataset was rendered with. `MD5SUMS.txt` archived beside them.
- *Emission gain:* G/R 1.00000 and B/R 0.9916–0.9927 over 400 random views — constant.
  All depth is HALF float; step 0.24–1.95 mm (median 0.98).
- *Registration* (s0000, 378 view pairs, 18.8 M points, 0.49 mm depth step): corrected
  coordinates agree to **0.28 mm** median (p90 1.1 mm); detail registers down to 2 mm
  (r = 0.72), 1 mm r = 0.48. Legacy: 18.2 mm median.
- *Oracle* (s0000–s0002, 60 views, server tiles): the normal-map vessel ridges, projected
  with Cycles' exact BOX conventions at the corrected position, are **locked to the GT's
  fine detail** — in every large single-texture region (n = 26k–272k) the blind search
  picks exactly Cycles' variant out of 8, by a 10–20× margin; half-tile null r = 0.001;
  r decays with added position noise (0.057 → 0.025 at 0.5 mm → 0.001 at 4 mm). But the
  linear predictivity of this proxy is **weak**: pooled r = 0.057 (kidney 0.14, liver
  0.11, colon 0.10, bladder 0.09), r = 0.053 against the `full1024` residual, and
  `full1024`'s own output carries none of it (r = −0.001).
- *Interpretation:* the vessel information is present in, and registrable from, the
  existing dataset — the necessary condition holds. Sufficiency is NOT shown: the vessels
  appear only as glints whose visibility depends on the light, the view, and the tangent
  frame Blender derives for the normal map on UV-less meshes, which is not recorded. The
  half-noise point at 0.5 mm also puts the signal right at the positional precision the
  half-float depth allows.

**Pre-flight (2026-09-14) — NO-GO; write as a limitation.** `scripts/analysis/preflight.py`.
Two runs on s0098 + s0746 (train-split subjects with the strongest registered texture
signal, 0.076 / 0.055), 4 views per subject held out, 150 epochs, resize upsampling.
`preflight_tex` received the exact server tiles looked up at the corrected position
(15 ch); `preflight_ctrl` did not (8 ch). Every pre-training check passed: tiles
md5-identical to the rendering tiles; reconstruction 0.65 mm from the mesh; dataloader and
inference inputs byte-identical; texture signal +0.050 vs half-tile null −0.001.
- D0 pass — held-out GT shows registered vessel streaks: M3 0.066 (liver 0.150).
- D1 pass — the texture model fits fine detail on its training views: M1 0.507
  (`full1024`, trained on everything: 0.455).
- **D2 fail, decisively.** The texture model's streaks are not where the vessels are:
  held-out M3 **0.003** vs GT 0.066 (liver 0.003 vs 0.150) — and not even on its own
  training views (0.008 vs GT 0.051; the control, which never saw a texture, 0.012).
  Visually, training-view samples show soft highlights and no branching vessels. So the
  failure is not the data, the camera model, or missing information: **given the exact
  texture, a pix2pixHD generator under this objective (L1 10 / VGG 1 / GAN 1) does not
  render thin, sparse, view-dependent specular glints.**
- D3 "pass" is **not** evidence of texture use: M2 (partial high-frequency correlation)
  proved not to be texture-specific (liver 0.02) — it reflects two independent runs
  capturing different geometric detail. M3 is the texture-specific test.
- D4 lattice: resize upsampling cut the excess over GT from +3.4 pp (transpose models) to
  +0.8 pp (tex) / +1.2 pp (ctrl), but not to GT level. The pre-registered absolute 2.5%
  threshold was mis-calibrated — GT itself is 2.38% on these views — and the per-epoch
  sweep is noisy (8 views).
- D5: organ PSNR unchanged (21.06 vs 21.12 dB).
- Untested lever: the loss weighting. A less L1-dominated objective might synthesise
  vein-like glints, but whether they would be *registered* (M3) is unknown; likely
  plausible-but-invented detail.

**The glints are computable (2026-09-14) — `texture_learnability.py glint`,
`scripts/training/glint_model.py`.** Diagnosis of the pre-flight failure: the vessels are
specular glints whose position depends on the normal-map perturbation *and* the tangent
frame, camera and lights — none of which the texture buffer contained. All are recoverable:
- *Lights:* v25's 5 world-fixed area lights rebuilt from meta `organ_center`/`scene_scale`.
  Validation without any texture: replicated specular highlights match the GT's highlight
  structure at r = 0.32 (up to 0.50) over 8 views; a mirrored rig gives 0.04.
- *Material:* v25 TISSUES + REFERENCE_LOOK loaded from source without Blender.
- *Tangent frame (UV-less meshes):* 24 hypotheses + a tangent-free bound tested on the
  server (s0098, s0746, s0194; 60 views), ranked by texture-locked excess over a half-tile
  null. Winner **sphere_origin, signs (+,−), coat unperturbed: partial r 0.149 beyond
  geometric highlights, null 0.030, excess 0.119** — 2.8× the raw-ridge proxy. The winner
  is exactly Cycles' construction, confirmed on two independent points: MikkTSpace on
  `map_to_sphere` of *raw object coordinates* (Cycles' generated-coordinate encoding cancels
  out, so the sphere is about the object origin, not the texture-space box), and the (+,−)
  sign is exactly the flip between the analysis `v = −asin/π` and Cycles'
  `v = 1 − acos/π`. The canonical implementation reproduces it to 1.5e-6°.
- Also found: `generate_training_dataset.py` applies its key-light jitter only to a light
  named "KeyLight", which v25 never creates — `subj_key_energy_factor` was recorded but
  never applied.
**Pre-flight #2 (2026-09-15) — `preflight_glint` (texture + glint, 17 ch) vs the unchanged
`preflight_ctrl`; same subjects, held-out views, thresholds. D2 FAIL → NO-GO for veins,
but a different failure from pre-flight #1: the network now USES the glint input.**

| held-out views | GT | preflight_tex (#1) | preflight_glint (#2) | ctrl |
|---|---|---|---|---|
| M3 vessels, all organs | 0.066 | 0.003 | **0.016** | 0.001 |
| M3 vessels, liver | 0.150 | 0.003 | **0.047** | −0.001 |
| M3 vessels, kidneys | 0.107 / 0.068 | — | −0.002 / −0.012 | — |
| M1 fine detail, all / liver | — | 0.260 / 0.020 | **0.442 / 0.397** | 0.248 / 0.002 |
| organ PSNR | — | 21.06 dB | **23.04 dB** | 21.12 dB |
| lattice % (GT 2.38) | — | 3.20 | 2.85 | 3.55 |

- Vessel registration went from nothing to ~30% of the GT ceiling on liver (25% overall),
  none on the kidneys, and **plateaued** (M3 0.014 at epoch 50 → 0.016 at 149): below the
  pre-registered 50%. Visually (held-out panels): the large specular highlights are now in
  the right place and shape (the control smears them into horizontal streaks), with some
  fine sparkle inside them — but the thin vessel network across the dark liver is still
  absent.
- Caveat on the PSNR/M1 gain: both pre-flight models saw only 32 training views, where
  computed highlights help enormously. `full1024`, trained on the full dataset, already
  learns highlight placement from normals and depth, so most of that gain would likely
  shrink at full scale. The vein term is the part geometry cannot supply, and it is weak.
- D4: 2.85% vs GT 2.38% (+0.5 pp; ctrl +1.2 pp) — the absolute 2.5% bar remains
  mis-calibrated, as noted for pre-flight #1.
- Interpretation for the thesis: the complete chain is now measured — veins are in the
  data; the camera model was corrected; the texture was supplied; the glint mechanism was
  identified and computed exactly as Cycles does it; the network then uses it, but under
  this objective (L1 10) it reproduces the broad specular response and suppresses the thin,
  registration-sensitive vessel lines. Untested remaining lever: loss weighting with the
  glint input.

**Pre-flight #3 (2026-09-14) — `preflight_glint` with the pix2pixHD objective (no L1:
L1 0, VGG 10, FM 10, GAN 1); everything else identical. D2 narrowly FAILS (42% of the GT
ceiling vs the 50% bar), but removing L1 raised vessel registration further.**

| held-out views | GT | #1 texture | #2 glint, L1 10 | #3 glint, no L1 | ctrl |
|---|---|---|---|---|---|
| M3 vessels, all organs | 0.066 | 0.003 | 0.016 | **0.028** (42%) | 0.001 |
| M3 vessels, liver | 0.150 | 0.003 | 0.047 | **0.079** (53%) | −0.001 |
| M3 vessels, kidney L / R | 0.107 / 0.068 | −0.015 / −0.029 | −0.002 / −0.012 | **0.032** / −0.010 | — |
| M1 fine detail, liver | — | 0.020 | 0.397 | **0.449** | 0.002 |
| organ PSNR | — | 21.06 | 23.04 | 22.64 | 21.12 |
| lattice % (GT 2.38) | — | 3.20 | 2.85 | 3.09 | 3.55 |

- L1 was part of the suppression: each change moved vessel registration monotonically
  (0.003 → 0.016 → 0.028). On the liver it now clears half the GT ceiling; overall it is
  held back by the kidneys and smaller organs.
- **Plateau:** held-out M3 was 0.025 at epoch 25 and 0.028 at 149. Training fit keeps
  rising (train M1 0.487 → 0.608), so further epochs on 32 views mostly overfit. The
  limiting factor in the pre-flight is data diversity, not training length — which is the
  one thing a full-dataset retrain changes and a pre-flight cannot test.
- PSNR fell 0.4 dB against the L1 run (the expected perception–distortion trade), still
  +1.5 dB over the control at this data scale.
- Visually (held-out panels): highlights placed and shaped correctly; faint vessel
  sparkle in the right regions (e.g. the lower-right liver cluster); the fine lines are
  much fainter than in the GT. The control shows streak artefacts throughout.
- Best recipe found: resize upsampling + `texture` + `glint` + pix2pixHD objective.

**Artifacts:** run `results/training_runs/gbuf_pos1024` (`epoch_060.pt`), inference
`results/infer_gbuf_pos`, evaluation `EVALUATION/v25_gbuf_pos`.

---

## 5. How to write this up

- **Chapter 3 (Methodology)** gets the two-stage structure as a deliberate design: state
  the cost argument (days per dataset, long training) as the reason for the feasibility
  gate. Describe both renderer generations, with the v20→v25 changes as a table. Give the
  texture-grounding pipeline (§2a) its own subsection — it is the substantive v25
  contribution and it answers the obvious objection that the appearance is invented.
  Frame v25 by what it *adds* (measured appearance grounding), not by what v20 lacked.
- **Chapter 4 (Experiments)** describes the ablation design and why `rgb_1024` derives
  from `full1024` rather than `rgb_only`.
- **Chapter 5 (Results)** reports Stage 1 and Stage 2 side by side, with the ablation
  replicated at both. Always pair the multiview number with its GT floor. Always flag the
  FID as in-repo-only.
- **Chapter 6 (Limitations / Future work)** gets §4b. Be precise about what the
  `gbuf_pos1024` run shows: because of the undocumented lens shift and emission gain, its
  coordinate buffer was inconsistent across views by ~3 cm, so it is **not** evidence
  against position conditioning. Report the checkerboard lattice as an architectural
  limitation shared by `full1024`. The camera-model findings (shift, gain, half-float
  depth, unreproducible textures) are genuine dataset-documentation contributions.
- **Do not** describe Stage 1 as preliminary, failed, or throwaway. It is a completed
  experiment on a first-generation appearance model, and it carries the feasibility
  result plus a fully replicated ablation.
- The training target is `rgb_preview.png` — the **AgX tone-mapped** Cycles output, not
  raw scene-linear radiance (`config.py`: `target="preview_png"`). Say so precisely in
  Methodology.
