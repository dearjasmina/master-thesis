# 3-D view of the neural renders: feasibility study and execution plan

Question: can the demo end with an interactive 3-D model (drag, rotate, zoom in the browser)
built from the neural renderer's 20 output views?

Everything below was measured on the reference run `demo/runs/20260922-165227_s0006`
(s0006, test split). Prototype code and images are in `demo/research_3d/`, numbers in
`demo/research_3d/out/*.json`. All pixel/3-D geometry goes through
`scripts/training/camera_model.py`.

---

## 0. Verdict

**Yes, and it's cheap: no 3DGS is needed.** The recommended approach uses the organ meshes
we already have (they align with the renders to the pixel) and projects the 20 neural renders
onto them in the browser, blended by viewing angle ("view-dependent projective texturing",
i.e. the unstructured lumigraph, Buehler et al. 2001). A per-vertex "baked" colour is kept as
a fallback.

- **It needs no new geometry and no training.** The backend stage takes **2.2 s and 0.24 GB**
  and writes **0.5 MB** of new files. The viewer reuses the web meshes (6.1 MB) and the output
  PNGs (5.5 MB) that the page already loads. It is vanilla WebGL2 and works offline.
- **It is exact at the 20 cameras.** Rendered from a captured camera it reproduces that
  neural render at **42 dB** on the organs with the decimated web meshes (**55-60 dB** with the
  full-resolution meshes).
- **Between the cameras it interpolates, with an honest error.** Leave-one-out scores are
  **27.4 dB full frame / 20.6 dB on the organs**. The same procedure applied to the Cycles
  ground truth scores 27.6 / 20.7. So the loss comes from genuinely view-dependent shading
  (wet highlights, vessel glints) and 72° camera gaps, not from the network. The network's 20
  renders are as mutually consistent as Cycles' renders (gap **0.15-0.17 dB**).
- **3DGS is not worth it for the demo.** It would re-estimate geometry we already know
  exactly. It is lossy at the training views, where ULR is exact. It cannot beat the
  view-dependence floor with 16-20 views. It also needs a CUDA server or an uninstalled Mac
  tool, plus a vendored viewer. If the thesis wants the comparison anyway, §5 has the exact
  server commands and the comparator number it must beat (**26.2 dB full / 18.7 dB organs**
  on gsplat's own split).

---

## 1. Measured evidence

### 1.1 Alignment (Q1): everything downstream depends on this
`research_3d/align.py`: the full-resolution run meshes (`meshes/s0006/*.obj`, 21 organs,
276 k verts, 551 k tris, median edge 1.51 mm) were z-buffered through `camera_model.project`
and compared with Blender's `segid.exr` / `depth.exr`. The results are in `out/align.json`.

| view | fg IoU | organ IoU (area-wtd / min) | label agree | silhouette chamfer | depth abs diff median / p90 | unproject to nearest vertex median |
|---|---|---|---|---|---|---|
| v01 az0 el-50 | 1.0000 | 1.0000 / 0.9988 | 100.00 % | 0.00 px | 0.93 / 1.44 mm | 0.82 mm |
| v02 az0 el0 | 1.0000 | 0.9999 / 0.9993 | 100.00 % | 0.00 px | 0.94 / 1.50 mm | 0.90 mm |
| v07 az72 el30 | 1.0000 | 0.9996 / 0.9897 | 99.98 % | 0.00 px | 0.89 / 1.43 mm | 0.88 mm |
| v16 az216 el60 | 1.0000 | 0.9996 / 0.9891 | 99.98 % | 0.00 px | 0.85 / 1.36 mm | 0.75 mm |

- The alignment is **pixel-exact**. `out/align_v02_boundaries_crop.png` shows the rasterised
  organ boundaries (magenta) completely covering Blender's (green) on top of the neural render.
- The depth has a **constant signed bias**: the mesh raster is 0.85-0.94 mm *deeper* than
  depth.exr (relative 6.4e-4, std 0.41-0.46 mm). That is about one half-float step: the
  EXRs are stored as 32-bit float but carry half precision. v02's organs show only 239
  distinct depth values, 0.98 mm apart at Z = 1.49 m. It is harmless: all visibility tests
  below use a 3 mm tolerance.
- The web meshes (decimated to ≤ 15 k tris/organ, 128 k verts) also align. Silhouette IoU
  vs segid is 0.9994-0.9997, and the vertex-to-full-mesh distance is median 0.00 mm,
  p99 0.90 mm, max 1.33 mm.

### 1.2 How consistent are the 20 renders? Leave-one-out, network vs Cycles
`research_3d/fuse_eval.py` (`out/fuse_eval_depth.json`). For each held-out view h, h's own
geometry was coloured from the **other 19** renders (projection, depth test only, bilinear
colour) and compared with h's actual neural render. The identical procedure on the Cycles
`rgb_preview.png` gives the **floor**.

| mean of 20 held-out views | full frame PSNR | organ PSNR (covered px) |
|---|---|---|
| ULR (3 nearest-angle views, Buehler weights) from the neural renders | **27.45** (24.8-30.3) | **20.56** (17.7-24.2) |
| diffuse bake (cos²-weighted mean of all visible) from the neural renders | 27.76 (24.9-30.6) | 20.93 (17.9-24.8) |
| nearest single view | 26.39 | 19.30 |
| same ULR from the **Cycles GT** (floor) | 27.57 | 20.71 |
| same diffuse from the Cycles GT (floor) | 27.89 | 21.10 |
| *reference: the network itself vs Cycles at the real views* | *33.73* (29.7-37.8) | *26.15* |
| fused-from-network ULR vs Cycles GT at the held-out view | 27.28 | |

- Coverage: 95.9 % of a held-out view's organ pixels are seen by at least one of the other
  19 views. On average 5.2 views see each point, and the nearest source ray is a median
  **35.8°** away (30-47°).
- **Network minus Cycles gap: 0.17 dB (diffuse), 0.15 dB (ULR)** over 20 views, per-view
  −0.38 to +0.66. The network adds almost no multi-view inconsistency, which corroborates
  `EVALUATION/v25_gbuf_glint/multiview_summary.json` (gap −0.04 dB). A per-vertex
  cross-view colour spread gives the same answer (`recon_proto.py`): **20.5 (network) vs 19.3
  (Cycles)** RMS in 8-bit units.
- The error is inherent to the scene. `out/loo_v02_az+0_el+0_ulr_err.png` shows it sits on
  the liver's glints and wet highlights (view-dependent) and in never-seen concavities. The
  per-view strips are `out/loo_<view>.png`: held-out, ULR, diffuse, nearest, Cycles. Crops
  are in `out/loo_<view>_crop.png`.
- Visibility by depth test alone (what a shader can do) versus an extra organ-id test changes
  PSNR by < 0.05 dB (`out/fuse_eval_strict.json`) and raises coverage from 94.9 to 95.9 %.
- The lighting is world-fixed. The v25 three-point rig is placed once per subject in world
  space; only exposure and key energy are jittered, and only per subject. That is why fusion
  works at all.

### 1.3 Which 3-D representation? (`research_3d/bake_eval.py`, `out/bake_eval.json`)
Leave-one-out on 4 held-out views (v02, v07, v13, v18), rendered with the numba z-buffer
rasteriser in `common.py`.

| representation | size | LOO organ / full PSNR | look (`out/bake_v02_az+0_el+0_crop.png`) |
|---|---|---|---|
| per-vertex bake, web mesh (128 k v, 1.64 mm edges) | 0.5 MB colours | 21.52 / 28.86 | clean but soft, highlights smeared milky |
| per-vertex bake, full OBJ (276 k v) | 1.1 MB | 21.50 / 28.85 | same |
| per-vertex bake, OBJ ×4 subdivided (1.1 M v, 0.75 mm) | 4.4 MB | 21.44 / 28.79 | same |
| point cloud, 0.5 mm voxels (2.4 M pts) | 22 MB | 19.04 / 25.02 | sharper but aliased and holey |
| point cloud, 1 mm (0.76 M pts) | 6.8 MB | 19.34 / 25.00 | stair patterns |
| point cloud, 2 mm (0.18 M pts) | 1.6 MB | 19.44 / 24.66 | blocky |

- **Vertex density does not matter** (21.44-21.52 dB). The error is view-dependence, not
  resolution, so the existing web meshes are enough.
- Point clouds are about 2.5 dB worse and look worse. With exact meshes available there is
  no reason to throw the surface away.
- The 77-80 % of the organ **surface area** seen by at least one camera is the outer hull;
  the rest faces neighbouring organs. On the web meshes, 73 % of vertices are seen. Never-seen
  parts get the organ's flat colour.

### 1.4 Captured-view exactness and genuinely novel cameras
`research_3d/novel.py`, `research_3d/webref.py`:

| test | organ PSNR | full PSNR |
|---|---|---|
| ULR from all 20 at captured v02 / v11, full-res mesh (`out/captured_check.json`) | 55.4 / 60.1 | 42.0 / 41.2 |
| ULR from all 20 at captured v02, **web mesh** (`out/webref/webref.json`) | **42.2** | 41.5 |
| **baked** per-vertex colour at captured v02, web mesh (in-sample) | 22.2 | 30.2 |

- ULR is exact at captures. The baked mode is not, even at a capture: it averages the
  highlights away. `out/webref/_contact.png` shows ULR at v02, baked at v02, ULR at az36 el15
  and baked at az36 el15.
- Novel cameras (`out/novel.json`, `out/novel_panel.png`) at az/el (36,15), (108,45),
  (180,0), (252,−25), (324,75) and (0,90): 94.6-98.9 % of the visible organ pixels are seen
  by at least one of the 20 cameras, and the nearest capture is 20.6-37.9° away. **Views
  halfway between captures are as hard as the leave-one-out test** (72° azimuth spacing,
  30-50° elevation spacing), so §1.2's numbers describe them.
- Turntable at el +20°: `out/turntable_ulr.mp4` (36 frames), `out/turntable_sheet.png`.
  It is coherent, photoreal and seam-free. Highlights cross-fade (ghosted vessel glints on
  the liver between captures, see `out/novel_az36_el15_crop.png`).

### 1.5 Cost of the proposed stage (`research_3d/recon_proto.py`, `out/recon/recon.json`)
The stage reads the 20 depth EXRs and PNGs, bakes per-vertex colours for 128 k web-mesh
vertices, and writes cameras. **2.16 s, peak RSS 0.24 GB.** Outputs: `bake.bin` 511 KB and
`cameras.json` 11 KB. The web-space cameras match `camera_model.project` to 1.1e-6 px.

---

## 2. Approaches compared

| | quality | effort | runtime / memory | offline browser display | verdict |
|---|---|---|---|---|---|
| **(a) fused point cloud** | LOO 19.0-19.4 dB organs; aliasing and holes; view-independent | low | 8-11 s, ~1 GB (Python); 1.6-22 MB | easy (gl.POINTS) | dominated by (b) because the mesh is exact |
| **(b) per-vertex bake on the web meshes** | LOO 21.5 dB; at captures only 22.2 dB (highlights averaged away); soft, matte | low | 2.2 s / 0.24 GB; +0.5 MB | one extra vertex attribute; WebGL1 possible | **keep as fallback / "baked" mode** |
| **(b′) mesh + view-dependent projective texturing (ULR)** | **exact at the 20 captures (42 dB); LOO 20.6 dB organs / 27.4 dB full, = Cycles floor**; keeps sharpness and highlights, cross-fades them | medium (one WebGL2 shader + a visibility pre-pass) | backend same as (b); browser ~180 MB GPU at 1024² | WebGL2 + `EXT_color_buffer_float` (Chrome and Safari on Apple Silicon) | **recommended** |
| **(c) 3DGS, server (gsplat on L4)** | expected ≤ (b′) at novel views (same view-dependence floor, sparse 16-20 views, floaters); lossy at training views | medium-high: export ✓ (`export_colmap.py`), owner runs server, vendored viewer | est. 15-25 min per 30 k-step run on one L4 (unmeasured) | needs a vendored splat viewer (see below) | optional thesis experiment, not for the demo |
| **(c′) 3DGS on this Mac** | same as (c) | Brush (Apache-2.0) prebuilt `brush-app-aarch64-apple-darwin` v0.3.0, **download required, not done**; OpenSplat (AGPL-3.0, MPS) needs `brew install cmake opencv pytorch` + build | unknown; likely 2-4 GB, which conflicts with the demo's 3 GB-free rule | same | not recommended |
| **(d) render on demand** | the only true novel *neural* view | high: opt-in camera-override flag in `generate_training_dataset.py`, gbuffer + buffers + infer for one view | est. 4-7 s per click (Blender start-up dominates) | the result is just an image | future work; mention in thesis |

Notes on (c):
- **The local `gaussian-splatting/` clone (Inria) must not be used on this data.** It
  ignores `cx, cy` (symmetric frustum), which misplaces every pixel by (51.2, 20.5) px. A
  centred principal point lands only 72-83 % of the exported points on non-black pixels, vs
  88-97 % with the true one (`export_colmap.py --verify`). Its licence is also
  non-commercial / research-only (Inria + MPII), which matters given the Siemens context.
- gsplat (Apache-2.0) and its `examples/datasets/colmap.py` read `cx, cy` from PINHOLE
  cameras via `pycolmap`, so gsplat is the right tool.
- Viewer licences, if a splat ever went into the page:
  - `antimatter15/splat`: MIT, single-file WebGL, no dependencies. Its `.splat` format has
    no spherical harmonics, so colours are view-independent, which defeats 3DGS's one
    advantage over (b).
  - `mkkellogg/GaussianSplats3D`: MIT, three.js, SH up to degree 2, no longer developed.
  - `sparkjsdev/spark`: MIT, three.js, active.
  - All can be vendored; three.js would have to be vendored too.

---

## 3. Recommendation and honest caveats for the thesis

**Build (b′) with (b) as a toggle.** Card name: "3-D view: the 20 neural renders fused onto
the organ meshes". Suggested tagline: *exact at the 20 camera positions, blended in between.*

Caveats to state in the demo UI and the thesis:
1. **It is not a new neural render.** The network runs only at the 20 cameras. The
   in-between views are image-based interpolation of its outputs over the input mesh, and the
   geometry is the CT-derived mesh, not a reconstruction. Say "fused" or "projected", never
   "reconstructed by the network".
2. **Highlights are view-dependent.** Between captures, glints and wet highlights cross-fade
   and ghost (ULR), or smear into matte patches (baked mode). Mid-way views score about
   27 dB / 21 dB against a real render there, which is far below the network's own 33.7 dB /
   26.2 dB vs Cycles. Present them as a visualisation, not as quality evidence.
3. **The views are sparse.** 72° azimuth steps put the nearest capture up to about 38° away.
   4-5 % of the visible surface in novel views was never captured (shown in the organ's
   flat colour). Views above +60° or below −50° are only seen obliquely.
4. **Positive evidence, one subject.** The network's 20 renders fuse as coherently as Cycles'
   do (0.15-0.17 dB gap; colour spread 20.5 vs 19.3). This supports the thesis's multi-view
   consistency claim, but it is a smoke value from one subject (s0006), not a thesis metric.
   The metric is `eval_multiview.py`.
5. **Fusion only works because the lights are world-fixed.** Any future renderer version
   with camera-attached lights would break it.
6. **3DGS:** if a reviewer asks, geometry is known exactly and the views are sparse and
   view-dependent. §5 is the experiment that answers it with a number.

---

## 4. Execution plan (for the implementing agent)

Read first: `demo/PLAN.md` §7 (event contract and memory rules), this file,
`demo/research_3d/recon_proto.py` (the backend in 150 lines), `demo/research_3d/fuse_eval.py::blend`
(the reference blending maths) and `demo/research_3d/webref.py` (the viewer's test oracle).

Hard rules carried over from PLAN.md §7: one server; one run; ≥ 3 GB free before any run; no
installs; no CDN. Do not touch `scripts/training/*`.

### 4.1 Files
```
demo/recon.py                  NEW: build(run_dir, sid, emit=None) -> stats   (port of research_3d/recon_proto.py)
demo/pipeline.py               + stage "recon" after "infer" (STAGES, _run list, do_recon, _HINTS)
demo/server.py                 + POST /api/runs/<id>/recon  (backfill for runs made before this stage)
demo/static/js/reconviewer.js  NEW: WebGL2 viewer (ULR / baked / nearest), self-test
demo/static/js/app.js          + card 7 "3-D view", stepper node 7, replay handling
demo/static/css/app.css        + --recon accent (#F5E3A8, butter) and card styles
demo/static/index.html         + card markup (if cards are static)
```

### 4.2 Backend: `demo/recon.py`
Port `recon_proto.py` unchanged in maths, with these additions:
- `build(run_dir: Path, sid: str, emit=None)`. `emit(kind, **kw)` is a callback; pipeline
  passes a closure over `Run.progress/stat/artifact`. Order: read `web/manifest.json` and the
  organ `.bin`s, then for each view `vNN` (sorted) read `meta.json`, `camera_model.depth_m`
  and `output/<view>.png`, accumulate the per-vertex bake (depth test, `TOL_MM = 3.0`,
  weight cos², bilinear colour), and `emit("progress", done=i+1, total=20, label=view)`.
- If `gt/<view>.png` exists, accumulate the same weights on GT for the Cycles spread (as in
  `recon_proto.py`).
- Also compute **area-weighted seen fraction** from the web-mesh faces: a triangle counts as
  seen when all 3 vertices are seen. Expect about 0.77.
- Write to `<run>/recon/`:
  - `bake.bin`: `u32 nV` + `u8 rgba[nV]`, vertices concatenated in `manifest.organs` order.
    `a = 255` means seen by ≥ 1 camera. `a = 0` means never seen, and then rgb already holds
    the fallback (organ hex × 0.6). That way the shader can interpolate `vBake.rgb` across
    triangles that mix seen and unseen vertices without fading to black.
  - `cameras.json`: `{space, center_mm, tol_mm, organ_center_web[3], near_mm, far_mm, cameras:[{view, image, az, el, W, H, fx, fy, cx, cy, R[3][3], t[3], center[3]}]}`.
    - `R, t`: OpenCV world→camera in **web space** (mm, world − `center_mm`).
    - `fx, fy, cx, cy`: COLMAP pixel convention, top-left pixel centre (0.5, 0.5);
      `cx, cy = camera_model.principal_point(1024, 1024) = (563.2, 532.48)`.
    - `organ_center_web = meta.organ_center·1000 − center_mm`; `near_mm = 500`, `far_mm = 3000`.
    - `image` = `/runs/<id>/output/<view>.png`.
  - `viewer.json`: `{manifest: "/runs/<id>/web/manifest.json", cameras: ".../recon/cameras.json", bake: ".../recon/bake.bin", n_views: 20, stats: {...}}`.
  - `recon.json`: the stats.
- Include a `--check` CLI: `venv/bin/python demo/recon.py demo/runs/<id> --check` re-projects
  all web vertices with `cameras.json` and asserts they equal `camera_model.project` + 0.5
  within 1e-4 px (snippet: see the verification in §1.5; `research_3d/export_colmap.py::w2c_opencv`
  has the Blender→OpenCV flip).
- Errors: if web meshes are missing, raise `RuntimeError("web meshes missing — mesh stage did not run")`.

### 4.3 Pipeline integration and events (PLAN.md §7 style)
`STAGES = (..., "infer", "recon")`, and append `("recon", self.do_recon)` to the `_run` list.
`do_recon` runs **in-process** (0.24 GB; no subprocess, no pool) and returns stats.
`_HINTS["recon"] = "3-D view could not be built — the renders are unaffected"`.
If recon fails, the run is still `done` for the earlier stages. Emit the `error` event, but do
not mark the neural renders failed: set `status = "done"` if only recon failed.

```
recon   progress 1..20 (label = view)
        stat views "20" (int) · verts (int) · seen_pct % (float1, area-weighted) ·
             spread_net "RMS/255" (float1) · spread_cycles "RMS/255" (float1, only with gt/) · build_s s (float2)
        artifact viewer (json recon/viewer.json)   ← the UI needs only this
        artifact bake   (bin  recon/bake.bin)
        artifact cameras(json recon/cameras.json)  meta {n: 20}
        log "bake: 127786 vertices, 73.4 % seen by ≥1 camera"
```

### 4.4 Server backfill
`POST /api/runs/<id>/recon`:
- If `recon/viewer.json` exists, return `{viewer: url}`.
- Otherwise call `recon.build()` synchronously (≤ 5 s) under a module-level
  `threading.Lock` and return `{viewer: url, stats}`.
- Return 409 while a pipeline run holds `runs/.run.lock`.
- Do **not** append to `events.jsonl` (replay order must stay intact).

### 4.5 Frontend: `reconviewer.js` (WebGL2, no libraries)
**API:** `new ReconViewer(canvas)`; `await load(viewerJsonUrl, {meshBuffers?})` (reuse the
ArrayBuffers the mesh card already fetched); `setMode("ulr"|"baked"|"nearest")`; `snapTo(i)`;
`showCameras(bool)`; `dispose()`; `selftest(url)`.

**Load:**
1. Fetch the organ `.bin`s (format PLAN.md §7) and upload one VAO per organ: `aPos`
   (f32×3, web mm), `aNrm` (f32×3), `aBake` (u8×4 normalised, slice of `bake.bin`), and the
   u32 index buffer.
2. Photo array: `TEXTURE_2D_ARRAY`, `texStorage3D(RGBA8, 1024, 1024, 20)`. Fill each layer
   with `createImageBitmap(await (await fetch(img)).blob())` via `texSubImage3D`, with
   `UNPACK_FLIP_Y_WEBGL = false` (row 0 = image top), `LINEAR`, `CLAMP_TO_EDGE`, no
   mipmaps. Use plain RGBA8, not SRGB8: the PNG values go to the screen unchanged.
3. Visibility array: `TEXTURE_2D_ARRAY R32F 1024×1024×20` (`NEAREST`; needs
   `EXT_color_buffer_float`), one `DEPTH_COMPONENT24` renderbuffer 1024², one FBO. For each
   camera i: `framebufferTextureLayer(..., i)`, clear colour 1e9, clear depth, depth test
   LESS, no culling, draw all organs with the **visibility program**.

**Projection** (shared by all passes; the camera is always `(R, t, fx, fy, cx, cy)`, so a
snapped view matches its PNG to the pixel):
```glsl
// uploaded: uniformMatrix3fv(loc, false, R_rowmajor_flat)  =>  GLSL holds R^T, so use  p * uR  (== R p)
vec3 Xc = aPos * uR + uT;                              // mm, OpenCV camera, z forward
vec2 px = uK.xy * Xc.xy / Xc.z + uK.zw;                // pixels, COLMAP convention, row 0 = top
float zn = 2.0 * (Xc.z - uNF.x) / (uNF.y - uNF.x) - 1.0;
// visibility pass (renders into texture rows top->t=0):  ndc = 2*px/size - 1
// screen pass (y up on screen):                            ndc = vec2(2*px.x/W - 1, 1 - 2*px.y/H)
gl_Position = vec4(ndc * Xc.z, zn * Xc.z, Xc.z);
```
The visibility fragment shader writes `out float o; o = vZ;` (varying `vZ = Xc.z`).

**Main fragment shader** (the GPU twin of `research_3d/fuse_eval.py::blend` +
`webref.py` hole handling):
```glsl
#version 300 es
precision highp float; precision highp sampler2DArray;
#define NCAM 20
uniform mat3 uR[NCAM]; uniform vec3 uT[NCAM]; uniform vec4 uK[NCAM]; uniform vec3 uC[NCAM];
uniform sampler2DArray uImg, uVis; uniform vec2 uImgSize;   // (1024, 1024)
uniform vec3 uEye; uniform int uMode; uniform float uTol;
in vec3 vPos; in vec3 vNrm; in vec4 vBake; out vec4 frag;
void main() {
  vec3 fallback = vBake.rgb;              // baked colour; organ x 0.6 where never seen (see bake.bin)
  if (uMode == 1) { frag = vec4(fallback, 1.0); return; }            // baked
  vec3 n = normalize(vNrm), dEye = normalize(uEye - vPos);
  float th[4]; int id[4]; vec2 uv4[4]; float cs[4];
  for (int k = 0; k < 4; k++) { th[k] = 1e9; id[k] = -1; }
  for (int i = 0; i < NCAM; i++) {
    vec3 Xc = vPos * uR[i] + uT[i];
    if (Xc.z < 1.0) continue;
    vec2 uv = (uK[i].xy * Xc.xy / Xc.z + uK[i].zw) / uImgSize;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
    if (abs(texture(uVis, vec3(uv, float(i))).r - Xc.z) > uTol) continue;   // occluded
    vec3 dC = normalize(uC[i] - vPos);
    float a = acos(clamp(dot(dC, dEye), -1.0, 1.0)), c = max(dot(dC, n), 0.0);
    for (int k = 0; k < 4; k++) if (a < th[k]) {
      for (int m = 3; m > k; m--) { th[m] = th[m-1]; id[m] = id[m-1]; uv4[m] = uv4[m-1]; cs[m] = cs[m-1]; }
      th[k] = a; id[k] = i; uv4[k] = uv; cs[k] = c; break;
    }
  }
  if (id[0] < 0) { frag = vec4(fallback, 1.0); return; }            // never captured
  vec3 c0 = texture(uImg, vec3(uv4[0], float(id[0]))).rgb;
  if (uMode == 2) { frag = vec4(c0, 1.0); return; }                 // nearest
  float th4 = id[3] >= 0 ? th[3] : 3.14159265;
  vec3 acc = vec3(0.0); float ws = 0.0;
  for (int k = 0; k < 3; k++) {
    if (id[k] < 0) break;
    float w = clamp(1.0 - th[k] / th4, 0.0, 1.0) / (th[k] + 1e-3) * max(cs[k], 0.05);
    acc += w * texture(uImg, vec3(uv4[k], float(id[k]))).rgb; ws += w;
  }
  frag = vec4(ws > 0.0 ? acc / ws : c0, 1.0);
}
```
The uniform arrays use 20 × (9 + 3 + 4 + 3) floats, about 95 vec4, which is under WebGL2's
minimum of 224. No lighting is applied: the colours *are* the renders. Draw into a canvas
with `alpha:false` and a black clear colour, so the black backgrounds match the gallery.

**Orbit camera** (Z-up, because the 20 cameras orbit world Z; this is not meshviewer.js's
Y-up):
- `C = target + d·(cos el cos az, cos el sin az, sin el)` with `target = organ_center_web`.
- `f = normalize(target − C)`, `r = normalize(cross(f, ẑ))`, `u = cross(r, f)`.
- `R` rows = `[r; −u; f]`, `t = −R C`.
- Clamp el to [−89°, 89°] and d to [0.6, 2.5] × mean capture distance.
- Intrinsics are always capture-style: `fx = fy =` last snapped capture's fx (default v02),
  `cx, cy = (563.2, 532.48)`, all scaled by `S/1024` for a square canvas of S backing
  pixels. Then snapping never jumps.
- Drag: az/el. Wheel: d. Double-click: reset to v02.
- `snapTo(i)` sets `(R, t, K)` exactly to camera i. Derive `(az, el, d)` from its centre so
  the next drag continues from there, and animate (≤ 400 ms, honour reduced motion).

**HUD:** "nearest render vNN · 12°", which is the angle between `(C − target)` and
`(C_i − target)`, plus an **"exact"** pill when snapped.

**Self-test** (`?recon_selftest=1`, needed for R3):
- Load the cameras in `recon/selftest/cameras.json` (copied from
  `demo/research_3d/out/webref/cameras.json`). Their schema equals `cameras.json` entries.
- Render each offscreen into a 1024² RGBA8 FBO with the screen-pass projection, in modes
  ulr and baked.
- `readPixels`, flip rows (GL row 0 = bottom), compare with `recon/selftest/<name>_<mode>.png`
  (copied from `research_3d/out/webref/`) via PSNR in JS.
- Log `console.info("[recon-selftest]", json)` and show it in the card's Details.

**Fallback:** without WebGL2 or `EXT_color_buffer_float`, hide ULR/nearest and show the baked
mode only (the main program with `uMode = 1` needs no visibility pass). Without WebGL2 at
all, show a static note ("the 3-D view needs WebGL2") instead of the canvas.

### 4.6 Card 7 "3-D view" (after "Neural rendering", before Results)
- Top wash in `--recon`, stepper node 7 (a cube icon).
- One line: "The 20 neural renders projected onto the organ meshes. Exact at the 20 cameras,
  blended in between."
- Left: square canvas in a `--stage-dark` well, with a bottom-left HUD.
- Right:
  - segmented control **Blended · Baked · Nearest**;
  - a **Cameras** toggle (20 small frusta as `GL_LINES`, coloured per elevation ring, the
    nearest one highlighted);
  - a **4 × 5 snap grid** laid out like the gallery (rows el −50/0/30/60, cols az 0-288);
  - **Compare** (only when snapped): the real `output/<view>.png` as an `<img>` overlay with
    a drag wipe (`clip-path`), which shows the match.
- Stat tiles: views 20 · surface seen 77 % · cross-view spread network 20.5 / Cycles 19.3
  (Cycles only when GT exists) · build 2.2 s.
- Details (collapsible): caveats 1-3 of §3 in plain words, and the self-test result.
- Lazy WebGL: create the context only when the card enters the viewport **and** the
  `viewer` artifact exists. `dispose()` (`WEBGL_lose_context`) when another run is loaded.
- Replay of a run without recon events: show "Build 3-D view" (POST §4.4) and load the
  viewer on success.

### 4.7 Budgets
| item | budget | measured basis |
|---|---|---|
| backend stage | < 5 s, < 0.5 GB RSS, in-process | 2.16 s, 0.24 GB (`recon_proto.py`) |
| disk per run | +0.6 MB | bake 511 KB + cameras 11 KB |
| page download | +0.5 MB new (meshes 6.1 MB + PNGs 5.5 MB already loaded by earlier cards) | |
| GPU/browser | ≈ 180 MB at 1024² layers (84 MB photos + 84 MB R32F + 4 MB depth + 6 MB VBO); ≈ 50 MB with a `?recon_res=512` option | arithmetic |
| init | 20 visibility passes over 254 k tris + 20 image decodes, < 1 s | estimate, measure in R6 |
| frame | < 16 ms at 1024² backing store on the M1 Pro | estimate, measure in R6 |

### 4.8 Acceptance (report each with numbers)
- **R1 backend parity:** `recon.build` on `demo/runs/20260922-165227_s0006`. `bake.bin` must
  be byte-identical to `demo/research_3d/out/recon/bake.bin` (md5 `7c97617dea9e0e4661961906f935c46f`), and the stats must equal
  `demo/research_3d/out/recon/recon.json`: vertices 127786, seen 0.7337, spread 20.52 / 19.33.
  Time < 5 s, RSS < 0.5 GB.
  *(Post-ship fix, white-speckle report: never-seen vertices now get a geodesic colour
  infill from their seen neighbours instead of a flat organ-hex×0.6 fill — see
  `recon.infill_unseen`. This changes `bake.bin`'s bytes for `a=0` vertices only, so parity
  is now "seen (`a=255`) vertices byte-identical, stats unchanged" rather than whole-file
  md5 match; the flat colour remains the fallback only for a component with no seen vertex
  at all.)*
- **R2 cameras:** `demo/recon.py <run> --check` gives max deviation ≤ 1e-4 px (the research
  copy gives 1.1e-6).
- **R3 shader numerics:** the self-test gives PSNR ≥ 38 dB for all 6 references
  (3 cameras × ulr/baked). The ULR render at v02 vs `output/v02_az+0_el+0.png` is ≥ 38 dB on
  organ pixels (the CPU reference gives 42.2).
- **R4 interaction:** drag, zoom, snap to all 20, and the compare wipe at v02 shows no
  visible offset.
- **R5 replay:** an old run gets "Build 3-D view" (POST), then the viewer opens. A second
  POST returns the cached result.
- **R6 performance:** init time and frame time (Chrome performance panel) meet §4.7. Free
  memory during a live run stays ≥ `MIN_FREE_GB`.
- **R7 live run:** `recon` events arrive after `infer` and before `run_done`. The total run
  time rises by ≈ 2-3 s.
- **R8:** no console errors; the no-float fallback (baked only) works when forced with a
  `?recon_nofloat=1` flag.

---

## 5. Optional: 3DGS on gpusrv02 (the owner runs these)

The goal is one honest number for the thesis: can 3DGS beat mesh + ULR on held-out views?

**Data (already exported, 14 MB):** `demo/research_3d/out/gs_s0006/`.
- `images/`: the 20 neural renders.
- `sparse/0/{cameras,images,points3D}.bin`: one PINHOLE camera per view with the
  lens-shifted `cx, cy`, OpenCV world→camera in metres, and 184,290 initial points from the
  fused depth (2 mm voxels, render colours, empty tracks).
- These cameras equal `camera_model.project` to ≤ 4e-13 px.

To re-export another run: `venv/bin/python demo/research_3d/export_colmap.py --run demo/runs/<id> --out <dir>`
(about 6 s and 0.9 GB; it runs `--verify` automatically).

```bash
# --- on the Mac ---
scp -r demo/research_3d/out/gs_s0006 <user>@gpusrv02:~/gs/

# --- on gpusrv02 (fresh env; pick the cu tag that `nvidia-smi` supports) ---
mkdir -p ~/gs && cd ~/gs
python3 -m venv gsenv && source gsenv/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat && pip install -e . && pip install -r examples/requirements.txt   # first import JIT-builds CUDA (needs nvcc);
                                                                             # alternative: prebuilt wheel from https://docs.gsplat.studio/whl
cd examples
# test_every 5  ->  test = v01, v06, v11, v16 ; train = the other 16  (same split as research_3d/split_eval.py)
CUDA_VISIBLE_DEVICES=0 nohup python simple_trainer.py default \
    --data_dir ~/gs/gs_s0006 --data_factor 1 --result_dir ~/gs/results/s0006_default \
    --test_every 5 --max_steps 30000 --eval_steps 7000 30000 --save_steps 30000 \
    --save_ply --ply_steps 30000 --disable_viewer > ~/gs/default.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python simple_trainer.py mcmc \
    --data_dir ~/gs/gs_s0006 --data_factor 1 --result_dir ~/gs/results/s0006_mcmc \
    --test_every 5 --max_steps 30000 --eval_steps 7000 30000 --save_steps 30000 \
    --save_ply --ply_steps 30000 --disable_viewer > ~/gs/mcmc.log 2>&1 &
tail -f ~/gs/default.log                       # est. 15-25 min per run on one L4 (unmeasured)
cat ~/gs/results/s0006_*/stats/val_step29999.json     # gsplat's own PSNR/SSIM/LPIPS on the 4 test views

# --- back on the Mac ---
scp -r <user>@gpusrv02:~/gs/results/s0006_default/renders demo/research_3d/out/gs_default_renders
venv/bin/python demo/research_3d/split_eval.py --gs-sbs 'demo/research_3d/out/gs_default_renders/val_step29999_*.png'
```

- If `pip install -r examples/requirements.txt` fights the torch pin, install torch first as
  above, then the rest. Alternatively check out the gsplat git tag matching `pip show gsplat`.
- Leave `--depth_loss` off: it needs COLMAP point tracks, and the export has none.
- `split_eval.py --gs-sbs` checks that each image's left half *is* the expected test view
  (> 40 dB), then scores gsplat's renders exactly like ULR/diffuse.

**The number to beat** (`out/split_eval.json`, 16 train / 4 test, vs the held-out neural
renders):

| method | full-frame PSNR | organ PSNR |
|---|---|---|
| mesh + ULR from 16 | 26.16 | 18.68 |
| mesh + diffuse bake from 16 | 26.50 | 19.03 |

**Decision rule:** put 3DGS in the demo only if it beats ULR by ≥ 1 dB on the organs **and**
shows no floaters in a turntable. Also report its PSNR on the *training* views; ULR is exact
there.

**Do not** use `./gaussian-splatting` (Inria): it ignores `cx, cy`, and its licence is
research-only.

---

## 6. Prototype index (`demo/research_3d/`, all ≤ 1.8 GB peak RSS, one process at a time)

| script | what | runtime / peak RSS | outputs |
|---|---|---|---|
| `common.py` | loaders, OBJ reader, numba z-buffer rasteriser (perspective-correct), look-at cameras in the generator's convention (rotation identical to v07's to 1e-4), PSNR | | |
| `align.py` | Q1 alignment | 3 s / 0.5 GB | `out/align.json`, `out/align_v02_boundaries*.png` |
| `fuse_eval.py` | 20× leave-one-out, ULR/diffuse/nearest, network vs Cycles (`VIS_MODE=depth|strict`) | 36 s / 1.8 GB | `out/fuse_eval_{depth,strict}.json`, `out/loo_*.png` |
| `bake_eval.py` | representation comparison (per-vertex bakes, point clouds) | 42 s / 1.4 GB | `out/bake_eval.json`, `out/bake_v02_az+0_el+0_crop.png` |
| `novel.py` | novel cameras, turntable, captured-view exactness | 34 s / 1.5 GB | `out/novel_panel.png`, `out/novel_*`, `out/turntable_ulr.mp4`, `out/turntable_sheet.png`, `out/captured_check.json` |
| `recon_proto.py` | reference for the `recon` stage | 2.2 s / 0.24 GB | `out/recon/{bake.bin,cameras.json,recon.json}` |
| `webref.py` | CPU oracle for the WebGL self-test (web mesh, bake.bin) | 5 s / 1.0 GB | `out/webref/{cameras.json,*_ulr.png,*_baked.png,webref.json}` |
| `split_eval.py` | comparator on gsplat's split; scores gsplat renders | 5 s / 1.0 GB | `out/split_eval.json` |
| `export_colmap.py` | COLMAP binary export for gsplat/Brush + independent verify | 6 s / 0.9 GB | `out/gs_s0006/` |

`demo/research_3d/out/` is about 27 MB and is not covered by `.gitignore` (only `demo/runs/`
is). Add `demo/research_3d/out/` there if it should stay out of git.
