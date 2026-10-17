# CT → Photoreal — local demo app (plan)

A local web app that walks one CT through the whole Stage-2 pipeline, live, with every
stage's input → output, stats and timing, ending in the neural renders of all 20 views.

**Principle: the demo must run the thesis code, not a re-implementation.**
Meshes come from `extract_meshes.py`, G-buffers from `generate_training_dataset.py` in
Blender, and the network input from `infer.load_input` (the exact function training
used). The demo only orchestrates, times and visualises. If the demo reimplements a
pipeline step, its output would be worthless as evidence.

---

## 0. Verified facts (2026-09-22, measured on this Mac)

| item | value |
|---|---|
| checkpoint | `results/training_runs/gbuf_glint/checkpoints/epoch_060.pt` (2.1 GB; keys G, D, opt_G, opt_D, epoch, cfg; loads in 1.5 s) |
| preset | `full1024_glint` (inputs seg_rgb, depth, normals, segid, texture, glint → 17 ch, `upsample="resize"`) |
| generator | 182.8 M params; **364 ms/view on MPS** (warm), 1024² |
| input prep | `infer.load_input` ≈ 2.0 s/view on CPU (texture + glint dominate) |
| textures | `data/renders/textures/tissue_server/` — server tiles, 55/55 md5 match. **Never** use `data/renders/textures/tissue` (local tiles, wrong seed) |
| reference subject | `data/training_dataset_v25/s0006` — 20 views, G-buffers + `rgb_preview.png` GT, no `render.exr`. s0006 = **test** split |
| raw CT | `/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201/<sid>/{ct.nii.gz, segmentations/*.nii.gz}` |
| Blender | 5.1.2, `/Applications/Blender.app/Contents/MacOS/Blender` (also `blender` on PATH); its Python has numpy + nibabel |
| venv | `venv/bin/python` 3.10, torch 2.11 (MPS ok), nibabel, pyvista, opencv, OpenEXR, skimage. **No web framework** → stdlib server |
| smoke test | s0006 v02: output vs `rgb_preview.png` = 34.4 dB, vessel glints visible |
| server numbers (for honest comparison only) | Cycles 53.0 s/frame (server, supplied); network 98 ms/frame on L4 (`EVALUATION/v25_gbuf_glint/speed.json`) |

Channel layout of the 17-ch stack (`infer.load_input`, order of `input_buffers`):
`0-2 seg_rgb | 3 depth | 4-6 normals | 7 segid | 8-10 albedo | 11-13 normal-map | 14 roughness | 15 vessel glints | 16 flat highlights` — all in [-1, 1], background −1.

---

## 1. Layout

```
demo/
  PLAN.md                  this file
  README.md                how to launch + what each stage is
  run_demo.sh              exec venv/bin/python demo/server.py --open "$@"
  config.py                every path/constant in one place (below)
  server.py                stdlib ThreadingHTTPServer, JSON API + SSE, static files
  pipeline.py              Run object: stages, event emission, timing, run dir
  viz.py                   previews: CT windowing, mask overlays, EXR → PNG colormaps, web meshes
  static/
    index.html
    css/app.css
    js/app.js              state, SSE client, stage cards, results, lightbox
    js/meshviewer.js       vanilla WebGL organ viewer (no three.js — works offline)
    js/orbit.js            20-camera orbit diagram (canvas 2D)
    js/ui.js               count-up numbers, timers, small helpers
  runs/                    (gitignored) one folder per run
```

Only files outside `demo/` that may change:
- `scripts/training_dataset/generate_training_dataset.py`: two opt-in flags (§3.4).
- `.gitignore` (already done).

Do **not** touch `scripts/training/*`, the v25 renderer, `data/renders/textures/tissue`, or
anything under `data/training_dataset_v25` (read-only reference). No git commits.

### config.py
```python
REPO        = Path(__file__).resolve().parents[1]
CKPT        = REPO / "results/training_runs/gbuf_glint/checkpoints/epoch_060.pt"
PRESET      = "full1024_glint"
TEX_DIR     = REPO / "data/renders/textures/tissue_server"
GT_ROOTS    = [REPO / "data/training_dataset_v25"]
BLENDER     = "/Applications/Blender.app/Contents/MacOS/Blender"  # fallback shutil.which("blender")
BROWSE_HOME = Path("/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201")
RUNS        = REPO / "demo/runs"
SERVER_REF  = {"cycles_s_per_frame": 53.025, "cycles_where": "server, Cycles (supplied)",
               "net_ms_per_frame_l4": 98.3}
```

---

## 2. Run directory

```
demo/runs/<YYYYmmdd-HHMMSS>_<sid>/
  events.jsonl            every event, in order, with t (s since run start) — enables replay
  run.json                inputs, final timings, status
  ct/                     axial|coronal|sagittal.png, axial_stack.jpg (sprite), info.json
  masks/                  overlay_*.png, overlay_stack.jpg, volumes.json
  meshes/<sid>/<organ>.obj   (exactly what extract_meshes.py writes)
  web/meshes/<organ>.bin + manifest.json   (decimated, for the WebGL viewer)
  gbuffers/<sid>/vNN_.../{seg.png, depth.exr, normals.exr, segid.exr, meta.json} + tissue_ids.json, summary.json
  preview/<view>/{seg,depth,normals,segid,albedo,nmap,rough,glint,highlight}.png  (512 px)
  stack/<view>.npy        float16 17-ch input (so inference = exactly what was visualised)
  output/<view>.png (1024) + output/thumb/<view>.jpg (320)
  gt/<view>.png           only if GT is available AND alignment passes (§3.6)
```

---

## 3. Backend

### 3.1 Events (the contract between backend and UI)
One JSON object per line in `events.jsonl`, also streamed over SSE as `data: {...}\n\n`.
Every event has `t` (float seconds since run start), `type`, and `stage` when relevant.
Stages, in order: `ct`, `masks`, `mesh`, `gbuffer`, `buffers`, `infer`.

```
run_start   {run_id, subject, ct_path, split}
stage_start {stage}
progress    {stage, done, total, label}              # label = organ or view name
stat        {stage, key, label, value, unit, fmt}    # fmt: "int"|"float1"|"float2"|"text"|"ms"|"s"
artifact    {stage, key, url, view?, meta?}          # url served from /runs/<run_id>/...
log         {stage, line}                            # selected, human-readable lines only
stage_done  {stage, seconds, summary}
error       {stage, message, hint}
run_done    {seconds_total, timings: {stage: seconds}}
```
Artifact keys the UI relies on:
- ct: `axial`, `coronal`, `sagittal`, `axial_stack` (+meta `{n, w, h}`), `info` (json url)
- masks: `overlay_axial|coronal|sagittal`, `overlay_stack`, `volumes` (json url)
- mesh: `organ` (one per organ, meta `{name, color, verts, tris, ms}`), `manifest`
- gbuffer: `cameras` (json: per view `{view, cam_pos, target, az, el}` + `organ_center`, `radius`), per view `seg|depth|normals|segid` with `view`
- buffers: per view `albedo|nmap|rough|glint|highlight` with `view`
- infer: per view `output` (+ `thumb`) and `gt` when available

### 3.2 Stage `ct` (also runs on selection, before “Run”)
`POST /api/inspect {path}` accepts a `ct.nii.gz` file or a subject folder containing one.
- nibabel load; report: shape, voxel spacing (mm), physical extent (cm), dtype, orientation
  (`nib.aff2axcodes`), HU min / p1 / p99 / max (on a strided sample), file size, load time.
- Checks (each → pass / warn / fail, with one-line reason):
  3-D volume · spacing in [0.3, 10] mm · HU range looks like CT (min ≤ −900) ·
  `segmentations/` present · how many of the 29 tissues (`extract_meshes.TISSUE_NAMES`) exist ·
  split via `splits._bucket(sid, 1337, 0.075, 0.075)` → badge (test/val = held out; **train = warn “seen in training”**) ·
  GT reference available in `GT_ROOTS`.
- Previews: `nib.as_closest_canonical` for display only, soft-tissue window W400/L40,
  anterior up; three mid-slices at 512 px; axial sprite of ~64 evenly spaced slices at 256 px
  (single JPEG, vertical strip) for the scrubber.
- Subject id: the TotalSegmentator folder name (`sNNNN`) when it matches — **required**,
  because the generator seeds camera jitter with md5(subject). Otherwise a sanitised stem.
- Arbitrary CT without `segmentations/`: `fail` with hint “needs TotalSegmentator masks”.
  (Running TotalSegmentator is out of scope; it is not installed.)

### 3.3 Stage `masks`
Load the 29 masks that exist; volume per organ (mL); colour from
`glint_model.tissue_params()[name]["hex"]` (the flat seg colours the renderer uses).
Overlay previews on the same slices + an overlay sprite aligned with the CT sprite.
Stats: tissues found / 29, total segmented volume (L), time.

### 3.4 Stage `mesh`
For each organ: `extract_meshes.extract_mesh(seg_path, zooms)` + `extract_meshes.save_obj`
into `runs/<id>/meshes/<sid>/` (identical to the dataset path → identical camera framing).
Emit `progress` + `artifact organ` as each finishes so the viewer can pop it in.
Web mesh: `pyvista` decimate to ≤ 15 k triangles/organ, centre on the joint bbox centre,
write little-endian binary `[uint32 nV, uint32 nI, float32 pos*3nV, float32 nrm*3nV, uint32 idx*nI]`.
Stats: organs, total vertices, total triangles, time (and per-organ ms in details).

### 3.5 Stage `gbuffer` — Blender, G-buffers only
Add two **opt-in** flags to `generate_training_dataset.py`; defaults must leave the script
byte-for-byte behaviourally identical (verify with `git diff`):
- `--mesh_dir` (default `"data/meshes"`) replacing the hard-coded `Path("data/meshes")`.
- `--gbuffer_only` (store_true): skip the Cycles `render.exr` + `rgb_preview.png` block only.
  Everything else (seed, jitter order, materials, lights, seg, depth/normals/segid, meta.json)
  runs unchanged — the RNG draw order must not change.

Run as a subprocess from REPO as cwd, mirroring `run_dataset_generation.sh`:
`BLENDER --background --python scripts/training_dataset/generate_training_dataset.py -- --subject <sid> --mesh_dir <run>/meshes --output_dir <run>/gbuffers --tex_dir <TEX_DIR> --size 1024 --gbuffer_only`
(pass `--dataset` = the CT's parent dataset folder; framing comes from the meshes anyway).
Parse stdout line by line: `--- View [NN/20] <label>` → progress; `gbuffers →` → that view
is done → make its 4 previews and emit; `organ bbox:` and `sampling  :` → stats.
After the 1st view, emit `cameras` (all 20 camera positions are only known as views complete —
emit an updated `cameras` artifact after each view; the orbit diagram re-reads it).
Previews: seg as is; depth → soft pastel colormap on the organ mask (near = light), bg = plum
`#1E1B2B`; normals `(n+1)/2`; segid → organ hex colours.
Stats: views 20, s/view (mean), resolution, Blender version, mm/px, total.

### 3.6 Stage `buffers`
cfg = `preset("full1024_glint")`, `cfg.data.tex_dir = str(TEX_DIR)`; lut = `infer.build_segid_lut(subject_dir)`.
Per view in a `ProcessPoolExecutor(max_workers=min(6, os.cpu_count()-2))`:
`x = infer.load_input(view_dir, cfg, lut)` → `stack/<view>.npy` (float16) and previews of
ch 8-10 albedo, 11-13 normal-map, 14 roughness, 15 vessel glints, 16 highlights
(`(c+1)/2`, background → plum). Time texture and glint separately by wrapping
`infer.texture_buffer` and `infer.glint_model.glint_buffer` with timers inside the worker.
Stats: channels 17 (with the group breakdown), ms/view texture, ms/view glint, total.

### 3.7 Stage `infer`
Model loaded **once at server start** in a background thread (`/api/status` → loading|ready,
device); device MPS → fallback CPU; one warm-up forward. `torch.load(..., weights_only=False)`,
keep only `G` (free the rest). Per view: forward on the saved stack, write output PNG + thumb.
Stats: ms/view (mean excluding the first), device, params (182.8 M), total.
GT: if `GT_ROOTS/<sid>/<view>/rgb_preview.png` exists **and** the reference `seg.png` matches
our `seg.png` (mean abs diff < 2/255 and < 1 % pixels differing) → copy to `gt/` and emit `gt`.
Otherwise emit a `log` saying why GT is hidden. This guard stops a misaligned comparison.

### 3.8 Server (`server.py`)
`ThreadingHTTPServer` on 127.0.0.1:8765 (`--port`), `--open` opens the browser.
```
GET  /                         static/index.html          GET /static/*
GET  /api/status               {model: loading|ready|error, device, ckpt, preset, busy}
GET  /api/browse?path=         dirs + files (flag .nii/.nii.gz; flag folders that contain ct.nii.gz)
POST /api/pick                 native macOS picker via osascript `choose file` → {path} (cancel → {})
POST /api/inspect {path}       ct stage result + checks + preview urls (cached per path)
POST /api/run {path}           start run (409 if one is running) → {run_id}
GET  /api/runs                 past runs, newest first (subject, date, status, total s)
GET  /api/runs/<id>/events     SSE: live tail while running; for a finished run streams the file
                               (?replay=1&speed=4 → re-timed replay; default instant)
GET  /runs/<id>/<path>         files from that run dir only (reject .. / absolute)
```
One run at a time (lock). Each stage in try/except → `error` event with a hint; later
stages are skipped, the UI shows which one failed.

---

## 4. Frontend — look & feel

Offline, no frameworks, no CDN, no build step. ES modules.

**Type:** headings `ui-rounded, "SF Pro Rounded", -apple-system, system-ui`; body
`-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui`; numbers `tabular-nums`;
code/paths `ui-monospace, "SF Mono"`. Large, calm hierarchy: 40/28/18/14 px.

**Palette (CSS custom properties):**
```
--bg #FBF8F4  --surface #FFFFFF  --surface-2 #F6F2FA  --line #ECE6F2
--ink #2E2A3B --ink-2 #6B6580   --ink-3 #A39DB5
--ct #C9B8F0 (lavender) --masks #F9C6D9 (pink) --mesh #BDE8D4 (mint)
--gbuffer #BCDDF7 (sky) --buffers #FFD9B8 (peach) --infer #E9C6F2 (orchid)
--ok #8FD3B0 --warn #F7C98B --bad #F29CA3 --stage-dark #1E1B2B (image wells)
```
Cards: white, 20 px radius, 1 px `--line` border, shadow `0 1px 2px rgba(46,42,59,.04), 0 8px 24px rgba(46,42,59,.06)`.
Each stage card carries its accent as a soft top wash (accent → transparent) and a
coloured stage number badge. Renders always sit in dark `--stage-dark` rounded wells (the
images have black backgrounds — a cream frame would look cheap).

**Motion:** only transform/opacity/filter; easing `cubic-bezier(.2,.8,.2,1)`, 250-600 ms;
honour `prefers-reduced-motion` (disable non-essential animation).
- stepper: completed segments fill with a left→right gradient sweep; active node breathes (scale 1→1.06)
- numbers count up (rAF, 700 ms, easeOutCubic); running stage shows a live mm:ss.s timer
- input → output arrow: dashed line with flowing dash-offset while running, solid when done
- images enter with a fade + 8 px rise; neural outputs “develop”: `blur(14px) saturate(.2)` → sharp over 700 ms
- gallery tiles stagger in (30 ms each); check marks draw (stroke-dashoffset)
- mesh organs pop in (scale 0.6→1, ease-out-back) as each `organ` artifact arrives

**Page, top to bottom:**
1. **Top bar** — “CT → Photoreal” wordmark, model chip `gbuf_glint · epoch 60 · MPS` with a
   status dot (pulsing while loading, mint when ready), “Recent runs” button (drawer).
2. **Hero** — one-line pitch (“A CT scan in, twenty photoreal views out — Cycles-quality in
   milliseconds per view”), and the 6-node **pipeline stepper** (icon + label per stage;
   click scrolls to the card).
3. **Choose a CT** — big dashed drop card with a “Browse…” button (native picker via
   `/api/pick`; “Browse in app” link opens a modal folder browser with breadcrumb, folder
   rows, `.nii.gz` highlighted, `ct.nii.gz`-containing folders marked). After choosing:
   CT info card — stat tiles (shape, spacing, extent, HU range, orientation, size),
   checks list with drawn check marks, split badge, three orthogonal slices + axial scrubber
   (range slider over the sprite) with a “show masks” toggle once masks exist.
   Primary pill button **Run pipeline** (lavender→pink gradient, subtle shimmer on hover),
   disabled until checks pass.
4. **Stage timeline** — vertical, a thin line on the left links the cards. Each card: number
   badge, title, one-line “what happens here”, status pill (waiting / running · timer / done · 12.3 s),
   an **Input → Output** row, a row of 3-4 stat tiles, and a collapsible “Details”.
   - *Segmentation masks*: CT slice → colour overlay; organ chips with a tiny volume bar.
   - *Mesh extraction*: overlay → WebGL viewer (auto-rotate, drag to orbit, wheel zoom, soft
     two-light shading, organ colours); details = per-organ table (verts, tris, ms).
   - *G-buffer rendering*: orbit diagram (organ sphere, 5 azimuths × 4 elevations as dots,
     current camera draws a thin frustum line to the centre, dots fill in as views finish)
     → 2×2 tiles seg / depth / normals / segid of the latest view (crossfade), filmstrip of
     finished seg thumbs.
   - *Texture & glint buffers*: seg thumb → tiles albedo / normal map / roughness / vessel
     glints / highlights; plus the **17-channel stack** as fanned translucent layer cards
     (CSS 3D perspective; fan wider on hover), each labelled with its channel group.
   - *Neural rendering*: the fanned stack → the output developing in; ms/view, device.
5. **Results** — the hero render large (default the first equatorial view, `v02`), caption
   with view name + az/el. Click → **lightbox**: tabs *Neural render* · *Inputs* (all 9
   previews for that view) · *Compare with Cycles* (drag-slider before/after; tab hidden
   when no GT); ← → switch views, Esc closes. Below: the **orbit gallery**, 4 rows
   (el −50°, 0°, 30°, 60°) × 5 columns (az 0°, 72°, 144°, 216°, 288°) with axis labels, and a
   segmented control *Output · Seg · Normals · Glints* to show intermediate representations.
6. **Summary** — stacked horizontal bar of stage times (accent colours, widths animate in),
   total wall-clock, neural ms/view, and a small, honestly labelled reference row:
   “Cycles on the server: 53 s per view (measured) · this network on an L4: 98 ms per view”.
7. **Recent runs drawer** — list from `/api/runs`; “Replay” (instant or 4×) so a finished
   run can be shown without recomputing (safety net for a live demo).

Layout: max-width 1200 px, 24-32 px gutters, works down to ~900 px wide.

---

## 5. Acceptance (the executor reports each one with numbers)

- **A1** `git diff scripts/training_dataset/generate_training_dataset.py` = only the two flags.
- **A2** Full live run on `…/Totalsegmentator_dataset_v201/s0006` completes; per-stage timings.
- **A3 Parity (G-buffers):** our `s0006` seg/depth/normals/segid vs `data/training_dataset_v25/s0006`
  for all 20 views — mean abs diff and % differing pixels per buffer; `meta.json` camera_pos
  diff. Expect ~identical. If not, **report — do not tweak camera or render code.**
- **A4 Parity (network):** output from our buffers vs output from the reference buffers, 2 views, PSNR between them.
- **A5 Sanity:** PSNR of each output vs `rgb_preview.png`, 20 views (smoke value only, not a thesis number).
- **A6** Replay of the finished run over SSE (`curl -N`).
- **A7** Error path: a folder without `segmentations/` → clean `fail` check, no crash.
- **A8 UI:** in the browser pane — full live run; stepper, timers, count-ups, mesh viewer,
  orbit, stack, develop animation, gallery, lightbox (all tabs, arrows, Esc), replay;
  no console errors; screenshots of each section.

## 6. Execution order
1. Backend (config, pipeline, viz, server, generator flags) + A1-A7, using a throw-away
   `static/index.html` that just prints events.
2. Frontend against the real finished s0006 run (replay), then a live run → A8.
3. README + `run_demo.sh`.

---

## 7. As built (backend, verified 2026-09-22) — the frontend's source of truth

**Memory incident.** The first backend attempt froze the laptop (16 GB): several demo
servers ran at once, each holding the model and each starting a run with up to 6 buffer
workers (~1.5 GB each). Now enforced in code:
- one server per machine (`runs/.server.lock`, flock); one run per machine (`runs/.run.lock`);
- a run is refused below 3 GB free (`config.MIN_FREE_GB`); `POST /api/run` → 409 with a message;
- the `buffers` stage runs as its own subprocess (`demo/buffer_worker.py`), 1-2 workers by free memory;
- the checkpoint is `mmap`-loaded (only G is paged in).
Measured on the reference run: minimum free memory 3.0 GB.

**Reference run:** `demo/runs/20260922-165227_s0006` (s0006, test split) — 55.6 s total:
ct 0.0 (cached inspect) · masks 0.8 · mesh 13.4 · gbuffer 14.1 · buffers 18.0 · infer 9.4 s.

**Acceptance results.**
- A1 generator diff = the two opt-in flags only.
- A3 G-buffer parity vs server-rendered s0006, 20 views: camera_pos identical; segid differs
  on ≤ 0.001 % px; seg.png mean |Δ| 0.01/255; normals mean |Δ| 3.5e-4; depth mean |Δ| 0.54 mm
  (= half-float step), large Δ only on silhouette pixels.
- A4 network parity (4 views): live-pipeline output vs reference-buffer output 41.9-53.6 dB;
  vs GT the two differ by ≤ 0.15 dB (e.g. v02 34.29 vs 34.44 dB).
- A5 output vs `rgb_preview.png`: mean 33.7 dB (29.7-37.8) over 20 views — smoke value only.
- A6 replay streams all events. A7 no-masks CT → `fail` checks, `/api/run` refuses (400).

**Events actually emitted** (keys/fields the UI can rely on):
```
run_start   {run_id, subject, ct_path, split:null}          # split arrives as ct stat
ct      stat: shape[3] voxels · spacing[3] mm · extent[3] cm · hu_range[2] HU · orientation · split · load_time s
        artifact: axial|coronal|sagittal (png 512) · axial_stack (jpg sprite, meta {n:64,w:256,h:256}, vertical strip) · info (json)
        log: "[pass|warn|fail] <check>: <reason>"
masks   progress 1..29 (label = tissue) · artifact volumes (json) · overlay_axial|coronal|sagittal · overlay_stack (same sprite layout)
        stat: found "29/29" · total_volume L
mesh    per organ: artifact organ, meta {name, color, verts, tris, ms, web (url of .bin), web_verts, web_tris, bbox_min[3], bbox_max[3]}
        (mm, relative to the CT volume centre) + progress; log "<name>: empty, skipped"; artifact manifest (json)
        stat: organs · verts · tris
gbuffer stat mm_per_px (early) · progress done 0..20 (label = view) · per view artifact seg|depth|normals|segid (png 512, view=…)
        + artifact cameras (json, re-emitted after every view) · stat views · s_per_view · resolution
buffers log "20 views on N worker(s)" · per view artifact albedo|nmap|rough|glint|highlight (png 512) + progress
        stat channels 17 · texture_ms · glint_ms
infer   per view artifact output (png 1024) · thumb (jpg 320) · gt (png 1024, only when alignment passed) + progress
        stat ms_per_view · device · params (M)
run_done {seconds_total, timings{stage: s}}
error    {stage, message, hint}
```
JSON shapes:
- `ct/info.json` `{sid, ct_path, subject_root, shape, spacing_mm, extent_cm, dtype, orientation, hu_min, hu_p1, hu_p99, hu_max}`
- `masks/volumes.json` `{tissue: {present, volume_ml, color}}` — **volume_ml < 1 means empty** (e.g. lungs outside the FOV); show only non-empty.
- `web/manifest.json` `{center_mm[3], organs:[<same as organ meta>]}`
- `gbuffers/cameras.json` `{cameras:[{view, cam_pos[3] m, target[3] m, az, el}], organ_center[3] m, radius m}`
- web mesh `.bin`: little-endian `u32 nV, u32 nI, f32 pos[3nV] (mm, centred), f32 nrm[3nV], u32 idx[nI]`.
View names: `vNN_az{+A}_el{+E}`, NN 01-20, az ∈ {0,72,144,216,288}, el ∈ {-50,0,30,60}.
