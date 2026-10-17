# CT → Photoreal — local demo

A local web app that walks one CT scan through the whole Stage-2 pipeline, live:
CT → 29 organ segmentation masks → watertight meshes → cheap Blender G-buffers →
procedural texture/glint buffers → the trained pix2pixHD-style generator →
20 photoreal views. Runs entirely offline, no frameworks, no build step.

## Launch

The server holds the ~2.1 GB checkpoint in memory, so **only one instance should ever
run at a time** (it refuses to start a second copy via a flock on `demo/runs/.server.lock`).

```bash
./demo/run_demo.sh          # = venv/bin/python demo/server.py --open
```

This starts `ThreadingHTTPServer` on `http://127.0.0.1:8765/` and opens it in your
browser. Frontend files (`demo/static/**`) are served straight from disk with
`Cache-Control: no-cache`, so editing `index.html` / `app.css` / `js/*.js` and
reloading the page picks up changes immediately — no restart needed. A restart is
only needed after editing `server.py`, `pipeline.py`, `viz.py`, `config.py`, or
`buffer_worker.py`.

To restart after a backend change:
```bash
kill $(lsof -tiTCP:8765 -sTCP:LISTEN)
venv/bin/python demo/server.py &        # one background start; it refuses to start twice
```

## Using it

1. **Choose a CT** — type a path, use the quick-pick chips (held-out test subjects
   `s0006 s0005 s0018 s0020 s0055`), click **Browse…** for the native macOS file
   picker, or **Browse in app** for an in-page folder browser. Choosing a CT calls
   `POST /api/inspect`, which loads the volume, runs the pass/warn/fail checks, and
   renders three orthogonal slices plus an axial scrubber (with a "show masks" toggle
   once segmentation is available).
2. **Run pipeline** — disabled until every check passes and the model finishes
   loading (`/api/status` → `ready`). Starts a run (`POST /api/run`) and streams its
   events over SSE, driving every card below live.
3. **Pipeline** — one card per stage (`masks`, `mesh`, `gbuffer`, `buffers`, `infer`;
   the CT stage's own card is the "Choose a CT" card above). Each card shows an
   input → output row, a live `mm:ss.s` timer while running, count-up stat tiles, and
   a collapsible **Details** section (per-organ tables, raw log lines). A "follow
   along" toggle auto-scrolls to whichever stage is currently running.
   - *Segmentation masks*: CT slice → colour overlay, organ chips with volume bars.
   - *Mesh extraction*: overlay → a small hand-written WebGL viewer (drag to orbit,
     scroll to zoom, double-click to resume auto-rotate) — organs pop in one by one
     as `extract_meshes` finishes each one, and the camera smoothly reframes the
     union of everything loaded so far.
   - *G-buffer rendering*: a canvas-2D "orbit diagram" schematic of the 20 camera
     positions (5 azimuths × 4 elevations) that fill in as Blender finishes each
     view, next to a crossfading 2×2 preview (seg / depth / normals / seg-id) of the
     latest view and a filmstrip of finished views.
   - *Texture & glint buffers*: seg thumbnail → albedo / normal-map / roughness /
     vessel-glint / highlight tiles, plus the 17-channel network input visualised as
     a fanned stack of 9 translucent cards (widens on hover).
   - *Neural rendering*: the output "develops" in (blur → sharp) as each view's PNG
     arrives; stats are ms/view, device, parameter count.
4. **Results** — the hero render (defaults to `v02`, the first straight-on view).
   Click it (or any gallery tile) for the **lightbox**: *Neural render*, *Inputs*
   (all 9 preview channels for that view), and *Compare with Cycles* (a drag slider
   against the ground-truth render — hidden automatically when no GT passed the
   backend's alignment guard for that view). ← / → cycle through rendered views,
   Esc closes. Below it, the **orbit gallery** — 4 rows (el −50/0/30/60°) × 5 columns
   (az 0/72/144/216/288°) — with a segmented control to swap the whole grid between
   Output / Seg / Normals / Glints.
5. **Summary** — a stacked bar of the six stage timings, total wall-clock, this run's
   neural ms/view, and a clearly-labelled *server* reference line (Cycles at 53 s/view
   and this network at 98.3 ms/view on an L4, both measured elsewhere — never
   confused with this laptop's own numbers).
6. **Recent runs** (top-right) — a drawer listing past runs from `/api/runs`, each
   with **Replay instant / 1× / 4×** (`?replay=1&speed=N`), so the whole UI can be
   demonstrated from a finished run without recomputing anything. The empty state on
   first load offers a **Load last run** shortcut that does the same thing.

## Files

```
demo/static/index.html        page structure
demo/static/css/app.css       palette, type, motion, layout (see PLAN.md §4)
demo/static/js/app.js         state, SSE client, stage cards, results, lightbox
demo/static/js/meshviewer.js  vanilla WebGL organ mesh viewer (no three.js)
demo/static/js/orbit.js       canvas-2D camera-orbit schematic
demo/static/js/ui.js          count-up numbers, timers, small DOM helpers
```

All ES modules, no CDN, no bundler. The event contract these files rely on is
documented in `demo/PLAN.md` §7 (the as-built source of truth — §3.1 is the original
design and may differ in small ways).

## Memory safety

This was built and must be run on a 16 GB laptop that has previously been frozen by
running several heavy processes at once. The backend enforces:

- **One server per machine** — `demo/runs/.server.lock` (flock); a second `server.py`
  refuses to start and prints the port to use instead.
- **One run per machine** — `demo/runs/.run.lock`; `POST /api/run` returns **409** if
  another run is already in progress, or if free memory is below `config.MIN_FREE_GB`
  (3 GB).
- The `buffers` stage runs in its **own subprocess** (`demo/buffer_worker.py`) with
  1–2 workers sized by free memory, so its ~1 GB/worker footprint is released the
  moment the stage finishes.
- The checkpoint is **mmap-loaded**, so only the generator's tensors (not the
  optimiser states or discriminator also saved in the same file) are paged in.

Practical implications for anyone driving this UI:
- Don't start a second `server.py` — it will refuse, but don't rely on that; check
  first.
- Don't click **Run pipeline** twice — the button disables itself immediately and the
  backend lock would refuse the second call anyway, but a live run should only ever
  be started once at a time by hand too.
- Prefer **Replay** (instant/1×/4×) over live runs while iterating on the frontend —
  it exercises the exact same UI code path against a real finished run's
  `events.jsonl`, with zero compute cost.
- A live run takes roughly 1–3 minutes depending on machine load (see the timings in
  the Summary card of whichever run you last did) — mesh extraction, Blender
  G-buffer rendering, and the buffer worker are the expensive stages; inference
  itself is a few hundred ms per view.
