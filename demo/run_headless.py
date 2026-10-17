#!/usr/bin/env python
"""
run_headless.py — run one subject through the demo pipeline without the web UI.

Same code path as the browser demo (pipeline.Run), so the per-stage timings are
comparable; it just has no SSE client attached. Written for the GPU server, where
there is no browser and the generator should run on CUDA.

    python demo/run_headless.py --ct /path/to/sXXXX/ct.nii.gz
    python demo/run_headless.py --ct ... --repeat 3        # 3 runs, timings of each

Environment overrides (see demo/config.py): DEMO_CKPT, DEMO_TEX_DIR, DEMO_BLENDER,
DEMO_GT_ROOTS, DEMO_PRESET.

Writes the usual run directory under demo/runs/<run_id>/ (events.jsonl, run.json,
gbuffers/, output/, …) and prints a one-line timing summary per run. All runs are
also appended to demo/runs/headless_timings.csv, which is the file to copy back.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline               # noqa: E402  (also loads demo/config.py under a private name)

cfg = pipeline.cfg            # demo/config.py — never imported as bare `config`, see pipeline._load_demo_config

CSV_PATH = cfg.RUNS / "headless_timings.csv"
FIELDS = ["run_id", "subject", "device", "host", "ct", "masks", "mesh", "gbuffer",
          "buffers", "infer", "recon", "seconds_total", "views", "ms_per_view_infer",
          "s_per_view_gbuffer", "ms_per_view_texture", "ms_per_view_glint"]


def _per_view_stats(run_dir: Path) -> dict:
    """Pull the per-view numbers the stages emit, straight from events.jsonl."""
    out = {}
    f = run_dir / "events.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "stat":
            continue
        key = (e.get("stage"), e.get("key"))
        out[{("infer", "ms_per_view"): "ms_per_view_infer",
             ("gbuffer", "s_per_view"): "s_per_view_gbuffer",
             ("buffers", "texture_ms"): "ms_per_view_texture",
             ("buffers", "glint_ms"): "ms_per_view_glint",
             ("gbuffer", "views"): "views"}.get(key, "_")] = e.get("value")
    out.pop("_", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="path to ct.nii.gz (its folder must hold segmentations/)")
    ap.add_argument("--repeat", type=int, default=1, help="run N times; the first run is the cold one")
    args = ap.parse_args()

    import socket
    host = socket.gethostname()

    print(f"[headless] loading {cfg.CKPT}", flush=True)
    t0 = time.time()
    pipeline.MODEL.load()
    snap = pipeline.MODEL.snapshot()
    if snap["status"] != "ready":
        print(f"[headless] model failed to load:\n{snap['error']}", file=sys.stderr)
        return 1
    print(f"[headless] model ready on {snap['device']} "
          f"({snap['params_m']:.1f} M params) in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for i in range(args.repeat):
        try:
            run = pipeline.Run(args.ct)
        except pipeline.Busy as e:
            print(f"[headless] refused: {e}", file=sys.stderr)
            return 2
        print(f"[headless] run {i + 1}/{args.repeat}: {run.sid} -> {run.run_dir}", flush=True)
        run.run()
        meta = json.loads((run.run_dir / "run.json").read_text())
        if meta.get("status") != "done":
            print(f"[headless] run failed: {meta.get('error')}", file=sys.stderr)
            return 3
        row = {"run_id": meta["run_id"], "subject": meta["subject"],
               "device": str(snap["device"]), "host": host,
               "seconds_total": meta["seconds_total"], **meta["timings"],
               **_per_view_stats(run.run_dir)}
        rows.append(row)
        t = meta["timings"]
        print(f"[headless] {meta['run_id']}  total {meta['seconds_total']:.1f}s  "
              f"mesh {t.get('mesh', 0):.1f}  gbuffer {t.get('gbuffer', 0):.1f}  "
              f"buffers {t.get('buffers', 0):.1f}  infer {t.get('infer', 0):.1f}", flush=True)

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[headless] wrote {CSV_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
