"""
buffer_worker.py — the `buffers` stage, run as its own subprocess.

    venv/bin/python demo/buffer_worker.py <jobs.json> --workers 2

pipeline.py launches this like it launches Blender and reads one JSON line per finished
view from stdout. It is a separate process on purpose: on macOS every pool worker
re-imports the launching script, so a pool started inside the server would pull the
whole server (pyvista/VTK, nibabel, the model holder) into every worker. Here each
worker imports only what `infer.load_input` needs (~1 GB peak per worker, measured),
and all of it is returned to the OS the moment the stage ends.
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRAINING = _HERE.parent / "scripts" / "training"
# scripts/training must come BEFORE demo/: both have a config.py, and the worker needs
# the training one (preset). demo/ stays on the path for viz.
sys.path.insert(0, str(_TRAINING))
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))


def buffer_worker(job):
    view_dir, subject_dir, stack_path, preview_dir, preset_name, tex_dir = job

    import numpy as np
    import viz
    from config import preset
    import infer
    import glint_model

    gcfg = preset(preset_name)
    gcfg.data.tex_dir = tex_dir

    timing = {"texture_ms": 0.0, "glint_ms": 0.0}
    orig_tex, orig_glint = infer.texture_buffer, glint_model.glint_buffer

    def timed(fn, key):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            timing[key] = (time.perf_counter() - t0) * 1000.0
            return r
        return wrapper

    # infer.load_input calls both through these module attributes, so wrapping them
    # times the real calls without touching the thesis code.
    infer.texture_buffer = timed(orig_tex, "texture_ms")
    glint_model.glint_buffer = timed(orig_glint, "glint_ms")
    try:
        lut = infer.build_segid_lut(Path(subject_dir))
        x = infer.load_input(Path(view_dir), gcfg, lut)
    finally:
        infer.texture_buffer, glint_model.glint_buffer = orig_tex, orig_glint
    np.save(stack_path, x.astype(np.float16))

    preview_dir = Path(preview_dir)
    groups = {"albedo": x[..., 8:11], "nmap": x[..., 11:14], "rough": x[..., 14:15],
              "glint": x[..., 15:16], "highlight": x[..., 16:17]}
    for name, chan in groups.items():
        viz.save_png(preview_dir / f"{name}.png", viz.resize_rgb(viz.channel_group_to_rgb(chan), 512))
    return timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help="JSON list of [view_dir, subject_dir, stack_path, preview_dir, preset, tex_dir]")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())

    def report(view, **kw):
        print(json.dumps({"view": view, **kw}), flush=True)

    if args.workers <= 1:
        for job in jobs:
            view = Path(job[0]).name
            try:
                report(view, **buffer_worker(job))
            except Exception as e:
                report(view, error=f"{type(e).__name__}: {e}")
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(buffer_worker, job): Path(job[0]).name for job in jobs}
        for fut in as_completed(futs):
            try:
                report(futs[fut], **fut.result())
            except Exception as e:
                report(futs[fut], error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
