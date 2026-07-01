"""
infer.py — Run the trained neural renderer on input G-buffers to produce a photoreal
image, WITHOUT the (expensive) Cycles ground truth. This is the deployment path:
for a NEW subject you render only the CHEAP inputs (flat EEVEE seg + depth/normals/segid
geometry passes — no path tracing), then get path-traced-quality output in milliseconds.

    # one view
    python scripts/training/infer.py --preset full1024 \
        --checkpoint results/training_runs/full1024/checkpoints/epoch_060.pt \
        --view data/training_dataset/s0000/v05_az+72_el+0 --out results/infer

    # every view of a subject
    python scripts/training/infer.py --preset full1024 \
        --checkpoint results/training_runs/full1024/checkpoints/epoch_060.pt \
        --subject data/training_dataset/s0000 --out results/infer

Needs per view: seg.png, depth.exr, normals.exr, segid.exr, meta.json (+ the subject's
tissue_ids.json). Does NOT need rgb_preview.png / render.exr (the GT).
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import preset
from networks import define_G
from exr import read_render_exr
from dataset import ORGAN_TO_GID, NUM_CANONICAL, _resize


def build_segid_lut(subject_dir: Path) -> np.ndarray:
    tissues = {}
    p = subject_dir / "tissue_ids.json"
    if p.exists():
        tissues = json.load(open(p)).get("tissues", {})
    max_pid = max(tissues.values(), default=0)
    lut = np.zeros(max_pid + 1, dtype=np.float32)
    for name, pid in tissues.items():
        lut[int(pid)] = ORGAN_TO_GID.get(name, 0)
    return lut


def load_input(view_dir: Path, cfg, lut: np.ndarray) -> np.ndarray:
    """Build the normalised input stack (HxWxC in [-1,1]) from the cheap G-buffers."""
    meta = json.load(open(view_dir / "meta.json")) if (view_dir / "meta.json").exists() else {}
    planes = []
    for b in cfg.data.input_buffers:
        if b == "seg_rgb":
            img = cv2.imread(str(view_dir / "seg.png"), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            planes.append(img * 2.0 - 1.0)
        elif b == "depth":
            d = read_render_exr(str(view_dir / "depth.exr"), want=("image",))["image"][..., 0]
            ref = float(meta.get("depth_metric_ref", 1.0)) or 1.0
            planes.append((np.clip(d / ref, 0, 1)[..., None] * 2.0 - 1.0).astype(np.float32))
        elif b == "normals":
            n = read_render_exr(str(view_dir / "normals.exr"), want=("image",))["image"][..., :3]
            planes.append(np.clip(n * 2.0 - 1.0, -1, 1).astype(np.float32))
        elif b == "segid":
            pid = np.rint(read_render_exr(str(view_dir / "segid.exr"), want=("image",))["image"][..., 0]).astype(np.int64)
            pid = np.clip(pid, 0, len(lut) - 1)
            gid = lut[pid] / float(NUM_CANONICAL)
            planes.append((np.clip(gid, 0, 1)[..., None] * 2.0 - 1.0).astype(np.float32))
        else:
            raise ValueError(f"unknown input buffer '{b}'")
    planes = [_resize(p, cfg.data.size) for p in planes]
    return np.concatenate(planes, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full1024", choices=["proto512", "full1024", "rgb_only", "overfit"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--view", default=None, help="a single view dir")
    ap.add_argument("--subject", default=None, help="a subject dir (renders all its views)")
    ap.add_argument("--out", default="results/infer")
    args = ap.parse_args()

    cfg = preset(args.preset)
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = define_G(cfg).to(device).eval()
    G.load_state_dict(torch.load(args.checkpoint, map_location=device)["G"])
    print(f"[infer] {args.checkpoint} | inputs={cfg.data.input_buffers} | device={device}")

    if args.view:
        views = [Path(args.view)]
    elif args.subject:
        views = sorted(p for p in Path(args.subject).glob("v*") if (p / "seg.png").exists())
    else:
        ap.error("give --view or --subject")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    times = []
    for vdir in views:
        lut = build_segid_lut(vdir.parent)
        x = load_input(vdir, cfg, lut)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fake = G(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        img = ((fake[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8).transpose(1, 2, 0)
        name = f"{vdir.parent.name}_{vdir.name}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[infer] {name}  ({dt:.1f} ms)")

    if times:
        t = np.array(times[1:] or times)  # drop first (warm-up) if >1
        print(f"\n[infer] {len(times)} views → {out}")
        print(f"[infer] inference: {t.mean():.1f} ms/view (vs minutes/view of path tracing)")


if __name__ == "__main__":
    main()
