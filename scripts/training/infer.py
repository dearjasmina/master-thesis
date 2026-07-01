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


def render_view(G, vdir, cfg, lut, device):
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
    img = ((fake[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8).transpose(1, 2, 0)
    return img, dt


def montage(imgs, cols=5, th=256):
    tiles = [cv2.resize(i, (int(i.shape[1] * th / i.shape[0]), th)) for i in imgs]
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=15) for t in tiles]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full1024", choices=["proto512", "full1024", "rgb_only", "overfit"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input-buffers", default=None)
    ap.add_argument("--view", default=None, help="a single view dir")
    ap.add_argument("--subject", default=None, help="a subject dir (all its views)")
    ap.add_argument("--data-root", default="data/training_dataset", help="with --split")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="infer EVERY subject in this split (per-subject montage you can browse)")
    ap.add_argument("--out", default="results/infer")
    args = ap.parse_args()

    cfg = preset(args.preset)
    cfg.data.root = args.data_root
    if args.input_buffers:
        cfg.data.input_buffers = [b.strip() for b in args.input_buffers.split(",") if b.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = define_G(cfg).to(device).eval()
    G.load_state_dict(torch.load(args.checkpoint, map_location=device)["G"])
    print(f"[infer] {args.checkpoint} | inputs={cfg.data.input_buffers} | device={device}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "montages").mkdir(exist_ok=True)

    # Build the list of subject dirs to process.
    if args.view:
        subj_dirs = None
        single_views = [Path(args.view)]
    elif args.subject:
        subj_dirs = [Path(args.subject)]
    elif args.split:
        from splits import assign_splits
        root = Path(args.data_root)
        subs = [p.name for p in sorted(root.iterdir()) if p.is_dir() and any(p.glob("v*/seg.png"))]
        sm = assign_splits(subs, cfg)
        subj_dirs = [root / s for s in subs if sm.get(s) == args.split]
        print(f"[infer] split={args.split}: {len(subj_dirs)} subjects")
    else:
        ap.error("give --view, --subject, or --split")

    times = []
    if args.view:
        img, dt = render_view(G, single_views[0], cfg, build_segid_lut(single_views[0].parent), device)
        cv2.imwrite(str(out / f"{single_views[0].parent.name}_{single_views[0].name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        times.append(dt)
    else:
        for i, sdir in enumerate(subj_dirs):
            lut = build_segid_lut(sdir)
            views = sorted(p for p in sdir.glob("v*") if (p / "seg.png").exists())
            strips = []
            for vdir in views:
                fake, dt = render_view(G, vdir, cfg, lut, device)
                times.append(dt)
                cv2.imwrite(str(out / f"{sdir.name}_{vdir.name}.png"), cv2.cvtColor(fake, cv2.COLOR_RGB2BGR))
                # comparison strip: seg (simple input) | fake (inferred) | GT (if present)
                h, w = fake.shape[:2]
                seg = cv2.cvtColor(cv2.imread(str(vdir / "seg.png")), cv2.COLOR_BGR2RGB)
                parts = [cv2.resize(seg, (w, h)), fake]
                gt_p = vdir / "rgb_preview.png"
                if gt_p.exists():
                    parts.append(cv2.resize(cv2.cvtColor(cv2.imread(str(gt_p)), cv2.COLOR_BGR2RGB), (w, h)))
                strips.append(np.concatenate(parts, axis=1))  # seg|fake|GT
            if strips:
                cv2.imwrite(str(out / "montages" / f"{sdir.name}.png"),
                            cv2.cvtColor(montage(strips, cols=1), cv2.COLOR_RGB2BGR))
            if (i + 1) % 10 == 0:
                print(f"[infer] {i+1}/{len(subj_dirs)} subjects")

    if times:
        t = np.array(times[1:] or times)  # drop warm-up
        print(f"\n[infer] {len(times)} views → {out}  (per-subject montages in {out/'montages'})")
        print(f"[infer] inference: {t.mean():.1f} ms/view (vs minutes/view of path tracing)")


if __name__ == "__main__":
    main()
