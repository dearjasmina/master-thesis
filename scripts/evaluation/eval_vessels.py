"""
eval_vessels.py — does the model put the VESSELS where they are, on the test split?

PSNR/SSIM/LPIPS are dominated by large-scale shading and say nothing about the one feature
this thesis chased: the vessel glints. This scores exactly that, at scale, with the
statistic used throughout the pre-flights (thesis_context.md §4b):

    streak(Y)  = local-mean-removed |high-pass| of luminance — "thin bright detail here"
    ridge(P)   = the organ's normal-map vessel ridges, BOX-projected at the corrected
                 object-space position (texture_lookup + camera_model)
    M3         = corr(streak, ridge) inside eroded organ masks

M3 is computed for the GROUND TRUTH (the ceiling any model can reach on these views), for
the model, and for a half-tile-shifted null (a sanity floor: it must collapse). A model
that invents vein-like detail in the wrong places scores ~0 here, which is the point —
GAN texture that is not attached to the surface would flicker between views.

Per-subject values are bootstrapped over SUBJECTS, not views, so the CI is honest.

    python scripts/evaluation/eval_vessels.py --preset full1024_glint \
        --checkpoint results/training_runs/gbuf_glint/checkpoints/epoch_060.pt \
        --data-root /mnt/data/vulovic/data/training_dataset --split test \
        --subjects 30 --views-per-subject 4 --out EVALUATION/v25_gbuf_glint
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_common import (add_common_args, build_cfg, pick_device, load_generator,
                         open_dataset, view_dir_of, ensure_out, bootstrap_ci)

_TRAIN = Path(__file__).resolve().parents[1] / "training"
_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
for _p in (str(_TRAIN), str(_ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from texture_lookup import TextureBank            # noqa: E402
import preflight as PF                            # noqa: E402  (view_data/organ_masks/ridge_image/streak)
import texture_learnability as TL                 # noqa: E402  (highpass/erode/Corr/lum)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--subjects", type=int, default=30, help="test subjects to score")
    ap.add_argument("--views-per-subject", type=int, default=4)
    ap.add_argument("--organs", nargs="*", default=["liver", "kidney_left", "kidney_right",
                                                    "spleen", "heart", "small_bowel", "colon"])
    args = ap.parse_args()

    out = ensure_out(args.out)
    device = pick_device(args.device)
    cfg = build_cfg(args)
    G = load_generator(cfg, args.checkpoint, device)
    bank = TextureBank(cfg.data.tex_dir, cfg.data.tex_mm)
    ds = open_dataset(cfg, args.split)

    # group the split's views by subject, keep the first N views of N subjects
    by_subj = defaultdict(list)
    for i in range(len(ds)):
        vd = view_dir_of(ds, i)
        by_subj[vd.parent.name].append(vd)
    subs = sorted(by_subj)[: args.subjects]
    print(f"eval_vessels — {len(subs)} subjects x <= {args.views_per_subject} views, device={device}")

    per_subject = defaultdict(dict)          # subject -> {key: r}
    pooled = defaultdict(TL.Corr)            # (key, organ) -> Corr
    for s in subs:
        acc = defaultdict(TL.Corr)
        for vd in sorted(by_subj[s])[: args.views_per_subject]:
            V = PF.view_data(vd, bank)
            masks = PF.organ_masks(V, bank)
            if not masks:
                continue
            pred = PF.predict(G, cfg, vd, device)
            ridge = PF.ridge_image(V, bank, masks)
            null = PF.ridge_image(V, bank, masks, offset_mm=bank.tex_mm / 2)
            Yg, Yp = TL.lum(V["gt"]), TL.lum(pred)
            for organ, _, m, ev in masks:
                tr = TL.highpass(ridge, m, PF.SIG_BAND)[ev]
                tn = TL.highpass(null, m, PF.SIG_BAND)[ev]
                sg, sp = PF.streak(Yg, m)[ev], PF.streak(Yp, m)[ev]
                for key, (a, b) in (("gt", (tr, sg)), ("model", (tr, sp)),
                                    ("gt_null", (tn, sg)), ("model_null", (tn, sp))):
                    for o in ("all", organ):
                        if o == "all" or o in args.organs:
                            acc[(key, o)].add(a, b)
                            pooled[(key, o)].add(a, b)
        for k, c in acc.items():
            if c.n > 1000:
                per_subject[s]["|".join(k)] = c.r
        done = len(per_subject)
        if done % 5 == 0:
            print(f"  {done}/{len(subs)} subjects", flush=True)

    organs = ["all"] + [o for o in args.organs if ("gt", o) in pooled]
    print(f"\nM3 vessel registration (pooled over {len(per_subject)} subjects)")
    print(f"   {'organ':14}{'GT ceiling':>12}{'model':>10}{'% of GT':>9}{'GT null':>10}{'model null':>12}{'n px':>12}")
    rows = {}
    for o in organs:
        g, m_ = pooled[("gt", o)].r, pooled[("model", o)].r
        rows[o] = dict(gt=g, model=m_, pct=100 * m_ / g if g > 1e-9 else float("nan"),
                       gt_null=pooled[("gt_null", o)].r, model_null=pooled[("model_null", o)].r,
                       n=pooled[("gt", o)].n)
        r = rows[o]
        print(f"   {o:14}{r['gt']:>+12.3f}{r['model']:>+10.3f}{r['pct']:>8.0f}%"
              f"{r['gt_null']:>+10.3f}{r['model_null']:>+12.3f}{r['n']:>12}")

    # bootstrap over subjects (not views): the honest CI for a claim about the model
    boot = {}
    for key in ("gt|all", "model|all", "gt|liver", "model|liver"):
        vals = [v[key] for v in per_subject.values() if key in v]
        if len(vals) > 2:
            mean, lo, hi = bootstrap_ci(vals)
            boot[key] = dict(mean=mean, lo=lo, hi=hi, n_subjects=len(vals))
            print(f"   {key:12} per-subject mean {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"(n={len(vals)} subjects)")

    json.dump(dict(checkpoint=args.checkpoint, preset=args.preset, split=args.split,
                   subjects=list(per_subject), pooled=rows, bootstrap=boot),
              open(Path(out) / "vessels.json", "w"), indent=2)
    g, m_ = rows["all"]["gt"], rows["all"]["model"]
    print(f"\nVERDICT vessels: model reaches {100*m_/g if g>1e-9 else float('nan'):.0f}% of the GT's own "
          f"vessel registration ({m_:+.3f} vs {g:+.3f}); nulls {rows['all']['model_null']:+.3f} / "
          f"{rows['all']['gt_null']:+.3f} must be ~0")
    print(f"wrote {Path(out) / 'vessels.json'}")


if __name__ == "__main__":
    main()
