"""
eval_fid_kid.py — distribution-level realism (FID / KID)

PSNR, SSIM and LPIPS are all PAIRED: they ask "how close is this output to its own
ground truth". They cannot answer "do the outputs look like organ renders at all",
because a systematically desaturated or over-smoothed model can still score well on
every paired metric. FID and KID compare DISTRIBUTIONS of Inception features, so they
catch exactly that failure.

KID is reported alongside FID because FID is biased at small sample sizes — its
estimator has a bias that shrinks with n, so an FID computed over a few hundred test
views is not comparable to one over thousands. KID is unbiased, which matters here
where the test split is modest.

Reproducibility note: published FID numbers are notoriously hard to reproduce because
implementations differ in resizing, normalisation and Inception weights. This uses
torch-fidelity when available (the de-facto reference) and falls back to a documented
in-repo implementation otherwise; the JSON records which was used, so the number is
never silently from a different definition.

    pip install torch-fidelity        # recommended

Usage
-----
    python scripts/evaluation/eval_fid_kid.py \
        --checkpoint results/training_runs/full1024/checkpoints/latest.pt \
        --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from eval_common import (add_common_args, build_cfg, pick_device, load_generator,
                         open_dataset, to01, sample_indices, ensure_out)


def dump_pairs(G, ds, device, idxs, root):
    """Write generated and GT images as PNGs — the input format FID tools expect."""
    from PIL import Image
    fake_dir = root / "fake"
    real_dir = root / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for n, i in enumerate(idxs):
            b = ds[i]
            x = b["input"].unsqueeze(0).to(device)
            fake = G(x)[0]
            real = b["target"]
            Image.fromarray((np.clip(to01(fake), 0, 1) * 255).astype(np.uint8)).save(
                fake_dir / f"{n:05d}.png")
            Image.fromarray((np.clip(to01(real), 0, 1) * 255).astype(np.uint8)).save(
                real_dir / f"{n:05d}.png")
            if n % 50 == 0:
                print(f"    {n}/{len(idxs)}", end="\r", flush=True)
    print(f"    wrote {len(idxs)} pairs        ")
    return fake_dir, real_dir


def via_torch_fidelity(fake_dir, real_dir, kid_subset_size):
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(
        input1=str(fake_dir), input2=str(real_dir),
        fid=True, kid=True, kid_subset_size=kid_subset_size,
        verbose=False)
    return {
        "implementation": "torch-fidelity",
        "fid": float(m["frechet_inception_distance"]),
        "kid_mean": float(m["kernel_inception_distance_mean"]),
        "kid_std": float(m["kernel_inception_distance_std"]),
        "kid_subset_size": kid_subset_size,
    }


def _inception_features(paths, device, batch=32):
    """Pool3 features from torchvision Inception-v3, resized to 299 with bilinear."""
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import inception_v3, Inception_V3_Weights

    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                       transform_input=False).to(device).eval()
    net.fc = torch.nn.Identity()          # 2048-d pool3 features
    tf = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = torch.stack([tf(Image.open(p).convert("RGB"))
                               for p in paths[i:i + batch]]).to(device)
            feats.append(net(ims).cpu().numpy())
    return np.concatenate(feats, 0)


def _fid_from_feats(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    d = mu1 - mu2
    return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _kid_from_feats(a, b, subset_size, n_subsets=100, seed=0):
    """Unbiased MMD^2 with a degree-3 polynomial kernel (Binkowski et al.)."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    n = min(subset_size, len(a), len(b))
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), n, replace=False)]
        y = b[rng.choice(len(b), n, replace=False)]
        kxx = (x.dot(x.T) / d + 1) ** 3
        kyy = (y.dot(y.T) / d + 1) ** 3
        kxy = (x.dot(y.T) / d + 1) ** 3
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (n * (n - 1)) + kyy.sum() / (n * (n - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def via_fallback(fake_dir, real_dir, device, kid_subset_size):
    fa = sorted(fake_dir.glob("*.png"))
    re_ = sorted(real_dir.glob("*.png"))
    f1 = _inception_features(fa, device)
    f2 = _inception_features(re_, device)
    kid_m, kid_s = _kid_from_feats(f1, f2, kid_subset_size)
    return {
        "implementation": "in-repo fallback (torchvision Inception-v3, bilinear 299)",
        "fid": _fid_from_feats(f1, f2),
        "kid_mean": kid_m,
        "kid_std": kid_s,
        "kid_subset_size": kid_subset_size,
        "note": "NOT numerically comparable to published torch-fidelity/TF numbers; "
                "use only for comparing models evaluated by this same code path.",
    }


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--kid-subset-size", type=int, default=100)
    ap.add_argument("--keep-images", action="store_true",
                    help="keep the dumped PNGs instead of deleting them")
    args = ap.parse_args()

    out = ensure_out(args.out)
    device = pick_device(args.device)
    cfg = build_cfg(args)
    print(f"eval_fid_kid — device={device}")
    G = load_generator(cfg, args.checkpoint, device)
    ds = open_dataset(cfg, args.split)
    idxs = sample_indices(len(ds), args.max_samples)

    work = Path(tempfile.mkdtemp(prefix="fidkid_")) if not args.keep_images \
        else ensure_out(out / "fid_images")
    try:
        print(f"  rendering {len(idxs)} pairs ...")
        fake_dir, real_dir = dump_pairs(G, ds, device, idxs, work)

        if len(idxs) < 2 * args.kid_subset_size:
            print(f"  [warn] n={len(idxs)} is small for KID subset {args.kid_subset_size}; "
                  f"lowering it")
            args.kid_subset_size = max(10, len(idxs) // 2)

        try:
            res = via_torch_fidelity(fake_dir, real_dir, args.kid_subset_size)
        except ImportError:
            print("  torch-fidelity not installed — using the in-repo fallback.")
            print("  For numbers comparable with the literature: pip install torch-fidelity")
            res = via_fallback(fake_dir, real_dir, device, args.kid_subset_size)

        res.update({"n_images": len(idxs), "split": args.split,
                    "checkpoint": args.checkpoint, "target": cfg.data.target})
        print(f"\n  FID  {res['fid']:.3f}")
        print(f"  KID  {res['kid_mean']:.5f} +/- {res['kid_std']:.5f}")
        if len(idxs) < 500:
            print(f"  [note] FID is biased at small n (here {len(idxs)}). Quote KID as "
                  f"the primary distribution metric, and never compare FID across runs "
                  f"with different n.")
        f = out / "fid_kid.json"
        f.write_text(json.dumps(res, indent=2))
        print(f"wrote {f}")
    finally:
        if not args.keep_images:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
