"""
Visualise a CT volume: 3 orthogonal slices + HU histogram + metadata.

Usage:
    python scripts/visualise_volume.py data/raw/cq500_nifti/CQ500-CT-0.nii.gz
    python scripts/visualise_volume.py data/raw/totalsegmentator/s0001/ct.nii.gz
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/visualise_volume.py <path/to/volume.nii.gz>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path.name} ...")
    img  = nib.load(str(path))
    # extracts just the 3D numpy array of HU values — this is the (512, 512, 36) grid
    data = img.get_fdata(dtype=np.float32)
    # pulls the spacing values from the header — the (0.564, 0.564, 5.176) mm per voxel
    spacing = img.header.get_zooms()[:3]

    print(f"  Shape  : {data.shape}")
    print(f"  Spacing: {tuple(f'{s:.3f}mm' for s in spacing)}")
    print(f"  HU min : {data.min():.1f}  max: {data.max():.1f}  mean: {data.mean():.1f}")

    # just get middle index
    cx, cy, cz = (s // 2 for s in data.shape)
    vmin, vmax = np.percentile(data, 1), np.percentile(data, 99)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(path.name, fontsize=11)

    for ax, (sl, title) in zip(axes[:3], [
        (data[cx, :, :], f"Sagittal  x={cx}"),
        (data[:, cy, :], f"Coronal   y={cy}"),
        (data[:, :, cz], f"Axial     z={cz}"),
    ]):
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    flat = data.flatten()
    axes[3].hist(flat[(flat > -1100) & (flat < 3100)], bins=200,
                 color="steelblue", alpha=0.8, log=True)
    axes[3].axvline(-200, color="cyan",   linestyle="--", linewidth=0.8, label="air/tissue")
    axes[3].axvline(300,  color="orange", linestyle="--", linewidth=0.8, label="bone start")
    axes[3].set_xlabel("HU"); axes[3].set_ylabel("Count (log)")
    axes[3].set_title("HU histogram"); axes[3].legend(fontsize=7)

    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / f"{path.stem}_preview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
