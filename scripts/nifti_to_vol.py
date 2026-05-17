"""
Convert a preprocessed CT NIfTI volume to Mitsuba 3 .vol files for path tracing.

Produces two volumes:
  ct_density.vol  — extinction coefficient density [0,1], bone-focused mapping
  ct_albedo.vol   — spatially-varying single-scatter albedo [0,1]

Mitsuba axis ordering note:
  VolumeGrid(TensorXf(data)) expects data shaped (nz_mi, ny_mi, nx_mi, ch)
  where VolumeGrid.size() returns [nx_mi, ny_mi, nz_mi].
  NIfTI loaded as (nx, ny, nz) → transpose to (nz, ny, nx) before passing to Mitsuba.

Usage:
    python scripts/nifti_to_vol.py \
        --volume "data/processed/CQ500CT95 CQ500CT95.nii.gz" \
        --output data/renders/gt/CQ500CT95/
"""

import argparse
from pathlib import Path

import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import nibabel as nib
import numpy as np


# ── Transfer function: HU → extinction density ──────────────────────────────

def hu_to_density(hu: np.ndarray) -> np.ndarray:
    """
    Bone-focused 1D transfer function: HU → extinction density in [0, 1].
    Air and soft tissue are nearly transparent; dense bone is fully opaque.
    Control points calibrated to skull CT range:
      HU <= 200  → 0.0  (air, soft tissue: transparent)
      HU = 700   → 0.30 (trabecular bone: semi-transparent)
      HU = 1000  → 0.60 (dense bone)
      HU >= 1500 → 1.0  (cortical/dense bone: fully opaque)
    """
    ctrl_hu  = np.array([-1000, 200,  700,  1000, 1500, 3000], dtype=np.float32)
    ctrl_den = np.array([0.00,  0.00, 0.30, 0.60, 1.00, 1.00], dtype=np.float32)
    return np.interp(hu, ctrl_hu, ctrl_den).astype(np.float32)


# ── Transfer function: HU → single-scatter albedo ───────────────────────────

def hu_to_albedo(hu: np.ndarray) -> np.ndarray:
    """
    Single-scatter albedo [0, 1]:
      air          → 0.0  (no scattering)
      soft tissue  → 0.85 (high albedo — translucent)
      bone         → 0.60 (moderate — more absorbing)
    Albedo controls how much light scatters vs is absorbed at each point.
    """
    ctrl_hu  = np.array([-1000, -200, 0,    200,  700,  3000], dtype=np.float32)
    ctrl_alb = np.array([0.0,   0.0,  0.85, 0.85, 0.65, 0.60], dtype=np.float32)
    return np.interp(hu, ctrl_hu, ctrl_alb).astype(np.float32)


# ── Write helper ─────────────────────────────────────────────────────────────

def write_vol(path: Path, data_nxyz: np.ndarray):
    """
    Write a single-channel float32 Mitsuba .vol file.
    data_nxyz: shape (nx, ny, nz) in NIfTI/PyVista world convention.
    Mitsuba wants shape (nz, ny, nx, 1) for TensorXf → VolumeGrid.
    """
    nz_mi = data_nxyz.transpose(2, 1, 0).astype(np.float32)   # (nz, ny, nx)
    tensor = mi.TensorXf(nz_mi[..., np.newaxis])               # (nz, ny, nx, 1)
    grid   = mi.VolumeGrid(tensor)
    grid.write(str(path))
    nx, ny, nz = data_nxyz.shape
    print(f"  Written: {path.name}  VolumeGrid.size()={grid.size()}  "
          f"(nx={nx}, ny={ny}, nz={nz})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIfTI → Mitsuba .vol conversion")
    parser.add_argument("--volume", required=True, help="Path to preprocessed .nii.gz")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    vol_path = Path(args.volume)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {vol_path.name} …")
    img    = nib.load(str(vol_path))
    hu     = img.get_fdata(dtype=np.float32)   # shape (nx, ny, nz)
    zooms  = img.header.get_zooms()[:3]
    nx, ny, nz = hu.shape

    print(f"  Shape: {hu.shape}  Spacing: {tuple(float(z) for z in zooms)} mm")
    print(f"  HU range: [{hu.min():.0f}, {hu.max():.0f}]  "
          f"mean: {hu.mean():.1f}  std: {hu.std():.1f}")

    # Bone statistics (Siemens-style windowing reference)
    bone = hu[hu > 200]
    print(f"  Bone voxels (>200 HU): {len(bone):,}  "
          f"mean: {bone.mean():.1f}  std: {bone.std():.1f}")

    # Bounding box in mm (matches PyVista world space: origin=0, spacing=zooms)
    xmax, ymax, zmax = nx * zooms[0], ny * zooms[1], nz * zooms[2]
    print(f"  World AABB: [0,{xmax:.1f}] x [0,{ymax:.1f}] x [0,{zmax:.1f}] mm")

    print("Computing density volume …")
    density = hu_to_density(hu)
    print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]  "
          f"mean: {density.mean():.4f}")

    print("Computing albedo volume …")
    albedo = hu_to_albedo(hu)
    print(f"  Albedo range: [{albedo.min():.4f}, {albedo.max():.4f}]  "
          f"mean: {albedo.mean():.4f}")

    print("Writing .vol files …")
    stem = vol_path.stem.split('.')[0]  # e.g. "CQ500CT95 CQ500CT95"
    write_vol(out_dir / f"{stem}_density.vol", density)
    write_vol(out_dir / f"{stem}_albedo.vol",  albedo)

    # Save camera params for this volume (used by render_mitsuba_gt.py)
    dims_pv = (nx + 1, ny + 1, nz + 1)   # PyVista cell-centred dims
    cx = dims_pv[0] * float(zooms[0]) / 2
    cy = dims_pv[1] * float(zooms[1]) / 2
    cz = dims_pv[2] * float(zooms[2]) / 2
    radius = max(dims_pv) * max(zooms) * 0.9
    cam_pos   = (cx + radius * 0.6, cy - radius * 1.1, cz + radius * 0.5)
    cam_focal = (cx, cy, cz)

    import json
    meta = {
        "volume": str(vol_path),
        "shape_nxyz": [int(nx), int(ny), int(nz)],
        "spacing_mm": [float(z) for z in zooms],
        "aabb_mm": [0.0, 0.0, 0.0, float(xmax), float(ymax), float(zmax)],
        "camera": {
            "position":    [float(v) for v in cam_pos],
            "focal_point": [float(v) for v in cam_focal],
            "up":          [0.0, 0.0, 1.0],
            "fov_deg":     30.0,
        },
        "tf_notes": (
            "Bone-focused 1D TF. Density: piecewise linear, HU<=200->0, HU=700->0.30, "
            "HU=1000->0.60, HU>=1500->1.0. Albedo: air=0, soft tissue=0.85, bone=0.60-0.65. "
            "Siemens windowing NOT yet applied (waiting for LUT files from Simon). "
            "Current TF is empirical starting point."
        )
    }
    meta_path = out_dir / f"{stem}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Files in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
