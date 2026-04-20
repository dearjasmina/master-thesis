"""
Preprocessing pipeline for CQ500 head CT volumes.

For each patient:
  1. Select the best series (most slices = thinnest = highest quality)
  2. Reorient to RAS+ canonical orientation
  3. Resample to 1mm³ isotropic voxels
  4. Clip HU values to [-1000, 3000]
  5. Save to data/processed/

Usage:
    python scripts/preprocess.py --input data/raw/cq500_nifti/ --output data/processed/
    python scripts/preprocess.py --input data/raw/cq500_nifti/ --output data/processed/ --limit 10
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import nibabel as nib
import numpy as np
import SimpleITK as sitk


# ── Step 1 helpers: series selection ─────────────────────────────────────────

def group_by_patient(nifti_dir: Path) -> dict[str, list[Path]]:
    """
    Group all NIfTI files by patient ID.

    Files are named either:
      CQ500CT0 CQ500CT0.nii.gz            ← single/primary series
      CQ500CT0 CQ500CT0__CT PLAIN THIN.nii.gz  ← extra series (after __)

    The patient ID is everything before the first __ (or the whole stem if no __).
    """
    groups = defaultdict(list)
    for f in sorted(nifti_dir.glob("*.nii.gz")):
        # Strip .nii.gz, then take everything before __ as the patient ID
        stem = f.name.replace(".nii.gz", "")
        patient_id = stem.split("__")[0].strip()
        groups[patient_id].append(f)
    return dict(groups)


def pick_best_series(series_files: list[Path]) -> Path:
    """
    From a list of series for one patient, return the file with the most slices.

    Why most slices? More slices = thinner slice thickness = higher Z resolution.
    The thick reconstructions (32 slices at 5mm) exist for quick radiologist review;
    the thin ones (256+ slices at 0.6mm) are the original high-res acquisition.
    """
    def slice_count(f: Path) -> int:
        img = nib.load(str(f))
        return img.shape[2]  # Z dimension = number of slices

    return max(series_files, key=slice_count)


# ── Step 2: reorientation ─────────────────────────────────────────────────────

def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorient the volume to RAS+ (Right, Anterior, Superior) canonical orientation.

    Why? Different CT scanners store data with different axis conventions — some
    flip left/right, some flip head/feet. The affine matrix encodes this, and values
    can come out as negative (meaning a flipped axis). nib.as_closest_canonical()
    reorders and flips axes so the volume is always in the same standard orientation.
    After this, "left" always means anatomical left, "front" always means front of head.
    """
    return nib.as_closest_canonical(img)


# ── Step 3: resampling ────────────────────────────────────────────────────────

def resample_to_isotropic(img: nib.Nifti1Image, target_spacing_mm: float = 1.0) -> nib.Nifti1Image:
    """
    Resample the volume so all three axes have equal spacing (isotropic voxels).

    Why? Raw CT scans are anisotropic: typically 0.5mm in-plane but 5mm between slices.
    If you render this 3D volume directly, the shape is geometrically wrong — the head
    appears 10x squashed vertically. Resampling to 1mm³ makes every voxel represent
    the same physical size in X, Y, and Z, which is essential for correct 3D rendering.

    We use SimpleITK for resampling because it correctly handles the physical-space
    transform (it uses the affine/direction/origin metadata, not just array indices).
    Linear interpolation is standard for CT — it avoids the blocky artefacts of nearest-
    neighbour while not introducing the ringing of higher-order methods.
    Fill value = -1000 HU = air, so any new voxels created at the volume boundary
    (padding needed after resampling) are treated as background air.
    """
    # Convert nibabel → SimpleITK (preserves physical spacing and orientation)
    data = img.get_fdata(dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))  # SimpleITK uses ZYX order
    sitk_img.SetSpacing([float(s) for s in img.header.get_zooms()[:3]])

    original_spacing = sitk_img.GetSpacing()          # e.g. (0.49, 0.49, 5.0)
    original_size    = sitk_img.GetSize()              # e.g. (512, 512, 32)

    # Compute new size so the physical extent stays the same
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing_mm))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing_mm] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000.0)            # fill new border voxels with air
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(sitk_img)

    # Convert back to nibabel
    new_data = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0).astype(np.float32)
    new_affine = img.affine.copy()
    # Update the voxel spacing in the affine diagonal
    scale = target_spacing_mm / np.array(img.header.get_zooms()[:3])
    new_affine[:3, :3] = img.affine[:3, :3] * scale
    return nib.Nifti1Image(new_data, new_affine)


# ── Step 4: HU clipping ───────────────────────────────────────────────────────

def clip_hu(img: nib.Nifti1Image, hu_min: float = -1000.0, hu_max: float = 3000.0) -> nib.Nifti1Image:
    """
    Clip Hounsfield Unit values to the physically meaningful range [-1000, 3000].

    Why? Raw CT files contain values outside this range:
      -3024 HU = scanner sentinel for "outside field of view" (not real tissue)
      >3000 HU = usually metal implants or scan table artefacts

    The transfer function that maps HU → colour/opacity is designed around the
    physical range, so artefact values outside it would break the mapping.
    Clipping them to the boundary value (-1000 = air, 3000 = dense bone) is safe.
    """
    data = np.clip(img.get_fdata(dtype=np.float32), hu_min, hu_max)
    return nib.Nifti1Image(data, img.affine, img.header)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(img: nib.Nifti1Image, patient_id: str):
    """Quick sanity checks on the preprocessed volume — fails loudly if something is wrong."""
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]

    assert not np.any(np.isnan(data)),  f"{patient_id}: contains NaN values"
    assert not np.any(np.isinf(data)),  f"{patient_id}: contains Inf values"
    assert data.min() >= -1000.0,       f"{patient_id}: HU below -1000 ({data.min():.1f})"
    assert data.max() <= 3000.0,        f"{patient_id}: HU above 3000 ({data.max():.1f})"
    assert all(abs(z - 1.0) < 0.01 for z in zooms), \
        f"{patient_id}: spacing not isotropic after resampling: {zooms}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess_patient(series_files: list[Path], output_dir: Path, patient_id: str) -> bool:
    # ── Step 1: pick the series with the most slices ──────────────────────────
    best = pick_best_series(series_files)
    n_series = len(series_files)
    note = f"(picked from {n_series} series)" if n_series > 1 else ""

    # ── Step 2: load and reorient to RAS+ ─────────────────────────────────────
    img = nib.load(str(best))
    img = reorient_to_ras(img)

    # ── Step 3: resample to 1mm³ isotropic ────────────────────────────────────
    original_spacing = img.header.get_zooms()[:3]
    img = resample_to_isotropic(img, target_spacing_mm=1.0)

    # ── Step 4: clip HU to [-1000, 3000] ──────────────────────────────────────
    img = clip_hu(img)

    # ── Validate before saving ────────────────────────────────────────────────
    validate(img, patient_id)

    # ── Step 5: save ──────────────────────────────────────────────────────────
    out_path = output_dir / f"{patient_id}.nii.gz"
    nib.save(img, str(out_path))

    shape   = img.shape
    zooms   = img.header.get_zooms()[:3]
    orig_sp = tuple(f"{s:.2f}" for s in original_spacing)
    print(f"  [ok] {patient_id:<30}  {str(shape):<22}  {orig_sp} → 1mm³  {note}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess CQ500 NIfTI volumes")
    parser.add_argument("--input",  required=True, help="Directory of raw NIfTI files (cq500_nifti/)")
    parser.add_argument("--output", required=True, help="Output directory for preprocessed volumes")
    parser.add_argument("--limit",  type=int, default=None, help="Only process first N patients (for testing)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    patients = group_by_patient(input_dir)
    patient_ids = sorted(patients.keys())
    if args.limit:
        patient_ids = patient_ids[:args.limit]

    print(f"Found {len(patients)} patients, processing {len(patient_ids)}")
    print(f"Output → {output_dir}\n")

    ok = failed = skipped = 0
    for patient_id in patient_ids:
        out_path = output_dir / f"{patient_id}.nii.gz"
        if out_path.exists():
            print(f"  [skip] {patient_id}")
            skipped += 1
            continue
        try:
            preprocess_patient(patients[patient_id], output_dir, patient_id)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {patient_id}: {e}")
            failed += 1

    print(f"\nDone: {ok} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
