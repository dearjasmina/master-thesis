"""
Convert CQ500 (or any nested DICOM tree) to NIfTI (.nii.gz) files.

Walks the full directory tree to find leaf directories containing .dcm files,
so it handles arbitrarily deep layouts like:
    CQ500CT65 CQ500CT65 / Unknown Study / CT 2.55mm / *.dcm

Usage:
    # Single patient folder (one ZIP extracted)
    python scripts/dicom_to_nifti.py --input "data/raw/cq500/qure.headct.study/CQ500CT1 CQ500CT1" --output data/raw/cq500_nifti/

    # Whole dataset (all extracted patient folders)
    python scripts/dicom_to_nifti.py --input data/raw/cq500/qure.headct.study/ --output data/raw/cq500_nifti/

Output file is named after the top-level patient folder: CQ500CT1.nii.gz
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk


def find_dicom_dirs(root: Path) -> list[Path]:
    """Return all directories that directly contain at least one .dcm file."""
    return sorted({p.parent for p in root.rglob("*.dcm")})


def convert_series(dicom_dir: Path, output_path: Path) -> bool:
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))

    if not dicom_files:
        print(f"  [skip] no readable series in {dicom_dir}")
        return False

    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path))

    size    = image.GetSize()
    spacing = image.GetSpacing()
    print(f"  [ok] {output_path.name}  size={size}  spacing={tuple(f'{s:.3f}' for s in spacing)}")
    return True


def patient_name(dicom_dir: Path, input_root: Path) -> str:
    """Use the first path component below input_root as the output filename."""
    rel = dicom_dir.relative_to(input_root)
    return rel.parts[0]


def main():
    parser = argparse.ArgumentParser(description="Batch DICOM → NIfTI conversion")
    parser.add_argument("--input",  required=True, help="Root dir (patient folder or folder of patients)")
    parser.add_argument("--output", required=True, help="Output directory for .nii.gz files")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_dirs = find_dicom_dirs(input_dir)
    if not dicom_dirs:
        print(f"No .dcm files found under {input_dir}")
        sys.exit(1)

    print(f"Found {len(dicom_dirs)} DICOM series directory/ies under {input_dir}")

    ok = failed = skipped = 0
    seen_names: set[str] = set()

    for dicom_dir in dicom_dirs:
        name = patient_name(dicom_dir, input_dir)

        # If multiple series per patient, append series dir name to avoid collision
        out_name = name if name not in seen_names else f"{name}__{dicom_dir.name}"
        seen_names.add(name)

        out_file = output_dir / f"{out_name}.nii.gz"
        if out_file.exists():
            print(f"  [skip] already exists: {out_file.name}")
            skipped += 1
            continue

        if convert_series(dicom_dir, out_file):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} converted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
