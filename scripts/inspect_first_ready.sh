#!/usr/bin/env bash
# Waits for the first complete CQ500 zip, converts it to NIfTI, and visualises it.
# Run once and leave it — it will fire as soon as any zip finishes downloading.
#
# Usage: bash scripts/inspect_first_ready.sh

set -e
ZIP_DIR="data/raw/cq500/qure.headct.study"
NIFTI_DIR="data/raw/cq500_nifti"

echo "Watching $ZIP_DIR for completed ZIPs..."

while true; do
    FIRST_ZIP=$(ls "$ZIP_DIR"/*.zip 2>/dev/null | head -1)
    if [ -n "$FIRST_ZIP" ]; then
        echo "Found: $FIRST_ZIP"
        echo "Unzipping..."
        # Capture which top-level directory the ZIP extracts to
        EXTRACTED_DIR=$(unzip -Z1 "$FIRST_ZIP" | head -1 | cut -d/ -f1)
        unzip -q "$FIRST_ZIP" -d "$ZIP_DIR/"
        DICOM_DIR="$ZIP_DIR/$EXTRACTED_DIR"

        echo "DICOM dir: $DICOM_DIR"
        echo "Converting DICOM → NIfTI..."
        mkdir -p "$NIFTI_DIR"
        source venv/bin/activate
        python scripts/dicom_to_nifti.py --input "$DICOM_DIR" --output "$NIFTI_DIR/"

        # Find the NIfTI that was just created and visualise it
        NIFTI=$(ls -t "$NIFTI_DIR"/*.nii.gz 2>/dev/null | head -1)
        echo "Visualising $NIFTI ..."
        python scripts/visualise_volume.py "$NIFTI"
        break
    fi
    sleep 10
done
