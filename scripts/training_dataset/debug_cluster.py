"""
debug_cluster.py — Quick sanity checks for the cluster environment.
Run: python3 scripts/training_dataset/debug_cluster.py
"""
import sys

DATASET = "/home/vulovic/jasmina/dataset"
SUBJECT = "s0000"
TISSUE  = "liver"

print("=" * 60)
print("1. nibabel")
try:
    import nibabel as nib
    import numpy as np
    path = f"{DATASET}/{SUBJECT}/segmentations/{TISSUE}.nii.gz"
    img  = nib.load(path)
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    print(f"   OK  voxels={data.sum()}  shape={data.shape}  zooms={img.header.get_zooms()[:3]}")
except Exception as e:
    print(f"   FAIL: {e}")

print("=" * 60)
print("2. pyvista contour")
try:
    import pyvista as pv
    import numpy as np
    g = pv.ImageData()
    g.dimensions = (10, 10, 10)
    g.spacing    = (1, 1, 1)
    g.cell_data["v"] = np.random.rand(9 * 9 * 9)
    m = g.cell_data_to_point_data().contour([0.5])
    print(f"   OK  points={m.n_points}")
except Exception as e:
    print(f"   FAIL: {e}")

print("=" * 60)
print("3. full mesh extraction on liver (verbose errors)")
try:
    import nibabel as nib
    import numpy as np
    import pyvista as pv
    from pathlib import Path

    seg_path = Path(f"{DATASET}/{SUBJECT}/segmentations/{TISSUE}.nii.gz")
    ct_path  = Path(f"{DATASET}/{SUBJECT}/ct.nii.gz")

    ct   = nib.load(str(ct_path))
    zooms = ct.header.get_zooms()[:3]
    print(f"   zooms={zooms}")

    img  = nib.load(str(seg_path))
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    print(f"   data sum={data.sum()}  shape={data.shape}")

    sx, sy, sz = zooms
    grid = pv.ImageData()
    grid.dimensions = (data.shape[0]+1, data.shape[1]+1, data.shape[2]+1)
    grid.spacing    = (sx, sy, sz)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["val"] = data.flatten(order="F")
    print(f"   grid built OK")

    mesh = grid.cell_data_to_point_data().contour(isosurfaces=[0.5], scalars="val")
    print(f"   contour OK  points={mesh.n_points}")

    if mesh.n_points > 0:
        mesh2 = mesh.connectivity(extraction_mode="largest").smooth(n_iter=30, relaxation_factor=0.05)
        print(f"   smooth OK  points={mesh2.n_points}")
    else:
        print("   mesh empty after contour")

except Exception as e:
    import traceback
    print(f"   FAIL: {e}")
    traceback.print_exc()

print("=" * 60)
print("4. checking all 29 tissues exist for s0000")
TISSUES = [
    "autochthon_left","autochthon_right","lung_lower_lobe_left","lung_lower_lobe_right",
    "lung_upper_lobe_left","lung_upper_lobe_right","vertebrae_T12","vertebrae_L1",
    "vertebrae_L2","vertebrae_L3","vertebrae_L4","vertebrae_L5","heart","esophagus",
    "liver","stomach","gallbladder","spleen","kidney_right","kidney_left","pancreas",
    "duodenum","small_bowel","colon","urinary_bladder","aorta","inferior_vena_cava",
    "portal_vein_and_splenic_vein","superior_vena_cava",
]
from pathlib import Path
missing = [t for t in TISSUES if not Path(f"{DATASET}/{SUBJECT}/segmentations/{t}.nii.gz").exists()]
found   = len(TISSUES) - len(missing)
print(f"   {found}/29 files present")
if missing:
    print(f"   missing: {missing}")
