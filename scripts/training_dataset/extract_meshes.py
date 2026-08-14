"""
extract_meshes.py — Batch mesh extraction for all TotalSegmentator subjects.

For each subject, loads the 29 tissue segmentation NIfTI files, runs marching
cubes to produce a surface mesh, smooths it, and saves as OBJ.

Output: data/meshes/{subject}/{seg_name}.obj
Already-extracted subjects and tissues are skipped — safe to re-run.

Usage:
    python scripts/training_dataset/extract_meshes.py
    python scripts/training_dataset/extract_meshes.py --dataset /path/to/Totalsegmentator_dataset_v201
    python scripts/training_dataset/extract_meshes.py --workers 4 --start 0 --count 100
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import pyvista as pv


# The 29 tissues we render — must match TISSUES list in generate_training_dataset.py
TISSUE_NAMES = [
    "autochthon_left",
    "autochthon_right",
    "lung_lower_lobe_left",
    "lung_lower_lobe_right",
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "heart",
    "esophagus",
    "liver",
    "stomach",
    "gallbladder",
    "spleen",
    "kidney_right",
    "kidney_left",
    "pancreas",
    "duodenum",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "superior_vena_cava",
]


# Structures thin enough that heavy smoothing can erode or disconnect them.
THIN_STRUCTURES = (
    "aorta", "inferior_vena_cava", "superior_vena_cava",
    "portal_vein_and_splenic_vein", "esophagus",
)


def extract_mesh(seg_path: Path, zooms) -> pv.PolyData | None:
    img  = nib.load(str(seg_path))
    # float32, NOT uint8. With uint8, cell_data_to_point_data() below truncates every
    # 2x2x2 average back to 0/1 (2 distinct values instead of 9), so contour() runs on
    # a still-binary field and produces a marching-cubes staircase. That staircase is
    # the wavy corduroy visible on every organ in renders v20-v22.
    data = (img.get_fdata() > 0.5).astype(np.float32)
    if data.sum() < 50:
        return None

    sx, sy, sz = zooms
    grid = pv.ImageData()
    grid.dimensions = (data.shape[0]+1, data.shape[1]+1, data.shape[2]+1)
    grid.spacing    = (sx, sy, sz)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["val"] = data.flatten(order="F")

    mesh = (grid.cell_data_to_point_data()
                .contour(isosurfaces=[0.5], scalars="val"))
    if mesh.n_points == 0:
        return None

    # relaxation_factor 0.05 was ~10x too weak to repair the staircase; Bade et al.,
    # "Reducing Artifacts in Surface Meshes Extracted from Binary Volumes", use
    # lambda = 0.5 over ~20 iterations for liver. Thin structures get a gentler pass so
    # a 2-voxel-wide vessel is not thinned or broken.
    thin  = any(t in seg_path.name for t in THIN_STRUCTURES)
    mesh  = mesh.connectivity(extraction_mode="largest")
    return mesh.smooth(n_iter=12 if thin else 20,
                       relaxation_factor=0.15 if thin else 0.3)


def save_obj(mesh: pv.PolyData, path: Path):
    m = mesh.triangulate().compute_normals(cell_normals=False, point_normals=True)
    verts   = m.points
    normals = m.point_normals
    faces   = m.faces.reshape(-1, 4)[:, 1:]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for tri in faces:
            a, b, c = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def process_subject(args_tuple) -> tuple[str, int, list[str]]:
    subject, dataset_dir, mesh_base, force = args_tuple
    subj_dir = Path(dataset_dir) / subject
    seg_dir  = subj_dir / "segmentations"
    out_dir  = Path(mesh_base) / subject

    if not seg_dir.exists():
        return subject, 0, ["no segmentations dir"]

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ct_img = nib.load(str(subj_dir / "ct.nii.gz"))
        zooms  = ct_img.header.get_zooms()[:3]
    except Exception as e:
        return subject, 0, [f"ct load failed: {e}"]

    extracted = 0
    skipped   = 0
    missing   = []

    stale_uv = 0
    for name in TISSUE_NAMES:
        out_path = out_dir / f"{name}.obj"
        if out_path.exists() and not force:
            skipped += 1
            continue
        # The renderer prefers {name}_uv.obj over {name}.obj, so a leftover _uv.obj
        # from the old extraction would silently shadow everything written here.
        uv_path = out_dir / f"{name}_uv.obj"
        if force and uv_path.exists():
            uv_path.unlink()
            stale_uv += 1
        seg_path = seg_dir / f"{name}.nii.gz"
        if not seg_path.exists():
            missing.append(name)
            continue
        try:
            mesh = extract_mesh(seg_path, zooms)
            if mesh is None:
                missing.append(f"{name}(empty)")
                continue
            save_obj(mesh, out_path)
            extracted += 1
        except Exception as e:
            missing.append(f"{name}(err:{e})")

    total = extracted + skipped
    notes = []
    if skipped:
        notes.append(f"{skipped} already done")
    if missing:
        notes.append(f"{len(missing)} missing/failed")
    if stale_uv:
        notes.append(f"{stale_uv} stale _uv.obj removed")
    return subject, total, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=(
        "/Users/jasminavulovic/Documents/Masters/TEZAAA/Totalsegmentator_dataset_v201"))
    ap.add_argument("--mesh_dir", default="data/meshes")
    ap.add_argument("--workers",  type=int, default=4,
                    help="parallel subjects (set to 1 to disable multiprocessing)")
    ap.add_argument("--subjects", default="",
                    help="space-separated subject ids, e.g. --subjects \"s0050 s0001\". "
                         "Overrides --start/--count. Use this to validate the staircase "
                         "fix on one subject before committing to a full re-extraction.")
    ap.add_argument("--start",    type=int, default=0)
    ap.add_argument("--count",    type=int, default=9999)
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the OBJ exists, and delete the stale "
                         "{name}_uv.obj that would otherwise shadow it in the renderer. "
                         "Required when re-extracting after the staircase fix.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    mesh_base   = Path(args.mesh_dir)
    mesh_base.mkdir(parents=True, exist_ok=True)

    # Collect all subject dirs (skip meta.csv and non-dirs)
    all_subjects = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and d.name.startswith("s")
    ])

    if args.subjects.strip():
        wanted   = args.subjects.split()
        unknown  = [s_ for s_ in wanted if s_ not in all_subjects]
        if unknown:
            print(f"ERROR: not found in {dataset_dir}: {unknown}")
            sys.exit(1)
        subjects = wanted
    else:
        subjects = all_subjects[args.start : args.start + args.count]
    total    = len(subjects)

    print(f"\n{'='*60}")
    print(f"Mesh extraction — {total} subjects  ({args.workers} workers)")
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {mesh_base.resolve()}")
    print(f"Tissues : {len(TISSUE_NAMES)}")
    print(f"Mode    : {'FORCE re-extract (overwrites, clears _uv.obj)' if args.force else 'skip existing'}")
    print(f"{'='*60}\n")

    t0       = time.time()
    done     = 0
    task_args = [(s, str(dataset_dir), str(mesh_base), args.force) for s in subjects]

    if args.workers == 1:
        for ta in task_args:
            subj, n, notes = process_subject(ta)
            done += 1
            note_str = "  " + ", ".join(notes) if notes else ""
            print(f"[{done:>4}/{total}] {subj}  →  {n} tissues{note_str}")
            sys.stdout.flush()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_subject, ta): ta[0] for ta in task_args}
            for fut in as_completed(futures):
                subj, n, notes = fut.result()
                done += 1
                note_str = "  " + ", ".join(notes) if notes else ""
                print(f"[{done:>4}/{total}] {subj}  →  {n} tissues{note_str}")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  ({elapsed/total:.1f}s per subject)")


if __name__ == "__main__":
    main()
