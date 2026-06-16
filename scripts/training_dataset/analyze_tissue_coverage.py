#!/usr/bin/env python3
"""
Analyze tissue coverage across completed and remaining subjects.
Run on cluster: python3 scripts/training_dataset/analyze_tissue_coverage.py

Outputs:
  - Tissue representation in already-rendered subjects
  - Remaining subjects ranked by how many tissues they have
  - Recommended subjects to render next
"""
from pathlib import Path
from collections import defaultdict, Counter

TISSUES = [
    "autochthon_left", "autochthon_right",
    "lung_lower_lobe_left", "lung_lower_lobe_right",
    "lung_upper_lobe_left", "lung_upper_lobe_right",
    "vertebrae_T12", "vertebrae_L1", "vertebrae_L2",
    "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "heart", "esophagus", "liver", "stomach", "gallbladder",
    "spleen", "kidney_right", "kidney_left", "pancreas",
    "duodenum", "small_bowel", "colon", "urinary_bladder",
    "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "superior_vena_cava",
]

DATASET  = Path("/home/vulovic/jasmina/dataset")
TRAINING = Path("/home/vulovic/jasmina/master-thesis/data/training_dataset")
MESHES   = Path("/home/vulovic/jasmina/master-thesis/data/meshes")
MIN_BYTES = 50_000  # NIfTIs smaller than 50KB are effectively empty


# ── helpers ──────────────────────────────────────────────────────────────────

def completed_subjects():
    out = []
    for d in sorted(TRAINING.iterdir()):
        if d.is_dir() and len(list(d.glob("v*"))) >= 20:
            out.append(d.name)
    return out

def remaining_subjects():
    out = []
    for d in sorted(DATASET.iterdir()):
        if d.is_dir() and d.name.startswith("s"):
            out.append(d.name)
    return out

def tissues_from_meshes(subject):
    mesh_dir = MESHES / subject
    if not mesh_dir.exists():
        return set()
    return {f.stem for f in mesh_dir.glob("*.obj") if f.stem in TISSUES}

def tissues_from_raw(subject):
    seg_dir = DATASET / subject / "segmentations"
    if not seg_dir.exists():
        return set()
    return {
        t for t in TISSUES
        if (seg_dir / f"{t}.nii.gz").exists()
        and (seg_dir / f"{t}.nii.gz").stat().st_size > MIN_BYTES
    }


# ── completed: tissue representation ─────────────────────────────────────────

print("=" * 60)
print("COMPLETED SUBJECTS")
done = completed_subjects()
print(f"  Total: {len(done)}")

tissue_count = defaultdict(int)
for subj in done:
    for t in tissues_from_meshes(subj):
        tissue_count[t] += 1

if any(tissue_count.values()):
    print("\n  Tissue representation (from mesh files):")
    for t in TISSUES:
        n = tissue_count[t]
        pct = n / len(done) * 100 if done else 0
        print(f"    {t:42s} {n:4d} ({pct:5.1f}%)")
else:
    print("  (mesh files not found — skipping tissue breakdown for completed set)")


# ── remaining: tissue coverage ────────────────────────────────────────────────

print()
print("=" * 60)
print("REMAINING SUBJECTS (in raw dataset)")
remaining = remaining_subjects()
print(f"  Total: {len(remaining)}")
print("  Scanning tissue coverage (this takes ~30s)...")

coverage = {}
for subj in remaining:
    coverage[subj] = tissues_from_raw(subj)

dist = Counter(len(v) for v in coverage.values())
print("\n  Distribution of tissue counts:")
for n in sorted(dist.keys(), reverse=True):
    bar = "#" * dist[n]
    print(f"    {n:2d} tissues: {dist[n]:3d} subjects  {bar}")


# ── ranked recommendations ────────────────────────────────────────────────────

print()
print("=" * 60)
print("TOP REMAINING SUBJECTS BY TISSUE COUNT")
ranked = sorted(coverage.items(), key=lambda x: len(x[1]), reverse=True)

print(f"\n  {'Subject':<10} {'Tissues':>8}  Present")
print(f"  {'-'*8}  {'-'*6}  {'-'*40}")
for subj, tissues in ranked[:80]:
    preview = ", ".join(sorted(tissues)[:6])
    suffix = "..." if len(tissues) > 6 else ""
    print(f"  {subj:<10} {len(tissues):>6}/29  {preview}{suffix}")

# Save recommended list for easy use
out_file = Path("/home/vulovic/jasmina/master-thesis/recommended_subjects.txt")
with open(out_file, "w") as f:
    for subj, _ in ranked:
        f.write(f"{subj}\n")
print(f"\n  Full ranked list saved to: {out_file}")
print(f"  (use --subjects flag or feed into run_dataset_generation.sh)")
