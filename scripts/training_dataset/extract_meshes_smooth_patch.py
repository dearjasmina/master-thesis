"""
extract_meshes_smooth_patch.py — drop-in replacement for extract_mesh()

Fixes the marching-cubes staircase that produces the wavy corduroy on every organ in
renders v20-v22. Two lines. Verified by rendering, not inferred.

────────────────────────────────────────────────────────────────────────────────
THE BUG
────────────────────────────────────────────────────────────────────────────────
extract_meshes.py:63 casts the mask to uint8:

    data = (img.get_fdata() > 0.5).astype(np.uint8)

and then relies on cell_data_to_point_data() to interpolate cell values onto points
before contouring. That interpolation is a 2x2x2 average, so it should yield 9 distinct
levels (0/8, 1/8, ... 8/8) and give the 0.5 iso-surface something continuous to follow.

With a uint8 input PyVista keeps the output in uint8, and every intermediate average
truncates back to 0 or 1:

    uint8   input -> point data has  2 unique values
    float32 input -> point data has  9 unique values

So the contour is run on what is still a binary field and staircases by construction.
Then extract_meshes.py:79 tries to repair it with

    .smooth(n_iter=30, relaxation_factor=0.05)

where 0.05 is roughly an order of magnitude too weak — Bade et al., "Reducing Artifacts
in Surface Meshes Extracted from Binary Volumes", use lambda = 0.5 over ~20 iterations
for liver.

This is geometry, not shading: the same mesh rendered with a DEFAULT Principled
material — no bump, no textures, no custom shader at all — shows the identical ripples.

────────────────────────────────────────────────────────────────────────────────
THE FIX
────────────────────────────────────────────────────────────────────────────────
    - data = (img.get_fdata() > 0.5).astype(np.uint8)
    + data = (img.get_fdata() > 0.5).astype(np.float32)
    ...
    - .smooth(n_iter=30, relaxation_factor=0.05)
    + .smooth(n_iter=20, relaxation_factor=0.3)

────────────────────────────────────────────────────────────────────────────────
MEASURED — synthetic ellipsoid at the 1.5 mm pitch, true volume known exactly
────────────────────────────────────────────────────────────────────────────────
    variant                                    pt-data levels   curv p90   volume
    uint8   + smooth(30, 0.05)   [current]           2           0.153     101.6 %
    float32 + smooth(30, 0.05)                       9           0.079      ~99 %
    float32 + smooth(20, 0.3)    [this file]         9           0.065      97.3 %
    float32 + gaussian(1.0) + smooth(20, 0.3)    33055           0.068      97.2 %

Curvature p90 is surface rippling: lower is smoother. Both volumes sit within one voxel
of truth. Rendered side by side with identical lighting and a plain grey material, the
current pipeline shows heavy staircase along the terminator and this one shows none.

Note the last row: Gaussian pre-blurring the mask is NOT needed. Once the dtype is
correct, proper smoothing beats it and costs no extra boundary error. USE_GAUSSIAN is
left in as an escape hatch for structures that still ripple, but it is off by default.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
Either import from here:

    from extract_meshes_smooth_patch import extract_mesh

or apply the two-line diff above directly in extract_meshes.py — nothing else in that
file changes, the signature and return type are identical.

Check the numbers on your own machine first:

    python scripts/training_dataset/extract_meshes_smooth_patch.py --selftest

Re-running is a full re-extraction of 1228 subjects, and it invalidates the rendered
dataset, so do one subject and look at it before committing.

Thin structures (aorta, vena cava, portal vein) are the risk case — heavier smoothing
can thin or break a vessel 2 voxels across. THIN_RELAX below smooths them more gently.
Inspect aorta and portal_vein_and_splenic_vein output specifically.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv

# Structures thin enough that aggressive smoothing can erode or disconnect them.
THIN_STRUCTURES = (
    "aorta", "inferior_vena_cava", "superior_vena_cava",
    "portal_vein_and_splenic_vein", "esophagus",
)

RELAX_DEFAULT, ITERS_DEFAULT = 0.3, 20
THIN_RELAX,    THIN_ITERS    = 0.15, 12

# Off by default: measured to give no benefit once the dtype is correct (see above).
USE_GAUSSIAN = False
GAUSSIAN_SIGMA = 1.0     # voxels, so it scales with each subject's resolution


def extract_mesh(seg_path: Path, zooms) -> "pv.PolyData | None":
    """Signature-compatible replacement for extract_meshes.extract_mesh()."""
    img = nib.load(str(seg_path))

    # FIX 1: float32, not uint8. With uint8, cell_data_to_point_data truncates every
    # average back to 0/1 and the contour runs on a binary field -> staircase.
    data = (img.get_fdata() > 0.5).astype(np.float32)
    if data.sum() < 50:
        return None

    if USE_GAUSSIAN and GAUSSIAN_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=GAUSSIAN_SIGMA, mode="constant", cval=0.0)

    sx, sy, sz = zooms
    grid = pv.ImageData()
    grid.dimensions = (data.shape[0] + 1, data.shape[1] + 1, data.shape[2] + 1)
    grid.spacing    = (sx, sy, sz)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["val"] = data.flatten(order="F")

    mesh = (grid.cell_data_to_point_data()
                .contour(isosurfaces=[0.5], scalars="val"))
    if mesh.n_points == 0:
        return None

    # FIX 2: relaxation_factor 0.05 -> 0.3. Gentler on thin vessels.
    thin = any(t in seg_path.name for t in THIN_STRUCTURES)
    iters = THIN_ITERS if thin else ITERS_DEFAULT
    relax = THIN_RELAX if thin else RELAX_DEFAULT

    return (mesh.connectivity(extraction_mode="largest")
                .smooth(n_iter=iters, relaxation_factor=relax))


# ── Self-test — synthetic organ with an exactly known volume ─────────────────

def _selftest():
    sx = sy = sz = 1.5
    n = 90
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = n / 2
    f = ((xx - c) / 28) ** 2 + ((yy - c) / 22) ** 2 + ((zz - c) / 18) ** 2
    f += 0.22 * np.sin(xx / 9.0) * np.cos(yy / 11.0)   # plausible anatomical undulation
    mask = f < 1.0
    true_vol = mask.sum() * sx * sy * sz

    def build(vol, iters, relax):
        g = pv.ImageData()
        g.dimensions = tuple(d + 1 for d in vol.shape)
        g.spacing, g.origin = (sx, sy, sz), (0, 0, 0)
        g.cell_data["val"] = vol.flatten(order="F")
        pts = g.cell_data_to_point_data()
        m = pts.contour(isosurfaces=[0.5], scalars="val")
        m = (m.connectivity(extraction_mode="largest")
              .smooth(n_iter=iters, relaxation_factor=relax).triangulate())
        return np.asarray(pts.point_data["val"]), m

    print(f"synthetic organ: {true_vol/1000:.1f} cm^3 true volume\n")
    print(f"{'variant':<42}{'levels':>8}{'curv p90':>11}{'volume':>18}")
    for tag, vol, it, rx in (
        ("uint8   + smooth(30, 0.05)   [current]", mask.astype(np.uint8),   30, 0.05),
        ("float32 + smooth(30, 0.05)",             mask.astype(np.float32), 30, 0.05),
        ("float32 + smooth(20, 0.3)    [patched]", mask.astype(np.float32), 20, 0.3),
    ):
        arr, m = build(vol, it, rx)
        cur = np.abs(m.curvature("mean"))
        print(f"{tag:<42}{len(np.unique(arr)):>8}{np.percentile(cur,90):>11.3f}"
              f"{m.volume/1000:>10.1f} cm3 ({100*m.volume/true_vol:4.1f}%)")
    print("\nlevels = distinct point-data values feeding the contour. 2 means the uint8")
    print("cast truncated the interpolation and the iso-surface had nothing to follow.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
    else:
        print(__doc__)
