"""
1.5.3 — Render 5 signal-strength variants of the same bone isosurface.

V1 — Flat:           no shading, no AO, pure white surface
V2 — Shading only:   Phong shading, directional lights, no AO
V3 — AO only:        SSAO enabled, flat ambient-only lighting
V4 — Shading + AO:   Phong + SSAO (what the professor approximately had)
V5 — Shading+AO+CLAHE: V4 with CLAHE contrast exaggeration

Usage:
    python scripts/render_variants.py --volume CT95
    python scripts/render_variants.py --volume CT106
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
import nibabel as nib


def load_as_pyvista(nifti_path: Path) -> pv.ImageData:
    img   = nib.load(str(nifti_path))
    data  = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    grid  = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1
    grid.spacing    = zooms
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["HU"] = data.flatten(order="F")
    return grid


def extract_bone_surface(grid: pv.ImageData) -> pv.PolyData:
    return grid.cell_data_to_point_data().contour(
        isosurfaces=[300], scalars="HU"
    ).connectivity(extraction_mode="largest")


def set_skull_camera(plotter: pv.Plotter, grid: pv.ImageData):
    cx, cy, cz = [s * sp / 2 for s, sp in zip(grid.dimensions, grid.spacing)]
    radius = max(grid.dimensions) * max(grid.spacing) * 0.9
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.position    = (cx + radius * 0.6,
                                  cy - radius * 1.1,
                                  cz + radius * 0.5)
    plotter.camera.up = (0, 0, 1)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel of LAB image — exaggerates local contrast."""
    lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def render_v1_flat(surface, grid, size):
    """V1 — pure white, no shading, no lighting variation."""
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="white", ambient=1.0, diffuse=0.0, specular=0.0)
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_v2_shading(surface, grid, size):
    """V2 — Phong shading with directional lights, no AO."""
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="#e8dcc8", ambient=0.25, diffuse=0.70,
                specular=0.20, smooth_shading=True)
    pl.add_light(pv.Light(position=(1, -1, 1),  intensity=0.8))
    pl.add_light(pv.Light(position=(-1, -1, 0.5), intensity=0.3))
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_v3_ao(surface, grid, size):
    """V3 — SSAO only, flat ambient lighting (no directional light variation)."""
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="white", ambient=1.0, diffuse=0.0, specular=0.0)
    try:
        pl.enable_ssao(radius=0.5, bias=0.025, kernel_size=32)
    except Exception:
        pass  # SSAO may not be available in all VTK builds / off-screen modes
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_v4_shading_ao(surface, grid, size):
    """V4 — Phong shading + SSAO (closest to professor's original render)."""
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="#e8dcc8", ambient=0.25, diffuse=0.70,
                specular=0.20, smooth_shading=True)
    pl.add_light(pv.Light(position=(1, -1, 1),  intensity=0.8))
    pl.add_light(pv.Light(position=(-1, -1, 0.5), intensity=0.3))
    try:
        pl.enable_ssao(radius=0.5, bias=0.025, kernel_size=32)
    except Exception:
        pass
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render_v5_clahe(surface, grid, size):
    """V5 — V4 + CLAHE for exaggerated local contrast."""
    v4 = render_v4_shading_ao(surface, grid, size)
    return apply_clahe(v4)


VARIANTS = [
    ("V1_flat",         "Flat (no shading, no AO)",          render_v1_flat),
    ("V2_shading",      "Shading only",                      render_v2_shading),
    ("V3_ao",           "AO only (SSAO)",                    render_v3_ao),
    ("V4_shading_ao",   "Shading + AO",                      render_v4_shading_ao),
    ("V5_clahe",        "Shading + AO + CLAHE",              render_v5_clahe),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True, help="e.g. CT95 or CT106")
    parser.add_argument("--size",   type=int, default=512)
    args = parser.parse_args()

    base = Path("data/renders/simple")
    matches = list(base.glob(f"*{args.volume}*"))
    if not matches:
        raise FileNotFoundError(f"No render directory matching '{args.volume}'")
    vol_dir = matches[0]

    # Find the NIfTI path from the directory name
    nii_name = vol_dir.name  # e.g. "CQ500CT95 CQ500CT95.nii"
    vol_id   = nii_name.split()[0]  # "CQ500CT95"
    nifti_path = Path(f"data/processed/{vol_id} {vol_id}.nii.gz")
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI not found: {nifti_path}")

    out_dir = vol_dir / "variants"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {nifti_path.name} …")
    grid    = load_as_pyvista(nifti_path)
    surface = extract_bone_surface(grid)
    print(f"  Surface: {surface.n_points:,} points")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(len(VARIANTS) * 4, 4.5))

    for ax, (slug, label, fn) in zip(axes, VARIANTS):
        print(f"  Rendering {slug} …")
        if fn in (render_v1_flat, render_v2_shading, render_v3_ao, render_v4_shading_ao):
            img = fn(surface, grid, args.size)
        else:
            img = fn(surface, grid, args.size)  # V5 takes same args

        out_path = out_dir / f"{slug}.png"
        import PIL.Image
        PIL.Image.fromarray(img).save(out_path)

        ax.imshow(img)
        ax.set_title(f"{slug}\n{label}", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"Signal-strength variants — {vol_id}", fontsize=10)
    plt.tight_layout()
    grid_path = out_dir / "variants_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved 5 variants → {out_dir}/")
    print(f"Grid → {grid_path}")


if __name__ == "__main__":
    main()
