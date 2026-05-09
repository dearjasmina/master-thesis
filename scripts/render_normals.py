"""
Render a surface normal map from a preprocessed CT NIfTI volume.

Normal convention (matches ControlNet normalbae):
  R = X axis (right = red)
  G = Y axis (up = green)
  B = Z axis (toward camera = blue)
  Normals mapped from [-1, 1] → [0, 255]

Usage:
    python scripts/render_normals.py --volume CT95
    python scripts/render_normals.py --volume CT106
"""

import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image


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
    return (
        grid.cell_data_to_point_data()
        .contour(isosurfaces=[300], scalars="HU")
        .connectivity(extraction_mode="largest")
    )


def set_skull_camera(plotter: pv.Plotter, grid: pv.ImageData):
    cx, cy, cz = [s * sp / 2 for s, sp in zip(grid.dimensions, grid.spacing)]
    radius = max(grid.dimensions) * max(grid.spacing) * 0.9
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.position    = (cx + radius * 0.6,
                                  cy - radius * 1.1,
                                  cz + radius * 0.5)
    plotter.camera.up = (0, 0, 1)


def render_normals(grid: pv.ImageData, size: int = 512) -> np.ndarray:
    """
    Render surface normals as an RGB image.
    Each pixel's colour encodes the surface normal direction at that point.
    Background pixels (no surface) → (128, 128, 255) = neutral blue pointing toward camera.
    """
    surface = extract_bone_surface(grid)
    surface = surface.compute_normals(
        consistent_normals=True,
        auto_orient_normals=True,
        non_manifold_traversal=False,
    )

    # Map normals [-1,1] → [0,255] per channel
    normals = surface.point_data["Normals"]  # (N, 3) float array

    # Assign normal components as RGB scalars for rendering
    # We render each channel separately and composite
    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = [0.502, 0.502, 1.0]  # neutral blue = (128,128,255) = facing camera

    # Colour the mesh by its normal vectors mapped to RGB
    normal_colours = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)  # (N, 3)
    # PyVista needs RGBA or we pass as scalar; easiest: set mesh colour array
    surface.point_data["NormalRGB"] = normal_colours.astype(np.float32)

    # Render R, G, B channels into separate screenshots, then composite
    channels = []
    for ch, ch_name in enumerate(["NX", "NY", "NZ"]):
        surface.point_data[ch_name] = ((normals[:, ch] + 1.0) / 2.0 * 255)

    # Simplest approach: render with RGB mapped from normals using a flat unlit shader
    # We build an RGBA array and assign it
    rgba = np.ones((surface.n_points, 4), dtype=np.uint8) * 255
    rgba[:, :3] = normal_colours
    surface.point_data["RGBA"] = rgba

    pl.add_mesh(
        surface,
        scalars="RGBA",
        rgba=True,
        ambient=1.0,
        diffuse=0.0,
        specular=0.0,
    )
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)  # RGB numpy array
    pl.close()
    return img


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

    vol_id     = vol_dir.name.split()[0]
    nifti_path = Path(f"data/processed/{vol_id} {vol_id}.nii.gz")

    print(f"Loading {nifti_path.name} …")
    grid = load_as_pyvista(nifti_path)

    print("Rendering normal map …")
    normals_img = render_normals(grid, args.size)

    out_path = vol_dir / "normals.png"
    Image.fromarray(normals_img).save(out_path)
    print(f"Saved → {out_path}")

    # Quick check
    non_bg = (normals_img[:, :, 2] < 250).sum()  # non-blue pixels = surface pixels
    print(f"Surface pixels: {non_bg:,}  ({non_bg / args.size**2 * 100:.1f}% of image)")


if __name__ == "__main__":
    main()
