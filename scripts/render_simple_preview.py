"""
Quick single-volume renderer for Phase 1.5 experiments.
Produces three buffers needed for zero-shot ControlNet testing:
  - color.png   : volume render with bone transfer function
  - depth.png   : normalised depth map of bone isosurface
  - edges.png   : Canny edge map derived from color render

Usage:
    python scripts/render_simple_preview.py --volume "data/processed/CQ500CT95 CQ500CT95.nii.gz"
    python scripts/render_simple_preview.py --volume "data/processed/CQ500CT95 CQ500CT95.nii.gz" --out data/renders/simple/ct95/
"""

import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pyvista as pv


# ── Transfer function ────────────────────────────────────────────────────────

def make_opacity_tf(n_points: int = 256, hu_range: tuple = (-1000, 2000)) -> np.ndarray:
    """
    Piecewise-linear opacity transfer function over HU values.
    Maps HU → opacity [0, 1]:
      air and soft tissue → nearly transparent
      bone (>300 HU)     → progressively opaque
    Returns an array of N opacity values evenly spaced across hu_range.
    """
    hu_min, hu_max = hu_range
    hu_vals = np.linspace(hu_min, hu_max, n_points)

    # Control points: (HU, opacity)
    ctrl = [
        (-1000, 0.00),
        ( -200, 0.00),
        (  100, 0.00),
        (  200, 0.02),
        (  300, 0.10),
        (  500, 0.50),
        (  700, 0.80),
        ( 1500, 1.00),
        ( 2000, 1.00),
    ]
    ctrl_hu  = np.array([c[0] for c in ctrl])
    ctrl_op  = np.array([c[1] for c in ctrl])
    return np.interp(hu_vals, ctrl_hu, ctrl_op)


# ── Volume → PyVista grid ────────────────────────────────────────────────────

def load_as_pyvista(nifti_path: Path) -> pv.ImageData:
    """Load a NIfTI volume into a PyVista ImageData with correct mm spacing."""
    img    = nib.load(str(nifti_path))
    data   = img.get_fdata(dtype=np.float32)
    zooms  = img.header.get_zooms()[:3]

    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1       # cell-centred: dims = shape + 1
    grid.spacing    = zooms
    grid.origin     = (0.0, 0.0, 0.0)
    grid.cell_data["HU"] = data.flatten(order="F")   # Fortran order matches VTK
    return grid


# ── Camera: nice 3/4 front-top view of the skull ────────────────────────────

def set_skull_camera(plotter: pv.Plotter, grid: pv.ImageData):
    """Position camera at a 3/4 angle — shows both skull shape and interior structure."""
    cx, cy, cz = [s * sp / 2 for s, sp in zip(grid.dimensions, grid.spacing)]
    radius = max(grid.dimensions) * max(grid.spacing) * 0.9
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.position    = (cx + radius * 0.6,
                                  cy - radius * 1.1,
                                  cz + radius * 0.5)
    plotter.camera.up          = (0, 0, 1)


# ── Extract bone surface (shared between color and depth renders) ─────────────

def extract_bone_surface(grid: pv.ImageData) -> pv.PolyData:
    """
    Marching-cubes isosurface at HU=300, keeping only the largest connected
    component. This removes scan-table artifacts and stray fragments — the skull
    is always by far the largest bone mass in a head CT.
    """
    surface = grid.cell_data_to_point_data().contour(isosurfaces=[300], scalars="HU")
    # connectivity() labels each connected region; extract_largest() keeps only the skull
    return surface.connectivity(extraction_mode="largest")


# ── Buffer A: color surface render ───────────────────────────────────────────

def render_color(grid: pv.ImageData, size: int = 512) -> np.ndarray:
    """
    Render the bone isosurface with Phong shading and warm bone-like colour.
    Surface rendering is far more reliable than GPU volume rendering in off-screen
    (headless) mode on macOS/Linux where the OpenGL context may be limited.
    """
    surface = extract_bone_surface(grid)

    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(
        surface,
        color      = "#e8dcc8",   # warm off-white — bone colour
        ambient    = 0.25,
        diffuse    = 0.70,
        specular   = 0.20,
        smooth_shading = True,
    )
    pl.add_light(pv.Light(position=(1, -1, 1), intensity=0.8))
    set_skull_camera(pl, grid)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── Buffer B: depth map ──────────────────────────────────────────────────────

def render_depth(grid: pv.ImageData, size: int = 512) -> np.ndarray:
    """
    Normalised depth map of the bone isosurface.
    Reads the OpenGL Z-buffer after rendering; background → 0, closest bone → 255.
    """
    surface = extract_bone_surface(grid)

    pl = pv.Plotter(off_screen=True, window_size=[size, size])
    pl.background_color = "black"
    pl.add_mesh(surface, color="white")
    set_skull_camera(pl, grid)
    pl.screenshot(return_img=False)          # triggers full render + Z-buffer fill
    depth_raw = pl.get_image_depth(fill_value=np.nan)

    valid = depth_raw[~np.isnan(depth_raw)]
    if valid.size == 0:
        pl.close()
        return np.zeros((size, size), dtype=np.uint8)

    # Invert so closer = brighter, normalise to [0, 255]
    depth_norm = (depth_raw - valid.min()) / (valid.max() - valid.min() + 1e-8)
    depth_inv  = 1.0 - depth_norm
    depth_inv[np.isnan(depth_raw)] = 0.0
    pl.close()
    return (depth_inv * 255).astype(np.uint8)


# ── Buffer C: Canny edges ────────────────────────────────────────────────────

def render_edges(color_img: np.ndarray) -> np.ndarray:
    """Canny edge map from the color render — highlights bone boundaries."""
    gray  = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    return edges


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render simple CT buffers for Phase 1.5")
    parser.add_argument("--volume", required=True, help="Path to preprocessed .nii.gz")
    parser.add_argument("--out",    default=None,  help="Output directory (default: data/renders/simple/<stem>/)")
    parser.add_argument("--size",   type=int, default=512, help="Render resolution (default 512)")
    args = parser.parse_args()

    vol_path = Path(args.volume)
    out_dir  = Path(args.out) if args.out else Path("data/renders/simple") / vol_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {vol_path.name} ...")
    grid = load_as_pyvista(vol_path)
    print(f"  Grid dimensions : {grid.dimensions}  spacing: {grid.spacing}")

    print("Rendering color buffer ...")
    color = render_color(grid, args.size)
    cv2.imwrite(str(out_dir / "color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    print("Rendering depth buffer ...")
    depth = render_depth(grid, args.size)
    cv2.imwrite(str(out_dir / "depth.png"), depth)

    print("Computing edge buffer ...")
    edges = render_edges(color)
    cv2.imwrite(str(out_dir / "edges.png"), edges)

    # Quick side-by-side preview
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(vol_path.stem, fontsize=11)
    axes[0].imshow(color);                    axes[0].set_title("Color (bone TF)"); axes[0].axis("off")
    axes[1].imshow(depth, cmap="gray");       axes[1].set_title("Depth");           axes[1].axis("off")
    axes[2].imshow(edges, cmap="gray");       axes[2].set_title("Edges (Canny)");   axes[2].axis("off")
    plt.tight_layout()
    preview_path = out_dir / "preview.png"
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved buffers → {out_dir}/")
    print(f"  color.png  depth.png  edges.png  preview.png")
    plt.show()


if __name__ == "__main__":
    main()
