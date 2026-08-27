"""
hoa_vessel_stats.py — measure vessel-tree statistics from Human Organ Atlas volumes

Purpose
-------
The vessel generator in synth_tissue_textures.py has hand-picked parameters: trunk
radius, taper ratio, split probability, segment length. I chose those by eye. This
script measures the same quantities from real HiP-CT data so they can be cited instead
of invented, and writes them as JSON that the synthesiser can consume.

    Human Organ Atlas, https://human-organ-atlas.esrf.fr — CC-BY-4.0
    Each dataset has its own DOI which MUST be cited if you use it.
    HOA paper: Sci. Adv. 12, eadz2240 (2026), DOI 10.1126/sciadv.adz2240

What HOA is, and what it is not
-------------------------------
HiP-CT is X-ray phase-contrast tomography. It is GREYSCALE — there is no colour
information at any resolution, so it cannot ground albedo. It grounds STRUCTURE only.
Organ colour has to come from somewhere else (Visible Korean sectioned images).

Two further caveats that matter for how the numbers are used:

  · Resolution mismatch. HOA whole-organ scans are 19-25 um. The render samples at
    roughly 0.65 mm/px, ~30x coarser. Structure finer than that is invisible in the
    output, so the useful product is STATISTICS (branching ratios, calibre
    distributions), not the voxel data itself. The 101 um tier (~655 MB for liver) is
    the right one to work with — its voxels are already close to render scale.

  · Internal vs subcapsular. Tomography resolves the whole vascular tree; what the
    renderer draws is the subcapsular network visible through the capsule. --depth_mm
    restricts the analysis to a shell near the organ surface so the measured calibres
    correspond to what is actually visible.

Getting the data
----------------
Downloads are NOT scriptable: the portal's ZIP links embed a per-session token
(ids.esrf.fr/ids/getData?sessionId=...) that expires, and the alternative is Globus,
which needs an account. So fetch by hand:

  1. https://human-organ-atlas.esrf.fr/explore/organ/liver
  2. pick a donor -> a dataset -> download the 101.0um ZIP (~655 MB)
  3. unzip somewhere, then point --input at the directory of slices

Usage
-----
    python scripts/hoa_vessel_stats.py --selftest
    python scripts/hoa_vessel_stats.py --input /path/to/liver_101um --organ liver \
        --voxel_um 101 --out data/hoa_stats

Run --selftest first. It builds a synthetic tree with KNOWN parameters and checks the
measurement recovers them, which tells you the pipeline works before you spend a
download on it.

Requires numpy, scipy, Pillow; scikit-image for skeletonisation (falls back to a
coarser estimator without it). JP2 slices need Pillow built with OpenJPEG, or glymur.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

try:
    from skimage.morphology import skeletonize
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


# ── Volume loading ────────────────────────────────────────────────────────────

def load_volume(path: Path, max_slices: int, downsample: int, z_contig: bool = False):
    """Load a slice stack (.jp2/.tif/.png) as a float32 volume, subsampled."""
    exts = (".jp2", ".tif", ".tiff", ".png")
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in exts)
    if not files:
        sys.exit(f"no slice images found in {path} (looked for {exts})")

    if z_contig:
        # CONSECUTIVE slices from the middle, so z spacing equals in-plane spacing
        # after --downsample: an isotropic sub-volume. Hessian vesselness assumes
        # isotropy, and subsampling 22:1 makes a tube non-tubular in voxel space.
        step = max(1, int(downsample))
        mid = len(files) // 2
        half = (max_slices * step) // 2
        picked = files[max(0, mid - half): mid + half: step][:max_slices]
    else:
        step = max(1, len(files) // max_slices)
        picked = files[::step][:max_slices]
    print(f"  {len(files)} slices found, loading {len(picked)} (every {step})")

    from PIL import Image
    out = []
    for i, f in enumerate(picked):
        try:
            im = Image.open(f)
        except Exception as e:
            sys.exit(f"could not open {f.name}: {e}\n"
                     "JP2 needs Pillow with OpenJPEG support, or convert to TIFF first.")
        if downsample > 1:
            im = im.resize((im.width // downsample, im.height // downsample),
                           Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32))
        if i % 25 == 0:
            print(f"    {i}/{len(picked)}", end="\r", flush=True)
    vol = np.stack(out)
    print(f"  volume {vol.shape}  range {vol.min():.1f}..{vol.max():.1f}      ")
    return vol, step


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_tissue_and_vessels(vol, invert=False, bg_sigma_vox=12.0, k_sigma=1.1,
                               max_radius_mm=0.0, voxel_mm_for_cap=1.0,
                               organ_close=0, roi_erode=2, vessel_polarity=None,
                               roi_two_stage=False, roi2_pct=0.0,
                               roi2_texture=False, roi2_win=9,
                               roi_margin_mm=0.0, cap_spacing=None, fill_lumen=False):
    """Split into (ROI mask, vessel mask).

    Two decisions here are INDEPENDENT and conflating them into one --invert flag is
    what broke the hollow organs:

      1. How to find the region of interest.
      2. Whether vessels are darker or brighter than their surroundings inside it.

    ROI. Take the largest bright connected component, close it, fill it, erode it. The
    same operation does the right thing in both regimes: for a solid organ (liver,
    kidney) the largest bright component IS the organ, so this yields the organ; for an
    air-filled organ in a mount (lung) the largest bright component is the CONTAINER
    RING, and filling it yields the container interior — which contains the lung. Either
    way the vessels end up inside the ROI, which is all the measurement needs.
    --organ_close bridges gaps first, for sparse tissue that is not simply connected.

    Polarity. Liver and kidney parenchyma is bright with dark lumina. Lung parenchyma is
    air, so its vessels and airways are the BRIGHT structures. Set --vessel bright/dark
    explicitly rather than inferring it.

    Detection is by LOCAL contrast against a smoothed background, not a global
    threshold: these volumes have a slow intensity gradient across the parenchyma, and
    a global split lands on that gradient — it produced a "vessel" mask that was one
    blob in the middle of the liver, with entirely plausible-looking statistics.
    """
    finite = vol[np.isfinite(vol)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    v = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    # ── ROI ──
    t_organ = _otsu(v.ravel()[::7])
    solid = v > t_organ
    if organ_close:
        solid = ndi.binary_closing(solid, iterations=int(organ_close))
    lab, n = ndi.label(solid)
    if n == 0:
        sys.exit("no connected component found")
    sizes = ndi.sum(solid, lab, range(1, n + 1))
    organ = lab == (int(np.argmax(sizes)) + 1)
    # Fill PER SLICE, not in 3D. The mounting container is an open-ended cylinder, so a
    # 3D fill leaks out of both ends and returns the shell itself — which is how the
    # lung ROI came out empty. In 2D each slice's ring is a closed curve and fills
    # correctly. Solid organs are unaffected: a filled blob stays a filled blob.
    organ = np.stack([ndi.binary_fill_holes(organ[z]) for z in range(organ.shape[0])])
    if roi_erode:
        organ = ndi.binary_erosion(organ, iterations=int(roi_erode))

    # Pull the ROI in by a fixed PHYSICAL margin. This is the fix for the mounted
    # organs: the container wall is the single highest-contrast structure in the
    # volume, so it is also the brightest thing in a local-variance map, and every
    # threshold-based attempt to find the organ selected the ring instead of the lung.
    # It cannot be out-thresholded — but it is always at the ROI boundary, so removing
    # a shell of known thickness removes it deterministically. A few millimetres also
    # discards the sub-pleural halo where the local-background estimate is unreliable.
    if roi_margin_mm > 0:
        # Per-slice (2D), not 3D. The container is a cylinder so the margin is an
        # in-plane quantity, and a 3D transform is bounded by the subsampled z-extent
        # — with slices 1.86 mm apart the 3D distance maxes out at ~4 mm, so an 8 mm
        # margin deleted the entire ROI.
        d_in = np.stack([ndi.distance_transform_edt(organ[z]) for z in
                         range(organ.shape[0])]) * voxel_mm_for_cap
        organ = organ & (d_in > roi_margin_mm)

    # Optional second stage: the first pass returns the CONTAINER interior for a
    # mounted organ, which is right for finding vessels but wrong for "how deep below
    # the organ surface is this". Re-running the same largest-component-fill on tissue
    # inside that ROI yields the ORGAN envelope, so --depth_mm then measures depth
    # below the organ surface rather than below the container wall. Heavy closing is
    # needed because lung tissue is sparse and not simply connected.
    if roi_two_stage:
        inside_v = v[organ]
        # Otsu fails for an air-filled organ: lung is ~70 % air by volume, so the
        # between-class split lands up at the bright hilar structures and the "organ"
        # comes out as a few small blobs — which is why the lung ROI silently stayed
        # the container ring. A percentile cut is the robust choice when the two
        # populations are that lopsided; roi2_pct is the air fraction to discard.
        if roi2_texture:
            # INTENSITY CANNOT FIND THE LUNG BOUNDARY. Air inside the lung and air
            # outside it are the same value, and the septa sit barely above both:
            # measured p90 = 0.102 while p96 = 0.872, the latter being the container
            # wall. Every intensity threshold therefore returns either the whole
            # container interior or nothing.
            #
            # Texture does separate them. Lung parenchyma is septa alternating with
            # air, so its LOCAL VARIANCE is high; the mounting medium around it is
            # smooth. Threshold local standard deviation, then close so the septa
            # merge into a solid body and the fill gives the pleural envelope.
            w = max(int(roi2_win), 3)
            m1 = ndi.uniform_filter(v, size=w)
            m2 = ndi.uniform_filter(v * v, size=w)
            sd_local = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
            t2 = float(np.percentile(sd_local[organ], roi2_pct if roi2_pct > 0 else 70))
            tis = (sd_local > t2) & organ
        elif roi2_pct > 0:
            t2 = float(np.percentile(inside_v, roi2_pct))
            tis = (v > t2) & organ
        else:
            t2 = _otsu(inside_v[::5]) if inside_v.size > 100 else 0.5
            tis = (v > t2) & organ
        tis = ndi.binary_closing(tis, iterations=max(int(organ_close), 4))
        lab2, n2 = ndi.label(tis)
        if n2:
            sz2 = ndi.sum(tis, lab2, range(1, n2 + 1))
            env = lab2 == (int(np.argmax(sz2)) + 1)
            env = np.stack([ndi.binary_fill_holes(env[z]) for z in range(env.shape[0])])
            print(f"  stage-2 thr {t2:.3f} -> envelope {env.mean():.3f}")
            print(f"  stage-2 thr {t2:.4f} -> envelope {env.mean():.3f}")
            if env.mean() > 0.01:
                organ = env
                print(f"  two-stage ROI -> organ envelope {organ.mean():.3f} of volume")

    # ── vessels by local contrast ──
    pol = vessel_polarity or ("bright" if invert else "dark")
    sig = max(int(round(bg_sigma_vox)), 3)
    om = organ.astype(np.float32)
    local_bg = ndi.gaussian_filter(v * om, sig) / (ndi.gaussian_filter(om, sig) + 1e-6)
    dev = v - local_bg
    sd = float(dev[organ].std())
    thr = (k_sigma * sd) if pol == "bright" else (-k_sigma * sd)
    vessels = (dev > thr) if pol == "bright" else (dev < thr)
    vessels &= organ
    vessels = ndi.binary_opening(vessels, iterations=1)

    # Fill tube cross-sections. With --vessel bright the detector finds the WALL of a
    # large vessel or bronchus, not the whole tube: the lumen inside is dark. A
    # distance transform on a ring returns half the WALL THICKNESS, not the vessel
    # radius — so the raw mask mixes true radii (small vessels, which are solid bright
    # dots in cross-section) with wall half-thicknesses (large ones). Filling per slice
    # turns each ring into the full cross-section, which is also the quantity the
    # renderer needs: the visible outer width of the vessel, not its wall.
    if fill_lumen:
        vessels = np.stack([ndi.binary_fill_holes(vessels[z])
                            for z in range(vessels.shape[0])])

    # Hollow organs need a calibre cap: heart chambers and colon lumen are large
    # low-attenuation regions inside the ROI and would be measured as giant "vessels",
    # so the radius statistics would describe a chamber rather than a coronary.
    if max_radius_mm and max_radius_mm > 0:
        dist_mm = ndi.distance_transform_edt(vessels, sampling=cap_spacing)
        lab2, n2 = ndi.label(vessels)
        if n2:
            peak = np.asarray(ndi.maximum(dist_mm, lab2, range(1, n2 + 1)))
            drop = np.zeros(n2 + 1, bool)
            drop[1:] = peak > max_radius_mm
            removed = int(drop[1:].sum())
            vessels = vessels & ~drop[lab2]
            if removed:
                print(f"  calibre cap {max_radius_mm} mm removed {removed} component(s)")
    return organ, vessels, float(thr)


def write_debug_png(vol, organ, vessels, path):
    """Overlay of the mid slice: organ outline green, vessels red.

    Numbers alone hid a 60 % vessel fraction as if it were plausible. Look at the mask.
    """
    from PIL import Image
    z = vol.shape[0] // 2
    a = vol[z].astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([g, g, g], -1)
    # Dim what is EXCLUDED rather than only outlining what is included — an outline
    # around the container looked identical to an outline around the organ, which hid
    # several bad ROIs behind confident-looking numbers.
    rgb[~organ[z]] *= 0.25
    edge = organ[z] ^ ndi.binary_erosion(organ[z], iterations=2)
    rgb[vessels[z]] = [1.0, 0.15, 0.15]
    rgb[edge] = [0.2, 1.0, 0.2]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def _otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    m = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    mt = m[-1]
    denom = w0 * (1 - w0)
    denom[denom == 0] = 1e-12
    var_b = (mt * w0 - m) ** 2 / denom
    return float((edges[:-1] + edges[1:])[np.argmax(var_b)] / 2)


def surface_shell(tissue, depth_mm, voxel_mm, spacing=None):
    """Voxels within depth_mm of the organ surface — the subcapsular region."""
    if depth_mm <= 0:
        return tissue
    d = ndi.distance_transform_edt(tissue, sampling=spacing)
    return tissue & (d <= depth_mm)


# ── Measurement ───────────────────────────────────────────────────────────────

def vessel_statistics(vessels, voxel_mm, spacing=None):
    """Radii from the distance transform, topology from the skeleton.

    `spacing` is the PHYSICAL voxel size (z, y, x) in mm. It has to be passed: slices
    are subsampled on load, so the volume is strongly anisotropic — 1.86 mm between
    slices against 0.169 mm in plane at typical lung settings, an 11:1 ratio. An
    unweighted distance transform silently treats those as equal and every radius,
    depth and skeleton length comes out wrong.
    """
    stats = {}
    dist_mm = ndi.distance_transform_edt(vessels, sampling=spacing)

    if HAVE_SKIMAGE:
        skel = skeletonize(vessels)
        # a voxel's neighbour count on the skeleton classifies it:
        # 1 = endpoint, 2 = along a segment, >=3 = bifurcation
        k = np.ones((3, 3, 3), np.uint8); k[1, 1, 1] = 0
        nb = ndi.convolve(skel.astype(np.uint8), k, mode="constant")
        nb = np.where(skel, nb, 0)
        n_branch = int(np.sum(nb >= 3))
        n_end = int(np.sum(nb == 1))
        n_seg = int(np.sum(skel))
        radii = dist_mm[skel]
        stats["skeleton_voxels"] = n_seg
        stats["branch_points"] = n_branch
        stats["end_points"] = n_end
        # mean centreline length between bifurcations, in mm
        if n_branch > 0:
            stats["segment_length_mm"] = round(n_seg * voxel_mm / max(n_branch, 1), 4)
        # Murray's law: r_parent^3 = sum r_daughter^3 -> symmetric taper = 2^(-1/3)
        stats["taper_murray_symmetric"] = round(2.0 ** (-1.0 / 3.0), 4)
    else:
        radii = dist_mm[vessels]
        stats["note"] = "scikit-image absent — topology skipped, radii only"

    radii = radii[radii > 0]
    if radii.size:
        for q in (5, 25, 50, 75, 95, 99):
            stats[f"radius_p{q}_mm"] = round(float(np.percentile(radii, q)), 4)
        stats["radius_mean_mm"] = round(float(radii.mean()), 4)
        stats["radius_max_mm"] = round(float(radii.max()), 4)
    stats["vessel_volume_fraction"] = round(float(vessels.mean()), 5)
    return stats


def vesselness_radii(vol, roi, voxel_mm, radii_mm, dark, sigma_to_radius,
                     resp_pct=99.0, use_otsu=True, use_hyst=True,
                     hyst_lo=96.0, hyst_hi=99.5, filt="sato"):
    """Calibre by SCALE SELECTION — the principled alternative to segment-then-measure.

    A Hessian ridge filter responds to tubular structure as a whole object at scale
    sigma, which matters for two reasons:

      · It does not care whether the wall or the lumen is bright. That is exactly the
        failure that made the lung numbers meaningless: a bright-wall detector returns
        broken crescents, and a distance transform on a crescent measures arc
        thickness, not vessel radius.
      · The sigma that MAXIMISES the response is proportional to tube radius, so
        calibre is read off directly from which scale won rather than inferred from a
        segmentation that may not correspond to the vessel at all.

    sigma_to_radius is CALIBRATED on synthetic tubes of known radius by --selftest,
    not assumed: the constant depends on the filter's normalisation convention.
    """
    from skimage.filters import sato, frangi, apply_hysteresis_threshold
    sigmas = [max(r / (sigma_to_radius * voxel_mm), 0.6) for r in radii_mm]
    best = which = None
    for i, sg in enumerate(sigmas):
        if filt == "frangi":
            # Frangi suppresses PLATE-like structure explicitly via its eigenvalue
            # ratios; Sato does not and responds happily to sheets. The heart's
            # myocardial trabeculae and the colon's layered wall ARE sheets, which is
            # why sato flagged 2.7 % / 0.85 % of those volumes as "vessel".
            r = np.asarray(frangi(vol, sigmas=[sg], black_ridges=dark), np.float32)
        else:
            r = np.asarray(sato(vol, sigmas=[sg], black_ridges=dark), np.float32)
        # NORMALISE EACH SCALE against its own response distribution inside the ROI.
        # Raw Sato response grows with structure size, so an unnormalised max-over-
        # scales is won by the largest vessels everywhere and the small ones never
        # compete. That is a selection bias, not anatomy: it is why the liver read
        # p50 = 2.4 mm (only the few biggest vessels cleared the threshold) while
        # visibly detecting three blobs in a slice that contains dozens of vessels.
        ref = float(np.percentile(r[roi], 99.5)) if roi.any() else 1.0
        r = r / max(ref, 1e-12)
        if best is None:
            best, which = r, np.zeros(r.shape, np.uint8)
        else:
            m = r > best
            best = np.where(m, r, best)
            which = np.where(m, np.uint8(i), which)
    inside = best[roi]
    if inside.size == 0:
        return {}, np.zeros_like(roi)
    # Threshold on the response's own structure (Otsu) rather than a fixed percentile.
    # A percentile fixes the detected COUNT in advance, which is exactly the wrong
    # thing to fix when the question is "how many vessels are there".
    # HYSTERESIS, not a single threshold. Vessels are <1 % of voxels, so Otsu — which
    # maximises between-class variance and implicitly assumes comparable class sizes —
    # cuts far too high and keeps only trunks. Seeding on strong response and growing
    # into CONNECTED weaker response is the standard remedy for thin branching
    # structure: it recovers distal branches that any single global threshold drops.
    if use_hyst:
        hi = float(np.percentile(inside, hyst_hi))
        lo = float(np.percentile(inside, hyst_lo))
        mask = apply_hysteresis_threshold(best, lo, hi) & roi
        thr = lo
    else:
        thr = _otsu(inside[::max(1, inside.size // 200000)]) if use_otsu \
            else float(np.percentile(inside, resp_pct))
        mask = (best >= thr) & roi
    sel = which[mask]
    rad = np.array([radii_mm[i] for i in sel], np.float32) if sel.size else np.array([])
    st = {"method": "vesselness (Sato, scale selection)",
          "response_threshold_pct": resp_pct,
          "detected_fraction": round(float(mask.mean()), 6),
          "scales_mm": [round(r, 3) for r in radii_mm]}
    if rad.size:
        for q in (5, 25, 50, 75, 95, 99):
            st[f"radius_p{q}_mm"] = round(float(np.percentile(rad, q)), 4)
        st["radius_mean_mm"] = round(float(rad.mean()), 4)
        st["radius_max_mm"] = round(float(rad.max()), 4)
        st["scale_histogram"] = {str(round(r, 3)): int((rad == np.float32(r)).sum())
                                 for r in radii_mm}
    return st, mask


def to_generator_params(stats, voxel_mm):
    """Map measurements onto vessel_tree() arguments in synth_tissue_textures.py."""
    p = {}
    if "radius_p99_mm" in stats:
        # trunk half-width; the generator's start_w is a radius in mm
        p["start_w_mm"] = round(stats["radius_p99_mm"], 3)
    if "segment_length_mm" in stats:
        p["step_mm"] = round(stats["segment_length_mm"], 3)
    p["taper"] = stats.get("taper_murray_symmetric", 0.794)
    if stats.get("branch_points") and stats.get("end_points"):
        # each bifurcation adds one terminal; fraction of nodes that bifurcate
        tot = stats["branch_points"] + stats["end_points"]
        p["split"] = round(stats["branch_points"] / max(tot, 1), 3)
    return p


# ── Self-test ─────────────────────────────────────────────────────────────────

def calibrate_sigma_to_radius(voxel_mm=0.169, dark=False):
    """Measure the sigma -> radius constant on synthetic tubes of KNOWN radius.

    Sato's filter peaks at a sigma proportional to the tube radius, but the constant
    depends on the normalisation convention, so it is measured here rather than taken
    from the paper. Straight cylinders of several radii are rendered into a volume,
    the filter is swept over sigma, and the peak-response sigma is regressed against
    the true radius.
    """
    from skimage.filters import sato, frangi, apply_hysteresis_threshold
    N = 96
    true_r_mm = [0.25, 0.4, 0.6, 0.9]
    ratios = []
    for R in true_r_mm:
        vol = np.zeros((N, N, N), np.float32) + (1.0 if dark else 0.0)
        rr = R / voxel_mm
        zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
        tube = np.hypot(yy - N / 2, xx - N / 2) <= rr        # axis along z
        vol[tube] = 0.0 if dark else 1.0
        vol = ndi.gaussian_filter(vol, 0.8)
        sig_grid = np.linspace(0.6, max(rr * 2.5, 3.0), 18)
        resp = []
        for sg in sig_grid:
            r = np.asarray(sato(vol, sigmas=[sg], black_ridges=dark), np.float32)
            resp.append(float(r[:, N // 2, N // 2].mean()))     # on the tube axis
        best_sigma = float(sig_grid[int(np.argmax(resp))])
        ratios.append((R / voxel_mm) / max(best_sigma, 1e-6))
        print(f"  true R = {R:.2f} mm ({rr:5.2f} vox)   peak sigma = {best_sigma:5.2f} "
              f"vox   R/sigma = {ratios[-1]:.3f}")
    k = float(np.median(ratios))
    print(f"\n  sigma_to_radius = {k:.3f}   (radius = {k:.3f} x sigma x voxel_mm)")
    return k


def _selftest():
    """Synthesise a tree with KNOWN parameters, then check we measure them back."""
    print("building a synthetic vessel tree with known parameters ...")
    N = 160
    voxel_mm = 0.1
    TRUE_TRUNK_MM, TRUE_TAPER, TRUE_STEP_MM = 0.9, 0.79, 2.0
    vol = np.zeros((N, N, N), np.float32) + 0.5          # parenchyma
    vol[:4] = vol[-4:] = 0.0                              # background slabs
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]

    def draw(p, d, r_mm, depth):
        if depth == 0 or r_mm < 0.15:
            return
        d = d / np.linalg.norm(d)
        steps = int(TRUE_STEP_MM / voxel_mm)
        for _ in range(steps):
            p = p + d
            if not (0 <= p[0] < N and 0 <= p[1] < N and 0 <= p[2] < N):
                return
            rr = r_mm / voxel_mm
            m = ((zz - p[0])**2 + (yy - p[1])**2 + (xx - p[2])**2) <= rr*rr
            vol[m] = 0.05                                  # lumen: dark
        for s in (-1, 1):
            nd = d + s * 0.5 * rng.normal(size=3)
            draw(p.copy(), nd, r_mm * TRUE_TAPER, depth - 1)

    draw(np.array([N/2, N/2, 8.0]), np.array([0.0, 0.1, 1.0]), TRUE_TRUNK_MM, 5)

    tissue, vessels, thr = segment_tissue_and_vessels(vol)
    st = vessel_statistics(vessels, voxel_mm)
    gp = to_generator_params(st, voxel_mm)

    print(f"\n  otsu threshold {thr:.3f}   vessel volume fraction "
          f"{st['vessel_volume_fraction']:.4f}")
    print(f"\n  {'quantity':<22}{'true':>10}{'measured':>12}")
    print(f"  {'trunk radius (mm)':<22}{TRUE_TRUNK_MM:>10.2f}"
          f"{gp.get('start_w_mm', float('nan')):>12.2f}")
    print(f"  {'segment length (mm)':<22}{TRUE_STEP_MM:>10.2f}"
          f"{gp.get('step_mm', float('nan')):>12.2f}")
    print(f"  {'taper ratio':<22}{TRUE_TAPER:>10.2f}{gp.get('taper', float('nan')):>12.2f}")
    print("\n  radius percentiles (mm):",
          {k: v for k, v in st.items() if k.startswith("radius_p")})
    if not HAVE_SKIMAGE:
        print("\n  NOTE: scikit-image missing, topology not measured "
              "(pip install scikit-image)")
    print("\n  Interpretation: trunk radius should land near the true value. Segment\n"
          "  length is measured as skeleton length per bifurcation, so it reads long\n"
          "  when branches are long relative to the tree — treat it as an upper bound.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="directory of HOA slice images")
    ap.add_argument("--organ", default="liver")
    ap.add_argument("--voxel_um", type=float, default=101.0)
    ap.add_argument("--max_slices", type=int, default=300)
    ap.add_argument("--downsample", type=int, default=2,
                    help="in-plane downsample; 101um/2 ~ 0.2mm, close to render scale")
    ap.add_argument("--depth_mm", type=float, default=6.0,
                    help="restrict to this depth below the organ surface — the "
                         "subcapsular vessels are the ones the renderer draws. "
                         "0 = whole organ.")
    ap.add_argument("--invert", action="store_true",
                    help="use if vessels are BRIGHTER than parenchyma in your dataset")
    ap.add_argument("--out", default="data/hoa_stats")
    ap.add_argument("--vessel", default="", choices=["", "dark", "bright"],
                    help="are vessels darker or brighter than their surroundings? "
                         "dark for solid parenchyma (liver/kidney), bright for "
                         "air-filled lung. Independent of how the ROI is found.")
    ap.add_argument("--organ_close", type=int, default=0,
                    help="closing iterations before picking the ROI component — "
                         "bridges gaps in sparse tissue")
    ap.add_argument("--roi_two_stage", action="store_true",
                    help="after finding the container interior, find the ORGAN inside "
                         "it. Needed for mounted hollow organs so --depth_mm measures "
                         "depth below the organ surface, not the container wall.")
    ap.add_argument("--roi2_pct", type=float, default=0.0,
                    help="stage-2 threshold as a percentile of intensities inside the "
                         "container, instead of Otsu. Use for air-filled organs: ~65 "
                         "for lung, where Otsu is defeated by the air fraction.")
    ap.add_argument("--method", default="contrast", choices=["contrast", "vesselness"],
                    help="contrast = local-contrast segmentation then distance "
                         "transform. vesselness = Hessian ridge filter with scale "
                         "selection, which measures calibre directly and is immune to "
                         "the wall-vs-lumen problem. Forces isotropic slice loading.")
    ap.add_argument("--z_contig", action="store_true",
                    help="load consecutive slices (isotropic sub-volume) instead of "
                         "subsampling across the whole organ")
    ap.add_argument("--sigma_to_radius", type=float, default=1.0,
                    help="constant mapping filter sigma to tube radius; measured by "
                         "--selftest rather than assumed")
    ap.add_argument("--radii_mm", default="0.15,0.25,0.4,0.6,0.9,1.4,2.0",
                    help="candidate vessel radii in mm for scale selection")
    ap.add_argument("--filter", default="sato", choices=["sato", "frangi"],
                    help="ridge filter. frangi suppresses PLATE-like structure, which "
                         "sato does not — use it for hollow organs whose walls and "
                         "trabeculae are sheets sato mistakes for vessels.")
    ap.add_argument("--no_hyst", action="store_true",
                    help="disable hysteresis, use a single global threshold")
    ap.add_argument("--hyst_lo", type=float, default=96.0,
                    help="grow threshold, percentile of response inside the ROI")
    ap.add_argument("--hyst_hi", type=float, default=99.5,
                    help="seed threshold, percentile of response inside the ROI")
    ap.add_argument("--resp_fixed", action="store_true",
                    help="use the fixed --resp_pct percentile instead of Otsu on the "
                         "response. A percentile fixes the detected count in advance, "
                         "which biases the calibre distribution toward large vessels.")
    ap.add_argument("--resp_pct", type=float, default=99.0,
                    help="keep voxels above this percentile of vesselness response")
    ap.add_argument("--fill_lumen", action="store_true",
                    help="fill detected components per slice. REQUIRED with "
                         "--vessel bright: otherwise a large vessel is detected as its "
                         "wall only and its 'radius' is really half the wall thickness. "
                         "Small vessels are solid dots either way, so the raw mask "
                         "silently mixes two different quantities.")
    ap.add_argument("--roi_margin_mm", type=float, default=0.0,
                    help="erode the ROI inward by this many mm. Use ~6 for a mounted "
                         "organ: the container wall is the highest-contrast structure "
                         "present and cannot be out-thresholded, but it always sits at "
                         "the ROI boundary, so a fixed margin removes it for certain.")
    ap.add_argument("--roi2_texture", action="store_true",
                    help="find the stage-2 organ envelope by LOCAL VARIANCE instead of "
                         "intensity. Required for lung: intra- and extra-pulmonary air "
                         "are the same intensity so no threshold separates them, but "
                         "only the parenchyma is textured.")
    ap.add_argument("--roi2_win", type=int, default=9,
                    help="local-variance window in voxels (~1.5 mm at lung settings)")
    ap.add_argument("--roi_erode", type=int, default=2,
                    help="erode the ROI to drop the capsule / container wall")
    ap.add_argument("--max_radius_mm", type=float, default=0.0,
                    help="discard vessel components larger than this inscribed radius. "
                         "Required for hollow organs (heart chambers, colon lumen), "
                         "which are otherwise measured as giant vessels. 0 = off.")
    ap.add_argument("--bg_sigma", type=float, default=12.0,
                    help="voxel sigma of the local background estimate. Should be a few "
                         "times the largest vessel radius you want to catch.")
    ap.add_argument("--k_sigma", type=float, default=1.1,
                    help="vessel threshold in std-devs of the local residual. Lower "
                         "catches more (and more noise).")
    ap.add_argument("--debug_png", default="",
                    help="write a mid-slice overlay (organ outline green, vessels red). "
                         "Always check this — a wrong mask still produces confident numbers.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        print("\n--- vesselness scale calibration ---")
        calibrate_sigma_to_radius()
        return
    if not args.input:
        sys.exit("--input required (or use --selftest)")

    voxel_mm = args.voxel_um / 1000.0 * args.downsample
    print(f"HOA vessel statistics — {args.organ}, effective voxel {voxel_mm*1000:.0f} um")
    vol, zstep = load_volume(Path(args.input), args.max_slices, args.downsample,
                             args.method == 'vesselness' or args.z_contig)
    z_mm = args.voxel_um / 1000.0 * zstep
    spacing = (z_mm, voxel_mm, voxel_mm)
    print(f"  spacing (z,y,x) = {z_mm:.3f}, {voxel_mm:.3f}, {voxel_mm:.3f} mm"
          f"   anisotropy {z_mm/voxel_mm:.1f}:1")

    tissue, vessels, thr = segment_tissue_and_vessels(
        vol, args.invert, args.bg_sigma, args.k_sigma,
        args.max_radius_mm, voxel_mm, args.organ_close, args.roi_erode,
        args.vessel or None, args.roi_two_stage, args.roi2_pct,
        args.roi2_texture, args.roi2_win, args.roi_margin_mm, spacing,
        args.fill_lumen)
    print(f"  resid thr {thr:+.4f}   organ {tissue.mean():.3f}   "
          f"vessels {vessels.mean():.4f} of volume")
    if vessels.mean() > 0.25:
        print("  [warn] vessel fraction implausibly high — try --invert")

    if args.method == "vesselness":
        # Restrict to the subcapsular shell BEFORE filtering, not after: the renderer
        # draws the vessels visible through the capsule, and a whole-organ measurement
        # is dominated by the hilar/portal trunks, which are an order of magnitude
        # larger and would set the trunk calibre far too high.
        radii = [float(x) for x in args.radii_mm.split(",") if x.strip()]
        v_roi = surface_shell(tissue, args.depth_mm, voxel_mm, spacing)
        print(f"  vesselness ROI: {v_roi.mean():.4f} of volume "
              f"(depth {args.depth_mm} mm)")
        vstats, vmask = vesselness_radii(
            vol, v_roi, voxel_mm, radii,
            dark=((args.vessel or ("bright" if args.invert else "dark")) == "dark"),
            sigma_to_radius=args.sigma_to_radius, resp_pct=args.resp_pct,
            use_otsu=not args.resp_fixed, use_hyst=not args.no_hyst,
            hyst_lo=args.hyst_lo, hyst_hi=args.hyst_hi, filt=args.filter)
        vessels = vmask
        print(f"  vesselness: detected {vstats.get('detected_fraction', 0):.5f} "
              f"of volume, radius p50 {vstats.get('radius_p50_mm','-')} mm")

    if args.debug_png:
        # Draw the region ACTUALLY analysed. In vesselness mode that is the
        # subcapsular shell, not the whole organ — outlining the organ while
        # analysing a shell makes the central vessels look like missed detections
        # when they were deliberately excluded.
        shown_roi = v_roi if args.method == "vesselness" else tissue
        write_debug_png(vol, shown_roi, vessels, args.debug_png)
        print(f"  debug overlay -> {args.debug_png}")

    shell = surface_shell(tissue, args.depth_mm, voxel_mm, spacing)
    vessels_shell = vessels & shell
    print(f"  subcapsular shell ({args.depth_mm} mm): "
          f"{vessels_shell.mean():.4f} of volume")

    result = {
        "source": "Human Organ Atlas (HiP-CT), https://human-organ-atlas.esrf.fr",
        "licence": "CC-BY-4.0 — cite the dataset DOI",
        "organ": args.organ,
        "voxel_mm_effective": voxel_mm,
        "spacing_mm_zyx": [round(z_mm,4), round(voxel_mm,4), round(voxel_mm,4)],
        "depth_mm": args.depth_mm,
        "whole_organ": vessel_statistics(vessels, voxel_mm, spacing),
        "subcapsular": vessel_statistics(vessels_shell, voxel_mm, spacing),
    }
    if args.method == "vesselness":
        result["vesselness"] = vstats
    result["generator_params_subcapsular"] = to_generator_params(
        result["subcapsular"], voxel_mm)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.organ}_vessel_stats.json"
    f.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {f}")
    print(json.dumps(result["generator_params_subcapsular"], indent=2))
    print("\nFeed start_w_mm / step_mm / taper / split into vessel_tree() in "
          "synth_tissue_textures.py, and cite the dataset DOI.")


if __name__ == "__main__":
    main()
