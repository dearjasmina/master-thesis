"""
texture_learnability.py — can organ texture be learned from the EXISTING dataset?

Answers that question from the data, before any retraining. Every v25 organ is shaded
from exactly three image textures (albedo / normal / rough) in BOX projection at
--tex_mm; make_material() gates all procedural layers with `not has_albedo`, and every
organ class has all three maps. So the texture at a pixel is a deterministic function of
the object-space point — which we can only reconstruct from HALF-float depth.

Subcommands
-----------
  oracle        Box-project the real albedo texture at the reconstructed position and
                measure how well it predicts the fine detail in the actual GT. Includes
                a blind search over projection conventions (does it independently land
                on Cycles' own?), a half-tile offset null, and a position-noise
                dose-response. Optional --checkpoint also scores what the network misses.
  registration  Reconstruct the same surface point from two views and measure how far
                apart the two reconstructions land (mm), the transfer function for
                surface-locked sinusoids, and how well the real texture registers.
  spectrum      Mean power spectrum over many predicted images: a lattice produced by the
                generator's ConvTranspose2d layers is screen-locked and shows as sharp
                peaks at fixed pixel frequencies; a surface-locked lattice smears out.
  phase         (needs a gbuf_pos1024 checkpoint) Shift the position input and see
                whether the output lattice moves — proves encoding leakage directly.
  depth-survey  Distribution of the half-float depth step across the dataset.

Every subcommand prints a verdict line with the numbers it rests on.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from exr import read_render_exr                                   # noqa: E402
from dataset import unproject_position, CANONICAL_TISSUES, ORGAN_TO_GID  # noqa: E402
# The camera model and the BOX projection live in the training code and are imported
# here rather than copied: these tests must validate exactly what the dataloader feeds
# the network.
import camera_model as cm                                                # noqa: E402
from texture_lookup import (TEX_CLASSES, TEX_MM, BLEND, tex_class, box_weights,  # noqa: E402
                            AXIS_UV, CYCLES_FLIP_U, sample_wrap, project_axis, box_project)


# ── data ──────────────────────────────────────────────────────────────────────
_TEX = {}


def texture(tex_dir, cls, kind="albedo"):
    key = (str(tex_dir), cls, kind)
    if key not in _TEX:
        a = cv2.imread(str(Path(tex_dir) / f"{cls}_{kind}.png"), cv2.IMREAD_UNCHANGED)
        if a is not None:
            a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
            if a.ndim == 3:
                a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB)
        _TEX[key] = a
    return _TEX[key]


def lum(rgb):
    if rgb.ndim == 2:
        return rgb
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def subject_lut(subj_dir):
    p = Path(subj_dir) / "tissue_ids.json"
    tissues = json.load(open(p)).get("tissues", {}) if p.exists() else {}
    lut = np.zeros(max(tissues.values(), default=0) + 1, dtype=np.int64)
    for name, pid in tissues.items():
        lut[int(pid)] = ORGAN_TO_GID.get(name, 0)
    return lut


# ── camera model: see scripts/training/camera_model.py for what meta.json omits ──
CAMERA_SHIFT = cm.CAMERA_SHIFT
EMISSION_GAIN = cm.EMISSION_GAIN
unproject_corrected = cm.unproject


LEGACY_UNPROJECT = False     # --legacy: reproduce the buffer gbuf_pos1024 was trained on


def load_view(vd):
    vd = Path(vd)
    meta = json.load(open(vd / "meta.json"))
    gt = cv2.cvtColor(cv2.imread(str(vd / "rgb_preview.png")), cv2.COLOR_BGR2RGB)
    Nraw = read_render_exr(str(vd / "normals.exr"), want=("image",))["image"][..., :3]
    Zraw = read_render_exr(str(vd / "depth.exr"), want=("image",))["image"][..., 0]
    Z = Zraw.astype(np.float64) / EMISSION_GAIN[0]            # true planar depth (m)
    P_corr = unproject_corrected(Z, meta)
    P_leg = unproject_position(vd, meta, 0)[..., :3].astype(np.float64) * 1000.0
    P = P_leg if LEGACY_UNPROJECT else P_corr
    N = (Nraw.astype(np.float64) if LEGACY_UNPROJECT else Nraw.astype(np.float64) / EMISSION_GAIN) * 2.0 - 1.0
    N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    pid = np.rint(read_render_exr(str(vd / "segid.exr"), want=("image",))["image"][..., 0])
    lut = subject_lut(vd.parent)
    gid = lut[np.clip(pid.astype(np.int64), 0, len(lut) - 1)]
    zf = Z[gid > 0]
    steps = np.diff(np.unique(zf)) if zf.size else np.array([])
    return dict(dir=vd, meta=meta, gt=gt.astype(np.float32) / 255.0, P=P, N=N,
                P_corr=P_corr, P_leg=P_leg, Z=Z, gid=gid,
                depth_step_mm=float(np.median(steps) * 1000.0) if steps.size else float("nan"))


def list_views(args):
    """--views (explicit dirs) or --subjects under --data-root."""
    if args.views:
        return [Path(v) for v in args.views]
    root = Path(args.data_root)
    subs = args.subjects or sorted(p.name for p in root.iterdir() if p.is_dir())
    subs = subs[: args.max_subjects] if args.max_subjects else subs
    out = []
    for s in subs:
        out += sorted(d for d in (root / s).glob("v*") if (d / "meta.json").exists())
    return out


# ── statistics ────────────────────────────────────────────────────────────────
def highpass(img, mask, sigma):
    """Mask-aware high-pass: subtract a normalised-convolution blur inside `mask`."""
    m = mask.astype(np.float64)
    low = cv2.GaussianBlur(img * m, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(m, (0, 0), sigma), 1e-6)
    return (img - low) * m


class Corr:
    """Streaming Pearson correlation."""
    def __init__(self):
        self.n = 0; self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
        self.n += x.size; self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    @property
    def r(self):
        if self.n < 3:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx ** 2 / self.n
        vy = self.syy - self.sy ** 2 / self.n
        return float(cov / np.sqrt(max(vx * vy, 1e-30)))


def erode(mask, px):
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


# ── oracle ────────────────────────────────────────────────────────────────────
def predict_views(views, ckpt, preset_name, device_name):
    """Run a trained generator on each view (for scoring what the network misses)."""
    import torch
    sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
    from config import preset
    from networks import define_G
    import infer as I
    cfg = preset(preset_name)
    dev = torch.device(device_name)
    G = define_G(cfg).to(dev).eval()
    ck = torch.load(ckpt, map_location="cpu")
    G.load_state_dict(ck["G"] if "G" in ck else ck)
    out = {}
    for vd in views:
        x = I.load_input(vd, cfg, I.build_segid_lut(vd.parent))
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            y = G(t)[0].float().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)
        out[str(vd)] = (y + 1.0) * 0.5
    return out


def cmd_oracle(args):
    views = list_views(args)
    print(f"oracle: {len(views)} views  tex_dir={args.tex_dir}  tex_mm={args.tex_mm}  "
          f"highpass sigma={args.sigma}px")
    preds = (predict_views(views, args.checkpoint, args.preset, args.device)
             if args.checkpoint else {})

    # What to compare. The vessel network lives ONLY in the tangent-space normal map
    # (the albedo tile is soft mottling), and it appears in the GT as thin specular
    # streaks. So for `ridge` the texture signal is the normal map's deviation from flat
    # and the GT signal is local streak energy |high-pass|; both have their local mean
    # removed so only line placement is scored, not lighting or density.
    if args.signal == "ridge":
        def tex_scalar(cls):
            t = texture(args.tex_dir, cls, "normal")
            return None if t is None else np.sqrt((t[..., 0] - 0.5) ** 2 + (t[..., 1] - 0.5) ** 2).astype(np.float64)
        gstat = lambda Y, m: highpass(np.abs(highpass(Y, m, args.sigma)), m, args.band)
        tstat = lambda img, m: highpass(img, m, args.band)
    else:
        def tex_scalar(cls):
            t = texture(args.tex_dir, cls, "albedo")
            return None if t is None else lum(t).astype(np.float64)
        gstat = lambda Y, m: highpass(Y, m, args.sigma)
        tstat = lambda img, m: highpass(img, m, args.sigma)
    print(f"signal: {args.signal}")

    variants = [(s, fu, fv) for s in (0, 1) for fu in (0, 1) for fv in (0, 1)]
    search = defaultdict(Corr)            # (cls, axis, sign, variant) -> Corr
    cyc = defaultdict(Corr)               # (cls, tag) -> Corr    tag in {cycles, null, noiseX, pred}
    steps = []
    rng = np.random.default_rng(0)

    for i, vd in enumerate(views):
        V = load_view(vd)
        steps.append(V["depth_step_mm"])
        Ygt = lum(V["gt"]).astype(np.float64)
        Ypr = lum(preds[str(vd)]).astype(np.float64) if preds else None
        w = box_weights(V["N"])
        for gid in np.unique(V["gid"]):
            if gid <= 0:
                continue
            organ = CANONICAL_TISSUES[gid - 1]
            cls = tex_class(organ)
            T = tex_scalar(cls) if cls else None
            if T is None:
                continue
            organ_m = V["gid"] == gid
            ev = erode(organ_m, args.erode)
            if ev.sum() < 400:
                continue
            gt_hf = gstat(Ygt, organ_m)
            P, N = V["P"], V["N"]
            idx = np.nonzero(organ_m)

            # (1) blind convention search on single-texture regions
            for a in range(3):
                dom = ev & (w[..., a] >= 0.99)
                if dom.sum() < 200:
                    continue
                sgn = np.sign(N[..., a])
                for vi, (s, fu, fv) in enumerate(variants):
                    img = np.zeros_like(Ygt)
                    img[idx] = project_axis(T, P[idx], N[idx], a, s, fu, fv, tex_mm=args.tex_mm)
                    hf = tstat(img, organ_m)
                    for sg in (-1, 1):
                        sub = dom & (sgn == sg)
                        if sub.sum() >= 100:
                            search[(cls, a, sg, vi)].add(hf[sub], gt_hf[sub])

            # (2) Cycles-exact full projection + controls
            def score(tag, Pmm):
                img = np.zeros_like(Ygt)
                img[idx] = box_project(T, Pmm[idx], N[idx], tex_mm=args.tex_mm)
                hf = tstat(img, organ_m)
                cyc[(cls, tag)].add(hf[ev], gt_hf[ev])
                return hf
            hf_c = score("cycles", P)
            score("null+half_tile", P + args.tex_mm / 2.0)
            for s_add in args.noise_mm:
                score(f"noise{s_add:g}mm", P + rng.normal(0, s_add, P.shape))
            if Ypr is not None:
                res_hf = gstat(Ygt - Ypr, organ_m)
                cyc[(cls, "vs_residual")].add(hf_c[ev], res_hf[ev])
                pr_hf = gstat(Ypr, organ_m)
                cyc[(cls, "vs_prediction")].add(hf_c[ev], pr_hf[ev])
        print(f"  [{i+1}/{len(views)}] {vd.parent.name}/{vd.name}  "
              f"depth step {V['depth_step_mm']:.2f} mm", flush=True)

    # ── report ──
    print(f"\ndepth step over these views: median {np.nanmedian(steps):.2f} mm "
          f"(range {np.nanmin(steps):.2f}-{np.nanmax(steps):.2f})")
    print("\n1) BLIND CONVENTION SEARCH (single-texture regions; 8 variants per axis/sign)")
    print("   'cyc' = the variant Cycles' own source predicts. A texture that is truly")
    print("   present and correctly registered gives one clear winner, and it should be cyc.")
    classes = sorted({k[0] for k in search})
    for cls in classes:
        for a in range(3):
            for sg in (-1, 1):
                rs = [search[(cls, a, sg, vi)].r if (cls, a, sg, vi) in search else np.nan
                      for vi in range(8)]
                if np.all(np.isnan(rs)):
                    continue
                n = search[(cls, a, sg, 0)].n
                cyc_vi = variants.index((0, int(CYCLES_FLIP_U[a](sg)), 0))
                order = np.argsort(np.nan_to_num(rs, nan=-9))[::-1]
                best, second = order[0], order[1]
                tag = "  <-- matches Cycles" if best == cyc_vi else "  (Cycles predicts v%d)" % cyc_vi
                print(f"   {cls:16} axis {'XYZ'[a]} sign {sg:+d} n={n:>7}  best v{best} r={rs[best]:+.3f}"
                      f"  2nd r={rs[second]:+.3f}  cyc r={rs[cyc_vi]:+.3f}{tag}")

    print("\n2) CYCLES-EXACT BOX PROJECTION vs GT fine detail (Pearson r of high-passed luminance)")
    tags = ["cycles", "null+half_tile"] + [f"noise{s:g}mm" for s in args.noise_mm]
    if preds:
        tags += ["vs_residual", "vs_prediction"]
    hdr = "   " + f"{'class':16}" + "".join(f"{t:>16}" for t in tags) + f"{'n':>9}"
    print(hdr)
    pooled = defaultdict(Corr)
    for cls in sorted({k[0] for k in cyc}):
        row = f"   {cls:16}"
        for t in tags:
            c = cyc.get((cls, t))
            row += f"{c.r:>+16.3f}" if c else f"{'-':>16}"
            if c:
                p = pooled[t]; p.n += c.n; p.sx += c.sx; p.sy += c.sy
                p.sxx += c.sxx; p.syy += c.syy; p.sxy += c.sxy
        row += f"{cyc[(cls, 'cycles')].n:>9}"
        print(row)
    print("   " + f"{'POOLED':16}" + "".join(f"{pooled[t].r:>+16.3f}" for t in tags))

    # Registration and predictive strength are separate questions. A tiny r can still be
    # a certain detection: the blind search picking Cycles' exact variant in every large
    # region, a zero null, and decay under added position noise cannot happen by chance.
    rc, rn = pooled["cycles"].r, pooled["null+half_tile"].r
    noise_r = [pooled[f"noise{s:g}mm"].r for s in sorted(args.noise_mm)]
    big = wins = 0
    for (cls_, a, sg, vi), c in search.items():
        if vi != 0 or c.n < 20000:
            continue
        rs = np.nan_to_num([search[(cls_, a, sg, k)].r if (cls_, a, sg, k) in search else np.nan
                            for k in range(8)], nan=-9)
        big += 1
        wins += int(int(np.argmax(rs)) == variants.index((0, int(CYCLES_FLIP_U[a](sg)), 0)))
    registered = (abs(rn) < 0.02 and rc > 0.02 and big > 0 and wins >= 0.8 * big
                  and (not noise_r or noise_r[-1] < 0.5 * rc))
    print(f"\nconvention search: Cycles' own variant wins {wins}/{big} single-texture regions "
          f"with n >= 20000 (chance 1/8 each)")
    print("position-noise decay: " + " -> ".join(f"{r:+.3f}" for r in [rc] + noise_r))
    if registered and rc >= 0.3:
        v = "texture is registered to the GT AND strongly predictive of its fine detail"
    elif registered:
        v = (f"texture IS registered to the GT (all controls pass) but only WEAKLY predictive "
             f"with this proxy (r^2 = {rc * rc * 100:.2f}% of fine-detail variance)")
    else:
        v = ("no registered texture signal — check geometry, tex_mm, and that these are the "
             "textures the dataset was rendered with")
    print(f"VERDICT oracle: r(cycles)={rc:+.3f}  r(null)={rn:+.3f} -> {v}")


# ── registration ──────────────────────────────────────────────────────────────
def project_to(Pmm, meta, W=1024, H=1024):
    """World mm -> ARRAY coordinates in another view, with the CORRECTED camera model.

    Always the verified model, never the legacy one: finding correspondences with a
    wrong-but-consistent camera makes that camera look self-consistent, because it
    reprojects its own reconstruction onto itself.
    """
    return cm.project(Pmm, meta, W, H)


def bilinear(img, u, v):
    H, W = img.shape[:2]
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx, fy = u - x0, v - y0
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def ridge_tiles(tex_dir, cls, tex_mm):
    """Vessel-line signal from the normal map, full and high-passed (finer than 1.5 mm)."""
    t = texture(tex_dir, cls, "normal")
    if t is None:
        return None, None
    T = np.sqrt((t[..., 0] - 0.5) ** 2 + (t[..., 1] - 0.5) ** 2).astype(np.float64)
    sig_tx = 1.5 * T.shape[0] / tex_mm
    pad = int(4 * sig_tx) + 1                 # tile is periodic: blur with wrap
    T_hf = T - cv2.GaussianBlur(np.pad(T, pad, mode="wrap"), (0, 0), sig_tx)[pad:-pad, pad:-pad]
    return T, T_hf


def cmd_registration(args):
    """Same surface point, two views: how far apart do the two coordinates land?

    Correspondences come from the corrected (mesh-verified) camera model and true depth.
    BOTH coordinate buffers are then read at exactly those matched pixels: the corrected
    one, and the legacy one gbuf_pos1024 was trained on. What a position-conditioned
    network needs is for one surface point to carry one coordinate in every view.
    """
    views = list_views(args)
    by_subj = defaultdict(list)
    for v in views:
        by_subj[v.parent.name].append(v)
    lambdas = [64, 32, 16, 8, 4, 2, 1, 0.5]
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(24, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * np.pi, 24)
    bufs = {"corrected": "P_corr", "legacy (gbuf_pos1024)": "P_leg"}
    st = {b: dict(d=[], sin={lam: Corr() for lam in lambdas}, tex=Corr(), tex_hf=Corr())
          for b in bufs}
    steps, n_pairs = [], 0

    for subj, vs in by_subj.items():
        V = [load_view(v) for v in vs]
        steps += [x["depth_step_mm"] for x in V]
        for i in range(len(V)):
            for j in range(len(V)):
                if i == j:
                    continue
                A, B = V[i], V[j]
                H, W = B["Z"].shape
                m = A["gid"] > 0
                u, v, Zp = project_to(A["P_corr"][m], B["meta"], W, H)
                ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (Zp > 0)
                if ok.sum() < args.min_overlap:
                    continue
                gA = A["gid"][m][ok]; uu, vv, zz = u[ok], v[ok], Zp[ok]
                x0 = np.floor(uu).astype(int); y0 = np.floor(vv).astype(int)
                same = np.ones_like(gA, bool)
                for dy in (0, 1):
                    for dx in (0, 1):
                        same &= B["gid"][y0 + dy, x0 + dx] == gA
                vis = same & (np.abs(bilinear(B["Z"], uu, vv) - zz) < args.depth_tol_mm / 1000.0)
                if vis.sum() < args.min_overlap:
                    continue
                n_pairs += 1
                sel = rng.choice(int(vis.sum()), size=min(int(vis.sum()), 60000), replace=False)
                us, vs_, gsel = uu[vis][sel], vv[vis][sel], gA[vis][sel]
                NA = A["N"][m][ok][vis][sel]; NB = bilinear(B["N"], us, vs_)
                for b, key in bufs.items():
                    PA = A[key][m][ok][vis][sel]
                    PB = bilinear(B[key], us, vs_)
                    st[b]["d"].append(np.linalg.norm(PA - PB, axis=1))
                    for lam in lambdas:
                        for dvec, ph in zip(dirs, phases):
                            st[b]["sin"][lam].add(np.sin(2 * np.pi * (PA @ dvec) / lam + ph),
                                                  np.sin(2 * np.pi * (PB @ dvec) / lam + ph))
                    for g in np.unique(gsel):
                        cls = tex_class(CANONICAL_TISSUES[g - 1])
                        T, T_hf = ridge_tiles(args.tex_dir, cls, args.tex_mm) if cls else (None, None)
                        if T is None:
                            continue
                        k = gsel == g
                        st[b]["tex"].add(box_project(T, PA[k], NA[k], tex_mm=args.tex_mm),
                                         box_project(T, PB[k], NB[k], tex_mm=args.tex_mm))
                        st[b]["tex_hf"].add(box_project(T_hf, PA[k], NA[k], tex_mm=args.tex_mm),
                                            box_project(T_hf, PB[k], NB[k], tex_mm=args.tex_mm))
        print(f"  {subj}: {len(vs)} views, pairs so far {n_pairs}", flush=True)

    if n_pairs == 0:
        raise SystemExit("no co-visible view pairs found")
    names = list(bufs)
    d = {b: np.concatenate(st[b]["d"]) for b in names}
    print(f"\ndepth step: median {np.nanmedian(steps):.2f} mm   view pairs: {n_pairs}   "
          f"matched points: {d[names[0]].size}")
    print("\n" + " " * 44 + "".join(f"{b:>24}" for b in names))
    for q, lab in ((50, "median"), (90, "p90"), (99, "p99")):
        print(f"   3D disagreement, same surface point, {lab:>6}" +
              "".join(f"{np.percentile(d[b], q):>21.2f} mm" for b in names))
    print("   cross-view r of a surface-locked sinusoid:")
    for lam in lambdas:
        print(f"      wavelength {lam:>5g} mm" + " " * 23 +
              "".join(f"{st[b]['sin'][lam].r:>+24.3f}" for b in names))
    print("   vessel texture (normal-map ridges), full   " + "".join(f"{st[b]['tex'].r:>+24.3f}" for b in names))
    print("   vessel texture, detail finer than 1.5 mm   " + "".join(f"{st[b]['tex_hf'].r:>+24.3f}" for b in names))
    for b in names:
        cut = [lam for lam in lambdas if st[b]["sin"][lam].r >= 0.5]
        print(f"VERDICT registration [{b}]: median disagreement {np.median(d[b]):.2f} mm; "
              f"surface detail registers (r>=0.5) down to {min(cut) if cut else '>64'} mm; "
              f"vessel ridges r={st[b]['tex'].r:+.3f} (fine {st[b]['tex_hf'].r:+.3f})")


# ── spectrum (screen-locked lattice) ──────────────────────────────────────────
def cmd_spectrum(args):
    P = args.patch
    win = np.outer(np.hanning(P), np.hanning(P))
    results, spectra = {}, {}
    for label, pattern in args.sets:
        import glob as _glob
        files = (sorted(Path(pattern).glob("*.png")) if Path(pattern).is_dir()
                 else sorted(Path(f) for f in _glob.glob(pattern)))
        files = files[: args.max_images]
        acc, n = np.zeros((P, P)), 0
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            if args.crop_panel is not None:              # grids: take one panel
                k, of = args.crop_panel
                w = im.shape[1] // of
                im = im[:, k * w:(k + 1) * w]
            Y = lum(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0)
            fg = erode(Y > 0.04, P // 2)
            ys, xs = np.nonzero(fg)
            for _ in range(args.patches_per_image):
                if ys.size == 0:
                    break
                c = np.random.randint(ys.size)
                y0, x0 = ys[c] - P // 2, xs[c] - P // 2
                if y0 < 0 or x0 < 0 or y0 + P > Y.shape[0] or x0 + P > Y.shape[1]:
                    continue
                patch = Y[y0:y0 + P, x0:x0 + P]
                patch = (patch - patch.mean()) * win
                acc += np.abs(np.fft.fftshift(np.fft.fft2(patch))) ** 2
                n += 1
        if n == 0:
            print(f"{label}: no patches"); continue
        S = acc / n
        c = P // 2
        def peak_ratio(fy, fx):
            yy, xx = c + int(round(fy * P)), c + int(round(fx * P))
            yy, xx = min(max(yy, 2), P - 3), min(max(xx, 2), P - 3)
            ring = np.concatenate([S[yy - 2:yy + 3, xx - 2:xx + 3].ravel()])
            ring = np.delete(ring, 12)
            return float(S[yy, xx] / max(np.median(ring), 1e-30))
        freqs = {"1/2 px (x)": (0, -0.5 + 1 / P), "1/2 px (y)": (-0.5 + 1 / P, 0),
                 "1/4 px (x)": (0, 0.25), "1/4 px (y)": (0.25, 0),
                 "1/8 px (x)": (0, 0.125), "1/8 px (y)": (0.125, 0),
                 "1/16 px (x)": (0, 0.0625), "1/16 px (y)": (0.0625, 0)}
        results[label] = {k: peak_ratio(*v) for k, v in freqs.items()}
        spectra[label] = S
        print(f"{label}: {n} patches from {len(files)} images")
    if args.save and spectra:
        tiles = []
        for label, S in spectra.items():
            L = np.log10(S + 1e-12)
            L = (L - np.percentile(L, 5)) / (np.ptp(np.percentile(L, [5, 99.9])) + 1e-9)
            t = cv2.resize(np.clip(L, 0, 1), (384, 384), interpolation=cv2.INTER_NEAREST)
            t = cv2.putText((t * 255).astype(np.uint8), label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            tiles.append(t)
        cv2.imwrite(args.save, np.concatenate(tiles, axis=1))
        print(f"mean log power spectra (DC at centre) -> {args.save}")
    print("\npeak / local-median power at the ConvTranspose2d stride frequencies "
          "(~1 = no peak; >>1 = screen-locked lattice)")
    labels = list(results)
    print("   " + f"{'frequency':14}" + "".join(f"{l:>18}" for l in labels))
    for k in next(iter(results.values())):
        print("   " + f"{k:14}" + "".join(f"{results[l][k]:>18.2f}" for l in labels))


# ── phase (encoding leakage) ──────────────────────────────────────────────────
def cmd_phase(args):
    import torch
    from config import preset
    from networks import define_G
    import dataset as D
    import infer as I
    cfg = preset(args.preset)
    dev = torch.device(args.device)
    G = define_G(cfg).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location="cpu")
    G.load_state_dict(ck["G"] if "G" in ck else ck)
    views = list_views(args)[: args.max_views]
    bands = cfg.data.position_bands
    finest = D.POSITION_BASE_LAMBDA_MM / (4.0 ** (bands - 1))
    orig = D.unproject_position
    shifts = {"0": 0.0, f"+{finest/2:g}mm (finest band sign-flip)": finest / 2,
              f"+{D.POSITION_BASE_LAMBDA_MM:g}mm (all bands identical)": D.POSITION_BASE_LAMBDA_MM}
    stats = defaultdict(list)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for vd in views:
        outs = {}
        for tag, sh in shifts.items():
            def shifted(d, meta, b, _sh=sh):
                x = orig(d, meta, b)
                fg = np.any(x != 0, axis=-1, keepdims=True)
                # rebuild from the raw coordinate so the bands see the shift consistently
                world = x[..., :3] + _sh / 1000.0
                planes = [np.clip(world, -2.0, 2.0)]
                mm = world * 1000.0
                for k in range(int(b)):
                    w = 2.0 * np.pi / (D.POSITION_BASE_LAMBDA_MM / (4.0 ** k))
                    planes += [np.sin(mm * w), np.cos(mm * w)]
                return (np.concatenate(planes, axis=-1) * fg).astype(np.float32)
            I.unproject_position = shifted
            x = I.load_input(vd, cfg, I.build_segid_lut(vd.parent))
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                outs[tag] = ((G(t)[0].float().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0) + 1) * 0.5)
        I.unproject_position = orig
        base = outs["0"]
        fg = lum(base) > 0.04
        for tag in list(shifts)[1:]:
            diff = np.abs(outs[tag] - base).mean(-1)
            stats[tag].append(float(diff[fg].mean() * 255))
        k = list(shifts)
        vis = np.concatenate([base, outs[k[1]], np.clip(np.abs(outs[k[1]] - base) * 8, 0, 1),
                              outs[k[2]], np.clip(np.abs(outs[k[2]] - base) * 8, 0, 1)], axis=1)
        cv2.imwrite(str(out_dir / f"{vd.parent.name}_{vd.name}_phase.png"),
                    cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"  {vd.parent.name}/{vd.name}: " + "  ".join(
            f"{t}: {stats[t][-1]:.2f}" for t in list(shifts)[1:]), flush=True)
    print("\nmean |output change| inside organs (0-255 scale):")
    for t, v in stats.items():
        print(f"   {t:40} {np.mean(v):.2f}")
    a, b = [np.mean(stats[t]) for t in list(shifts)[1:]]
    print(f"\nVERDICT phase: half-period shift changes output {a/max(b,1e-9):.1f}x more than a "
          f"shift that leaves the bands identical -> "
          + ("the lattice IS the Fourier encoding" if a > 3 * b else
             "output is NOT dominated by the encoding phase"))
    print(f"panels written to {out_dir}  (orig | half-period | diff x8 | band-identical | diff x8)")


# ── glint (can the vessel glints be COMPUTED from the dataset?) ────────────────
_BBOX = {}


def mesh_bbox(mesh_dir, subject, organ):
    """Blender's auto texture space for an imported OBJ: its vertex bounds (centre, half-size)."""
    key = (str(mesh_dir), subject, organ)
    if key not in _BBOX:
        f = Path(mesh_dir) / subject / f"{organ}.obj"
        if not f.exists():
            _BBOX[key] = None
        else:
            v = np.array([l.split()[1:4] for l in open(f) if l.startswith("v ")], np.float64)
            lo, hi = v.min(0), v.max(0)
            _BBOX[key] = ((lo + hi) / 2.0, (hi - lo) / 2.0)
    return _BBOX[key]


def partial_r(r_xy, r_xz, r_yz):
    """corr(x, y | z) from pooled correlations over identical samples."""
    return (r_xy - r_xz * r_yz) / np.sqrt(max((1 - r_xz ** 2) * (1 - r_yz ** 2), 1e-12))


def cmd_glint(args):
    """Render the vessel glints analytically under each tangent-frame hypothesis and score
    them against the GT's thin streaks, over and above plain geometric highlights."""
    import glint_model as gm
    from texture_lookup import TextureBank
    tp = gm.tissue_params()
    bank = TextureBank(args.tex_dir, args.tex_mm)
    views = list_views(args)
    PAD = 24
    signs = {"++": (1, 1), "+-": (1, -1), "-+": (-1, 1), "--": (-1, -1)}
    C = defaultdict(Corr)          # (hyp, organ_key, "gt"|"flat") -> Corr ; (hyp, key, "null_gt"|"null_flat")
    print(f"glint: {len(views)} views, tangent families {list(gm.TANGENT_FAMILIES)}, "
          f"sign variants {list(signs)}, coat uses perturbed normal: no (C0) / yes (C1)")

    for i, vd in enumerate(views):
        V = load_view(vd)
        meta, gid = V["meta"], V["gid"]
        Vd, P_m = gm.camera_dirs(V["P"], meta)
        rig = gm.light_rig(meta)
        Y = lum(V["gt"]).astype(np.float64)
        subj = Path(vd).parent.name
        for g in np.unique(gid):
            if g <= 0:
                continue
            organ = CANONICAL_TISSUES[g - 1]
            cls = tex_class(organ)
            t = tp.get(organ)
            nimg, ns = bank.get(cls, "normal") if cls else (None, 0)
            rimg, rs = bank.get(cls, "rough") if cls else (None, 0)
            if t is None or nimg is None or rimg is None:
                continue
            m_full = gid == g
            if int(erode(m_full, 3).sum()) < 400:
                continue
            ys, xs = np.nonzero(m_full)
            sl = (slice(max(ys.min() - PAD, 0), ys.max() + PAD + 1), slice(max(xs.min() - PAD, 0), xs.max() + PAD + 1))
            m = m_full[sl]
            ev = erode(m, 3)
            Pc, Nc, Vc, Pmc = V["P"][sl], V["N"][sl], Vd[sl], P_m[sl]
            Pn, Nn, Vn, Pmn = Pc[m], Nc[m], Vc[m], Pmc[m]
            w = box_weights(Nn)
            rough_px = t["rough"] * (0.6 + 0.8 * box_project(rimg, Pn, Nn, tex_mm=args.tex_mm, w=w, scale=rs))
            a_b, a_c = rough_px ** 2, np.full(len(Pn), t["coat_rough"] ** 2)
            f0b, f0c = gm.f0_from_ior(t["ior"]), gm.f0_from_ior(gm.COAT_IOR)
            shade = lambda Nb, Ncoat, tb=None, tc=None: (
                gm.specular(Nb, Vn, Pmn, rig, a_b, f0b, tb) + t["coat"] * gm.specular(Ncoat, Vn, Pmn, rig, a_c, f0c, tc))

            flat = shade(Nn, Nn)
            ref = max(float(np.median(flat)), 1e-12)

            def stat(vals):
                img = np.zeros(m.shape)
                img[m] = np.log1p(vals / ref)
                return highpass(np.abs(highpass(img, m, 1.5)), m, 6.0)[ev]

            gt_s = highpass(np.abs(highpass(Y[sl], m, 1.5)), m, 6.0)[ev]
            flat_s = stat(flat)
            keys = ["all"] + ([organ] if organ in ("liver", "kidney_left", "kidney_right", "spleen") else [])

            def score(name, vals, null_vals):
                s, sn = stat(vals), stat(null_vals)
                for k in keys:
                    C[(name, k, "gt")].add(s, gt_s)
                    C[(name, k, "flat")].add(s, flat_s)
                    C[(name, k, "null_gt")].add(sn, gt_s)
                    C[(name, k, "null_flat")].add(sn, flat_s)

            for k in keys:
                C[("flat", k, "gt")].add(flat_s, gt_s)

            # Tangent frames depend on geometry only: build once, reuse for real and null tiles.
            bb = mesh_bbox(args.mesh_dir, subj, organ)
            frames = {}
            for fam in gm.TANGENT_FAMILIES:
                if bb is None and fam != "sphere_origin":
                    continue
                T_, B_ = gm.screen_tangents(Pc, Nc, *gm.sphere_uv(Pc, fam, bb))
                frames[fam] = (T_[m], B_[m])

            def hypotheses(col):
                out = {}
                tilt = gm.max_tilt(col)
                out["free_C0"] = shade(Nn, Nn, tb=tilt)
                out["free_C1"] = shade(Nn, Nn, tb=tilt, tc=tilt)
                out["ridge"] = np.hypot(col[:, 0], col[:, 1])
                for fam, (Tm, Bm) in frames.items():
                    for sname, (sT, sB) in signs.items():
                        Np = gm.perturb(Nn, Tm, Bm, col, sT, sB)
                        out[f"{fam}{sname}_C0"] = shade(Np, Nn)
                        out[f"{fam}{sname}_C1"] = shade(Np, Np)
                return out

            tile = lambda shift: box_project(nimg, Pn + shift, Nn, tex_mm=args.tex_mm, w=w, scale=ns) * 2.0 - 1.0
            maps, nulls = hypotheses(tile(0.0)), hypotheses(tile(args.tex_mm / 2.0))
            for name, vals in maps.items():
                if name == "ridge":          # raw ridge magnitude, not a radiance: score it directly
                    s = highpass(_to_img(m, vals), m, 6.0)[ev]
                    sn = highpass(_to_img(m, nulls[name]), m, 6.0)[ev]
                    for k in keys:
                        C[(name, k, "gt")].add(s, gt_s); C[(name, k, "flat")].add(s, flat_s)
                        C[(name, k, "null_gt")].add(sn, gt_s); C[(name, k, "null_flat")].add(sn, flat_s)
                else:
                    score(name, vals, nulls[name])
        print(f"  [{i+1}/{len(views)}] {subj}/{Path(vd).name}", flush=True)

    # ── report ──
    rf = C[("flat", "all", "gt")].r
    names = sorted({k[0] for k in C if k[0] != "flat"})
    # Rank by TEXTURE-LOCKED excess (partial r minus its half-tile null), not raw partial r.
    # The local smoke test showed any normal-map perturbation — even a misplaced one — lifts
    # partial r by ~0.03-0.045, because added sparkle correlates with where highlights are.
    # That floor is not about vessel placement; only the excess over the null is.
    rows = []
    for h in names:
        r = C[(h, "all", "gt")].r
        pr = partial_r(r, C[(h, "all", "flat")].r, rf)
        prn = partial_r(C[(h, "all", "null_gt")].r, C[(h, "all", "null_flat")].r, rf)
        liv = C.get((h, "liver", "gt"))
        rows.append((pr - prn, h, r, pr, prn, liv.r if liv else float("nan")))
    rows.sort(key=lambda x: -x[0])
    print(f"\nplain geometric highlights (no normal map): r(flat, GT streaks) = {rf:+.3f}   <- light-rig check")
    print("\nhypotheses ranked by texture-locked excess = partial r - null partial r:")
    print(f"   {'hypothesis':26} {'excess':>8} {'partial r':>10} {'null partial':>13} {'raw r':>8} {'liver r':>8}")
    for ex, h, r, pr, prn, lv in rows:
        print(f"   {h:26} {ex:>+8.3f} {pr:>+10.3f} {prn:>+13.3f} {r:>+8.3f} {lv:>+8.3f}")
    ex_b, name_b, _, pr_b, prn_b, _ = rows[0]
    ex_r = next(x[0] for x in rows if x[1] == "ridge")
    T = dict(light=0.05, vessel=0.10, null_frac=1 / 3, over_ridge=1.5)
    checks = {
        "G1 light rig reproduces highlights": (rf >= T["light"], f"r(flat) {rf:+.3f} >= {T['light']}"),
        "G2 vessel glints computable": (pr_b >= T["vessel"], f"partial {pr_b:+.3f} ({name_b}) >= {T['vessel']}"),
        "G3 locked to the texture": (prn_b <= T["null_frac"] * pr_b, f"null partial {prn_b:+.3f} <= {pr_b:+.3f}/3"),
        "G4 beats the raw ridge proxy": (ex_b >= T["over_ridge"] * max(ex_r, 0.01),
                                         f"excess {ex_b:+.3f} >= {T['over_ridge']} x max(ridge excess {ex_r:+.3f}, 0.01)"),
    }
    print("\nDECISION (thresholds fixed before the server run)")
    for k, (flag, why) in checks.items():
        print(f"   {'PASS' if flag else 'FAIL'}  {k:36} {why}")
    f = {k.split()[0]: v[0] for k, v in checks.items()}
    if all(f.values()):
        v = f"GO — '{name_b}' computes the vessel glints; build the glint pre-flight on it"
    elif not f["G1"]:
        v = "STOP — the light/specular model does not reproduce these views' highlights; fix it before judging vessels"
    elif not f["G3"]:
        v = ("NO-GO — no texture-LOCKED glint signal (null ~ real). If `preflight.py check` A1 did not pass, "
             "the tiles are not the rendering tiles; otherwise -> LIMITATION")
    elif not f["G2"]:
        v = "NO-GO — glints are locked to the texture but too weak to be a useful input -> LIMITATION"
    else:
        v = "NO-GO — no better than the raw normal-map ridges the pre-flight already gave the network -> LIMITATION"
    print(f"\nVERDICT glint: {v}")


def _to_img(m, vals):
    img = np.zeros(m.shape)
    img[m] = vals
    return img


# ── geometry (gold standard: reconstructed points vs the subject's own meshes) ──
def cmd_geometry(args):
    """Distance from every reconstructed organ pixel to the organ meshes that were rendered.

    The only check that assumes nothing about the camera model. Pass: corrected median
    within ~1 mm (mesh vertex spacing ~1.5 mm sets the floor) on EVERY subject.
    """
    from scipy.spatial import cKDTree
    views = list_views(args)
    by_subj = defaultdict(list)
    for v in views:
        by_subj[v.parent.name].append(v)
    rng = np.random.default_rng(0)
    rows = []
    for subj, vs in by_subj.items():
        ids = json.load(open(vs[0].parent / "tissue_ids.json")).get("tissues", {})
        verts = []
        for name in ids:
            f = Path(args.mesh_dir) / subj / f"{name}.obj"
            if f.exists():
                verts.append(np.array([l.split()[1:4] for l in open(f) if l.startswith("v ")], np.float64))
        if not verts:
            print(f"  {subj}: no meshes under {args.mesh_dir}/{subj} — skipped"); continue
        allv = np.concatenate(verts)
        tree = cKDTree(allv)
        meta0 = json.load(open(vs[0] / "meta.json"))
        bbox_err = np.linalg.norm(np.array(meta0["organ_center"]) * 1000 - (allv.min(0) + allv.max(0)) / 2)
        for vd in vs[: args.views_per_subject]:
            V = load_view(vd)
            m = V["gid"] > 0
            sel = rng.choice(int(m.sum()), size=min(int(m.sum()), 20000), replace=False)
            dc = tree.query(V["P_corr"][m][sel])[0]
            dl = tree.query(V["P_leg"][m][sel])[0]
            rows.append((subj, vd.name, np.median(dc), np.percentile(dc, 90), np.median(dl), bbox_err))
            print(f"  {subj}/{vd.name:18} corrected median {np.median(dc):5.2f} p90 {np.percentile(dc,90):5.2f} mm"
                  f" | legacy median {np.median(dl):6.2f} mm | mesh-vs-meta centre {bbox_err:.2f} mm", flush=True)
    if not rows:
        raise SystemExit("no subjects with meshes")
    c = np.array([r[2] for r in rows]); l = np.array([r[4] for r in rows]); bb = np.array([r[5] for r in rows])
    bad = [f"{r[0]}/{r[1]}" for r in rows if r[2] > 1.5]
    print(f"\n{len(rows)} views, {len(by_subj)} subjects")
    print(f"   corrected: median of medians {np.median(c):.2f} mm, worst {c.max():.2f} mm")
    print(f"   legacy   : median of medians {np.median(l):.2f} mm, worst {l.max():.2f} mm")
    print(f"   mesh bbox centre vs meta organ_center: worst {bb.max():.2f} mm "
          f"(large = these meshes are not the ones that were rendered)")
    print("VERDICT geometry: " + ("PASS — corrected camera model holds on every view"
                                  if not bad and bb.max() < 3 else
                                  f"FAIL on {len(bad)} view(s): {bad[:6]}"))


# ── depth survey ──────────────────────────────────────────────────────────────
def cmd_depth_survey(args):
    views = list_views(args)
    rng = np.random.default_rng(0)
    if args.max_views and len(views) > args.max_views:
        views = [views[i] for i in rng.choice(len(views), args.max_views, replace=False)]
    steps, half, ratios = [], 0, []
    for vd in views:
        img = read_render_exr(str(vd / "depth.exr"), want=("image",))["image"].astype(np.float64)
        z = img[..., 0]
        fgm = z > 0.5 * z.max()
        zf = z[fgm]
        u = np.unique(zf)
        steps.append(float(np.median(np.diff(u)) * 1000.0) if u.size > 1 else np.nan)
        half += int(np.abs(zf - zf.astype(np.float16)).max() == 0)
        if img.shape[-1] >= 3:
            ratios.append((np.median(img[..., 1][fgm] / zf), np.median(img[..., 2][fgm] / zf)))
    s = np.array(steps)
    if ratios:
        r = np.array(ratios)
        print(f"emission gain check (expect G/R 1.00000, B/R 0.99213 everywhere): "
              f"G/R {r[:,0].min():.5f}-{r[:,0].max():.5f}   B/R {r[:,1].min():.5f}-{r[:,1].max():.5f}")
    print(f"{len(views)} views: HALF-float {half}/{len(views)}")
    for q in (0, 10, 50, 90, 100):
        print(f"   depth step p{q:<3} {np.nanpercentile(s, q):.3f} mm")
    for lo, hi in ((0, 0.6), (0.6, 1.2), (1.2, 2.5), (2.5, 99)):
        print(f"   {lo:>4}-{hi:<4} mm : {np.mean((s >= lo) & (s < hi))*100:5.1f}% of views")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--views", nargs="*", help="explicit view dirs")
        p.add_argument("--data-root", default="data/training_dataset")
        p.add_argument("--subjects", nargs="*")
        p.add_argument("--max-subjects", type=int, default=0)
        p.add_argument("--tex-dir", default="data/renders/textures/tissue")
        p.add_argument("--tex-mm", type=float, default=TEX_MM)
        p.add_argument("--legacy-unproject", action="store_true",
                       help="use dataset.unproject_position as gbuf_pos1024 was trained "
                            "(no lens shift, no emission gain) — for contrast only")

    p = sub.add_parser("oracle"); common(p)
    p.add_argument("--signal", choices=["ridge", "albedo"], default="ridge",
                   help="ridge: vessel lines from the normal map vs GT streak energy "
                        "(where v25's vessels actually are); albedo: colour mottling")
    p.add_argument("--band", type=float, default=6.0,
                   help="ridge mode: local-mean removal sigma (px)")
    p.add_argument("--sigma", type=float, default=1.5, help="screen high-pass sigma (px)")
    p.add_argument("--erode", type=int, default=3)
    p.add_argument("--noise-mm", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--preset", default="full1024")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("registration"); common(p)
    p.add_argument("--depth-tol-mm", type=float, default=5.0)
    p.add_argument("--min-overlap", type=int, default=2000)
    p.set_defaults(fn=cmd_registration)

    p = sub.add_parser("spectrum")
    p.add_argument("--sets", nargs=2, action="append", metavar=("LABEL", "DIR_OR_GLOB"), required=True)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--patches-per-image", type=int, default=8)
    p.add_argument("--max-images", type=int, default=300)
    p.add_argument("--crop-panel", type=int, nargs=2, default=None, metavar=("INDEX", "OF"))
    p.add_argument("--save", default=None, help="write the mean log spectra side by side (PNG)")
    p.set_defaults(fn=cmd_spectrum)

    p = sub.add_parser("phase"); common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preset", default="gbuf_pos1024")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-views", type=int, default=8)
    p.add_argument("--out", default="results/texture_phase")
    p.set_defaults(fn=cmd_phase)

    p = sub.add_parser("glint"); common(p)
    p.add_argument("--mesh-dir", default="data/meshes")
    p.set_defaults(fn=cmd_glint)

    p = sub.add_parser("geometry"); common(p)
    p.add_argument("--mesh-dir", default="data/meshes")
    p.add_argument("--views-per-subject", type=int, default=3)
    p.set_defaults(fn=cmd_geometry)

    p = sub.add_parser("depth-survey"); common(p)
    p.add_argument("--max-views", type=int, default=400)
    p.set_defaults(fn=cmd_depth_survey)

    args = ap.parse_args()
    global LEGACY_UNPROJECT
    LEGACY_UNPROJECT = bool(getattr(args, "legacy_unproject", False))
    print(f"[unproject] {'LEGACY (as gbuf_pos1024 trained)' if LEGACY_UNPROJECT else 'corrected: lens shift + emission gain'}")
    args.fn(args)


if __name__ == "__main__":
    main()
