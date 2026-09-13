"""
preflight.py — can the last full retrain learn organ texture? Decide on 2 subjects first.

Design: two short runs on the SAME 2 subjects, 4 views per subject held out.
    preflight_tex   full1024 inputs + `texture` buffer (15 ch), resize upsampling
    preflight_ctrl  full1024 inputs                    ( 8 ch), resize upsampling
Held-out views are new camera angles of surfaces the model has seen, so memorised pixels
cannot transfer to them — only a prediction driven by the looked-up texture can put
vessels in the right place. The control shows how much any model gets without the texture.

Stages, in order. Every check prints an ID and PASS/FAIL so a failure points at one piece.

  check      (before training)  A0 env · A1 tiles are the rendering tiles · A2 geometry on
             the meshes · A3 dataloader == texture_lookup == infer · A4 the buffer carries a
             registered texture signal · A5 dataloader speed · A6 both models build/run
  plan       choose 2 train-split subjects, fix held-out views, pre-register the pass
             thresholds in plan.json, write heldout.json, print the two training commands
  evaluate   (after training)   D0-D5 decision matrix on the held-out views, panels, json

Metrics (eroded masks of textured organs; Y = luminance):
  M1 fit          corr(HF(pred), HF(GT))                  does the model reproduce fine detail
  M3 vessels      corr(streak(Y), ridge(texture at P))    are streaks where the real vessel
                  computed for GT, tex, ctrl              ridges are (GT sets the ceiling)
  M2 added        corr(HF(tex - ctrl), HF(GT - ctrl))     is what the texture adds correct
  M4 lattice      share of HF power on the 32 px lattice  checkerboard (GT ~1.5%)
  M5 PSNR         organ-masked PSNR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import camera_model as cm                                           # noqa: E402
from texture_lookup import (TextureBank, texture_buffer, box_project,  # noqa: E402
                            tex_class)
from config import preset                                           # noqa: E402
from dataset import CANONICAL_TISSUES, ORGAN_TO_GID, PairedRenderDataset  # noqa: E402
import texture_learnability as TL                                   # noqa: E402

SIG_HF, SIG_BAND, ERODE = 1.5, 6.0, 3
DEFAULT_HELDOUT = ["v02", "v07", "v12", "v14"]     # az0 el0 · az72 el30 · az144 el60 · az216 el0
THRESHOLDS = {                                     # pre-registered; copied into plan.json
    "D0_min_M3_gt": 0.03,        # these views must show a registered texture at all
    "D1_min_M1_train_tex": 0.30, # the tex model must fit fine detail on its training views
                                 # (calibration: full1024, trained on everything, gets 0.455)
    "D2_min_M3_ratio": 0.50,     # tex streaks registered to ridges >= half the GT's own
    "D3_max_M3_ctrl_ratio": 0.25,# ... while the control gets <= a quarter of it
    "D3_min_M2": 0.10,           # what the texture adds must correlate with what ctrl misses
    "D4_max_lattice_pct": 2.5,   # GT ~1.5%; transpose models were 4.5-4.9%
    "D5_max_psnr_drop_db": 0.5,  # texture input must not cost overall quality
}


def ok(flag):
    return "PASS" if flag else "FAIL"


# ── shared per-view data ──────────────────────────────────────────────────────
def view_data(vd, bank):
    vd = Path(vd)
    meta = json.load(open(vd / "meta.json"))
    P = cm.unproject(cm.depth_m(vd), meta)
    N = cm.normals_world(vd)
    lut = TL.subject_lut(vd.parent)
    gid = lut[np.clip(cm.segid_pid(vd), 0, len(lut) - 1)]
    gt = cv2.cvtColor(cv2.imread(str(vd / "rgb_preview.png")), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return dict(dir=vd, meta=meta, P=P, N=N, gid=gid, gt=gt)


_RIDGE = {}


def ridge_tile(bank, cls):
    if cls not in _RIDGE:
        img, s = bank.get(cls, "normal")
        _RIDGE[cls] = None if img is None else np.sqrt(
            (img[..., 0] * s - 0.5) ** 2 + (img[..., 1] * s - 0.5) ** 2).astype(np.float32)
    return _RIDGE[cls]


def organ_masks(V, bank):
    """[(organ name, texture class, full mask, eroded mask)] for textured organs."""
    out = []
    for g in np.unique(V["gid"]):
        if g <= 0:
            continue
        name = CANONICAL_TISSUES[int(g) - 1]
        cls = tex_class(name)
        if cls is None or ridge_tile(bank, cls) is None:
            continue
        m = V["gid"] == g
        ev = TL.erode(m, ERODE)
        if ev.sum() >= 400:
            out.append((name, cls, m, ev))
    return out


def ridge_image(V, bank, masks, offset_mm=0.0):
    img = np.zeros(V["gid"].shape)
    for _, cls, m, _ in masks:
        idx = np.nonzero(m)
        img[idx] = box_project(ridge_tile(bank, cls), V["P"][idx] + offset_mm, V["N"][idx],
                               tex_mm=bank.tex_mm)
    return img


def streak(Y, m):
    return TL.highpass(np.abs(TL.highpass(Y, m, SIG_HF)), m, SIG_BAND)


def lattice_share(images, P=128, per=60, period=32, seed=0):
    rng = np.random.default_rng(seed)
    win = np.outer(np.hanning(P), np.hanning(P)); acc = np.zeros((P, P)); n = 0
    for im in images:
        Y = TL.lum(im)
        fg = cv2.erode((Y > 0.04).astype(np.uint8), np.ones((P + 1, P + 1), np.uint8)).astype(bool)
        ys, xs = np.nonzero(fg)
        for _ in range(per):
            if ys.size == 0:
                break
            c = rng.integers(ys.size); y0, x0 = ys[c] - P // 2, xs[c] - P // 2
            if y0 < 0 or x0 < 0 or y0 + P > Y.shape[0] or x0 + P > Y.shape[1]:
                continue
            p = Y[y0:y0 + P, x0:x0 + P]
            acc += np.abs(np.fft.fftshift(np.fft.fft2((p - p.mean()) * win))) ** 2; n += 1
    if n == 0:
        return float("nan")
    S = acc / n; c = P // 2; step = P // period
    yy, xx = np.mgrid[:P, :P] - c
    band = np.hypot(yy, xx) >= step
    on = band & (yy % step == 0) & (xx % step == 0)
    excess = sum(max(S[y, x] - np.median(S[max(y-2, 0):y+3, max(x-2, 0):x+3]), 0.0)
                 for y, x in zip(*np.nonzero(on)))
    return float(excess / S[band].sum() * 100.0)


def load_generator(preset_name, ckpt, device):
    import torch
    from networks import define_G
    cfg = preset(preset_name)
    G = define_G(cfg).to(device).eval()
    ck = torch.load(ckpt, map_location="cpu")
    G.load_state_dict(ck["G"] if isinstance(ck, dict) and "G" in ck else ck)
    return cfg, G


def predict(G, cfg, vd, device):
    import torch
    import infer as I
    x = I.load_input(Path(vd), cfg, I.build_segid_lut(Path(vd).parent))
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        y = G(t)[0].float().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)
    return (y + 1.0) * 0.5


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ── stage A: check ────────────────────────────────────────────────────────────
def cmd_check(args):
    root, subj = Path(args.data_root), args.subject
    views = sorted(d for d in (root / subj).glob("v*") if (d / "meta.json").exists())
    vd = next((d for d in views if d.name.startswith(args.view)), None)
    if vd is None:
        raise SystemExit(f"view {args.view}* not found under {root / subj}")
    results = {}
    print(f"pre-flight check on {subj}/{vd.name}\n")

    # A0 environment
    try:
        import torch, scipy  # noqa: F401
        results["A0"] = True
        print(f"A0 env            PASS  torch {torch.__version__} cuda={torch.cuda.is_available()} scipy OK")
    except Exception as e:
        results["A0"] = False
        print(f"A0 env            FAIL  {e}")

    # A1 tiles == rendering tiles
    tex_dir = Path(args.tex_dir)
    sums = tex_dir / "MD5SUMS.txt"
    if not sums.exists():
        results["A1"] = False
        print(f"A1 tiles          FAIL  {sums} missing — create it on the server: "
              f"md5sum {tex_dir}/*.png > {sums}")
    else:
        want = {Path(l.split()[1]).name: l.split()[0] for l in open(sums) if l.strip()}
        bad = [n for n, h in want.items()
               if not (tex_dir / n).exists() or hashlib.md5((tex_dir / n).read_bytes()).hexdigest() != h]
        results["A1"] = not bad and len(want) > 0
        print(f"A1 tiles          {ok(results['A1'])}  {len(want) - len(bad)}/{len(want)} tiles match MD5SUMS"
              + (f"  mismatched: {bad[:5]}" if bad else ""))

    bank = TextureBank(args.tex_dir, preset("preflight_tex").data.tex_mm)
    V = view_data(vd, bank)

    # A2 geometry on the meshes
    try:
        from scipy.spatial import cKDTree
        ids = json.load(open(root / subj / "tissue_ids.json"))["tissues"]
        verts = [np.array([l.split()[1:4] for l in open(Path(args.mesh_dir) / subj / f"{n}.obj")
                           if l.startswith("v ")], np.float64)
                 for n in ids if (Path(args.mesh_dir) / subj / f"{n}.obj").exists()]
        allv = np.concatenate(verts)
        m = V["gid"] > 0
        sel = np.random.default_rng(0).choice(int(m.sum()), min(int(m.sum()), 20000), replace=False)
        d = cKDTree(allv).query(V["P"][m][sel])[0]
        centre = np.linalg.norm(np.array(V["meta"]["organ_center"]) * 1000 - (allv.min(0) + allv.max(0)) / 2)
        results["A2"] = np.median(d) <= 1.5 and centre < 3.0
        print(f"A2 geometry       {ok(results['A2'])}  reconstruction->mesh median {np.median(d):.2f} mm "
              f"(p90 {np.percentile(d, 90):.2f}); mesh-vs-meta centre {centre:.2f} mm")
    except Exception as e:
        results["A2"] = False
        print(f"A2 geometry       FAIL  {type(e).__name__}: {e}")

    # A3 dataloader == the shared buffer functions == infer, for every derived buffer
    import infer as I
    import glint_model as gm
    from config import buffer_channels
    from texture_lookup import box_weights
    cfg = preset(args.preset); cfg.data.root = str(root); cfg.data.subjects = [subj]
    cfg.data.tex_dir = args.tex_dir
    ds = PairedRenderDataset(cfg, split="train")
    i = next(k for k, s in enumerate(ds.samples) if s["dir"].name == vd.name)
    t0 = time.time(); x = ds[i]["input"].numpy(); t_load = time.time() - t0
    lut = I.build_segid_lut(vd.parent)
    offs, o = {}, 0
    for b in cfg.data.input_buffers:
        n = buffer_channels(b, cfg.data.position_bands); offs[b] = (o, o + n); o += n
    diffs = {}
    if "texture" in offs:
        a, e = offs["texture"]
        diffs["texture"] = float(np.abs(x[a:e] - texture_buffer(vd, V["meta"], lut, CANONICAL_TISSUES, bank)
                                        .transpose(2, 0, 1)).max())
    if "glint" in offs:
        a, e = offs["glint"]
        diffs["glint"] = float(np.abs(x[a:e] - gm.glint_buffer(vd, V["meta"], lut, CANONICAL_TISSUES, bank, tex_class,
                                                               box_weights, box_project).transpose(2, 0, 1)).max())
    d_infer = float(np.abs(I.load_input(vd, cfg, lut).transpose(2, 0, 1) - x).max())
    results["A3"] = (x.shape == (cfg.input_nc(), 1024, 1024) and all(d < 1e-5 for d in diffs.values())
                     and d_infer < 1e-5)
    print(f"A3 buffer path    {ok(results['A3'])}  [{args.preset}] input {x.shape}; "
          + "; ".join(f"|dataloader-{k}_buffer| {d:.1e}" for k, d in diffs.items())
          + f"; |dataloader-infer| {d_infer:.1e}")

    # A4 the buffer carries a registered texture signal (oracle statistic, subject-wide)
    c_sig, c_null = TL.Corr(), TL.Corr()
    for v in views[: args.a4_views]:
        W_ = view_data(v, bank)
        masks = organ_masks(W_, bank)
        Y = TL.lum(W_["gt"])
        r0 = ridge_image(W_, bank, masks)
        rn = ridge_image(W_, bank, masks, offset_mm=bank.tex_mm / 2)
        for _, _, m, ev in masks:
            s = streak(Y, m)
            c_sig.add(TL.highpass(r0, m, SIG_BAND)[ev], s[ev])
            c_null.add(TL.highpass(rn, m, SIG_BAND)[ev], s[ev])
    results["A4"] = c_sig.r >= 0.03 and c_sig.r >= 5 * abs(c_null.r)
    print(f"A4 texture signal {ok(results['A4'])}  over {min(len(views), args.a4_views)} views: "
          f"r(ridge, GT streaks) = {c_sig.r:+.3f}, half-tile null = {c_null.r:+.3f} "
          f"(server oracle, 3 subjects: +0.057 / +0.001)")

    # A5 speed
    t0 = time.time()
    for k in range(min(3, len(ds))):
        ds[k]
    per = (time.time() - t0) / min(3, len(ds))
    results["A5"] = per < 4.0
    nw = cfg.data.num_workers
    print(f"A5 loader speed   {ok(results['A5'])}  {per*1000:.0f} ms/sample (first {t_load*1000:.0f} ms); "
          f"with {nw} workers ~{nw/per:.1f} samples/s")

    # A7 the glint buffer carries the texture-locked vessel-glint signal — measured on the
    # buffer the network will actually receive, against its own half-tile null.
    if "glint" in cfg.data.input_buffers:
        c7 = defaultdict(TL.Corr)
        for v in views[: args.a4_views]:
            W_ = view_data(v, bank)
            lut_v = I.build_segid_lut(v.parent)
            gb = gm.glint_buffer(v, W_["meta"], lut_v, CANONICAL_TISSUES, bank, tex_class, box_weights, box_project)
            gn = gm.glint_buffer(v, W_["meta"], lut_v, CANONICAL_TISSUES, bank, tex_class, box_weights, box_project,
                                 tex_shift_mm=bank.tex_mm / 2)
            Y = TL.lum(W_["gt"])
            for _, _, m, ev in organ_masks(W_, bank):
                s_gt, s_v, s_f, s_n = (streak(img, m)[ev] for img in (Y, gb[..., 0], gb[..., 1], gn[..., 0]))
                c7["v_gt"].add(s_v, s_gt); c7["f_gt"].add(s_f, s_gt); c7["v_f"].add(s_v, s_f)
                c7["n_gt"].add(s_n, s_gt); c7["n_f"].add(s_n, s_f)
        pr = TL.partial_r(c7["v_gt"].r, c7["v_f"].r, c7["f_gt"].r)
        prn = TL.partial_r(c7["n_gt"].r, c7["n_f"].r, c7["f_gt"].r)
        results["A7"] = (pr - prn) >= 0.05 and prn <= pr / 3
        print(f"A7 glint signal   {ok(results['A7'])}  over {min(len(views), args.a4_views)} views: partial r(vessel glints, "
              f"GT streaks | plain highlights) {pr:+.3f}, half-tile null {prn:+.3f}, excess {pr - prn:+.3f} "
              f"(server glint test, 3 subjects: 0.149 / 0.030)")

    # A6 models
    try:
        import torch
        dev = args.device or default_device()
        from networks import define_G
        for name, nc in ((args.preset, cfg.input_nc()), ("preflight_ctrl", 8)):
            c = preset(name)
            g = define_G(c).to(dev).eval()
            n_tr = sum(isinstance(mm, torch.nn.ConvTranspose2d) for mm in g.modules())
            with torch.no_grad():
                inp = torch.from_numpy(x[:nc]).unsqueeze(0).float().to(dev)
                y = g(inp)
            good = c.input_nc() == nc and n_tr == 0 and tuple(y.shape) == (1, 3, 1024, 1024) \
                and bool(torch.isfinite(y).all())
            results[f"A6_{name}"] = good
            print(f"A6 {name:14} {ok(good)}  input_nc={c.input_nc()} ConvTranspose2d={n_tr} "
                  f"out={tuple(y.shape)} on {dev}")
            del g
    except Exception as e:
        results["A6"] = False
        print(f"A6 models         FAIL  {type(e).__name__}: {e}")

    failed = [k for k, v in results.items() if not v]
    print("\nCHECK VERDICT: " + ("ALL PASS — proceed to `plan`" if not failed else
                                 f"FAILED {failed} — fix these before training"))


# ── stage B: plan ─────────────────────────────────────────────────────────────
def cmd_plan(args):
    from splits import assign_splits
    root = Path(args.data_root)
    cfg = preset("preflight_tex"); cfg.data.root = str(root)
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("v*/meta.json")))
    split = assign_splits(subjects, cfg)
    pool = subjects if args.allow_any_split else [s for s in subjects if split.get(s) == "train"]
    rng = np.random.default_rng(0)
    if len(pool) > args.scan:
        pool = sorted(rng.choice(pool, args.scan, replace=False))
    rows = []
    for s in pool:
        vs = sorted(d for d in (root / s).glob("v*") if (d / "meta.json").exists())
        held = [next((d.name for d in vs if d.name.startswith(h)), None) for h in args.heldout]
        if len(vs) < 20 or None in held:
            continue
        lut = TL.subject_lut(root / s)
        v2 = next(d for d in vs if d.name.startswith("v02"))
        gid = lut[np.clip(cm.segid_pid(v2), 0, len(lut) - 1)]
        px = {o: int((gid == ORGAN_TO_GID[o]).sum()) for o in ("liver", "kidney_left", "kidney_right", "spleen")}
        z = cm.depth_m(v2)[gid > 0]
        u = np.unique(z) if z.size else np.array([])
        step = float(np.median(np.diff(u)) * 1000) if u.size > 1 else float("nan")
        score = px["liver"] + px["kidney_left"] + px["kidney_right"]
        rows.append((score, s, px, step, held))
    rows.sort(key=lambda r: -r[0])
    print(f"scanned {len(pool)} {'' if args.allow_any_split else 'train-split '}subjects; top candidates "
          f"(textured-organ pixels in v02: liver + kidneys):")
    for score, s, px, step, _ in rows[:10]:
        print(f"   {s}  score {score:>7}  liver {px['liver']:>6}  kidneys {px['kidney_left']+px['kidney_right']:>6}"
              f"  spleen {px['spleen']:>6}  depth step {step:.2f} mm")

    # Move D0 before training: measure the registered-texture signal (GT streaks vs vessel
    # ridges) on each candidate's HELD-OUT views, and keep only subjects that clear it.
    bank = TextureBank(args.tex_dir, cfg.data.tex_mm)
    print(f"\nverifying the texture signal on the held-out views of the top {args.verify} "
          f"(D0 needs M3_gt >= {THRESHOLDS['D0_min_M3_gt']}):")
    verified = []
    for row in rows[: args.verify]:
        s, held = row[1], row[4]
        c = TL.Corr()
        for h in held:
            V = view_data(root / s / h, bank)
            masks = organ_masks(V, bank)
            ridge, Y = ridge_image(V, bank, masks), TL.lum(V["gt"])
            for _, _, m, ev in masks:
                c.add(TL.highpass(ridge, m, SIG_BAND)[ev], streak(Y, m)[ev])
        passed = c.r >= THRESHOLDS["D0_min_M3_gt"]
        print(f"   {s}  M3_gt {c.r:+.3f}  {ok(passed)}", flush=True)
        if passed:
            verified.append((c.r, row))
    verified.sort(key=lambda t: -t[0])
    chosen = [row for _, row in verified[: args.n_subjects]]
    if len(chosen) < args.n_subjects:
        raise SystemExit(f"only {len(chosen)} subject(s) show a registered texture signal — raise --verify or "
                         f"--scan. If none do, run `check` first: the tiles may not be the rendering tiles (A1/A4).")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    heldout = [f"{s}/{h}" for _, s, _, _, hs in chosen for h in hs]
    json.dump(heldout, open(out / "heldout.json", "w"), indent=2)
    plan = dict(created=time.strftime("%Y-%m-%d %H:%M"), data_root=str(root),
                tex_dir=args.tex_dir, subjects=[s for _, s, _, _, _ in chosen],
                heldout=heldout, thresholds=THRESHOLDS,
                tiles_md5sums=(hashlib.md5(open(Path(args.tex_dir) / "MD5SUMS.txt", "rb").read()).hexdigest()
                               if (Path(args.tex_dir) / "MD5SUMS.txt").exists() else None))
    json.dump(plan, open(out / "plan.json", "w"), indent=2)
    subs = ",".join(plan["subjects"])
    print(f"\nchosen: {plan['subjects']}   held out: {heldout}")
    print(f"wrote {out/'plan.json'} (thresholds pre-registered) and {out/'heldout.json'}")
    print("\nTRAIN (one GPU each; run both, in parallel if two GPUs are free):")
    for name, gpu in (("preflight_tex", 0), ("preflight_ctrl", 1)):
        print(f"  mkdir -p results/training_runs/{name} && CUDA_VISIBLE_DEVICES={gpu} python scripts/training/train.py "
              f"--preset {name} --data-root {root} --subjects {subs} --exclude-file {out/'heldout.json'} "
              f"--tex-dir {args.tex_dir} --output-dir results/training_runs/{name} "
              f"2>&1 | tee -a results/training_runs/{name}/train.log")


# ── stage D: evaluate ─────────────────────────────────────────────────────────
def evaluate_epoch(plan, tex_ckpt, ctrl_ckpt, device, bank, views_held, views_train, panels_dir=None,
                   tex_preset="preflight_tex"):
    cfg_t, G_t = load_generator(tex_preset, tex_ckpt, device)
    cfg_c, G_c = load_generator("preflight_ctrl", ctrl_ckpt, device)
    cfg_t.data.tex_dir = plan["tex_dir"]
    C = defaultdict(TL.Corr)
    imgs = defaultdict(list)
    psnr = defaultdict(list)
    for split, vds in (("heldout", views_held), ("train", views_train)):
        for vd in vds:
            V = view_data(vd, bank)
            masks = organ_masks(V, bank)
            if not masks:
                continue
            pt, pc = predict(G_t, cfg_t, vd, device), predict(G_c, cfg_c, vd, device)
            Yg, Yt, Yc = TL.lum(V["gt"]), TL.lum(pt), TL.lum(pc)
            ridge = ridge_image(V, bank, masks)
            organ_any = V["gid"] > 0
            for name, _, m, ev in masks:
                tr = TL.highpass(ridge, m, SIG_BAND)[ev]
                hg, ht, hc = (TL.highpass(Y, m, SIG_HF)[ev] for Y in (Yg, Yt, Yc))
                for key, sel in (("all", True), (name, name in ("liver", "kidney_left", "kidney_right"))):
                    if not sel:
                        continue
                    C[(split, "M1", "tex", key)].add(ht, hg)
                    C[(split, "M1", "ctrl", key)].add(hc, hg)
                    for mod, Y in (("gt", Yg), ("tex", Yt), ("ctrl", Yc)):
                        C[(split, "M3", mod, key)].add(tr, streak(Y, m)[ev])
                    C[(split, "TC", "x", key)].add(ht, hc)
            if split == "heldout":
                for mod, im in (("gt", V["gt"]), ("tex", pt), ("ctrl", pc)):
                    imgs[mod].append(im)
                for mod, im in (("tex", pt), ("ctrl", pc)):
                    mse = np.mean((im[organ_any] - V["gt"][organ_any]) ** 2)
                    psnr[mod].append(10 * np.log10(1.0 / max(mse, 1e-12)))
                if panels_dir is not None:
                    write_panel(panels_dir, vd, V, masks, pc, pt)
    # M2 = partial correlation of the texture model's fine detail with the GT's, controlling
    # for the control model's. NOT corr(HF(tex-ctrl), HF(GT-ctrl)): both of those contain
    # -HF(ctrl), so a noisy control correlates with itself — it read +0.40 for two untrained
    # models in the smoke test. All three correlations are pooled over identical pixels.
    for split, _, _, key in [k for k in list(C) if k[1] == "TC"]:
        r_tg, r_cg, r_tc = C[(split, "M1", "tex", key)].r, C[(split, "M1", "ctrl", key)].r, C[(split, "TC", "x", key)].r
        den = np.sqrt(max((1 - r_tc ** 2) * (1 - r_cg ** 2), 1e-12))
        C[(split, "M2", "tex", key)] = Fixed((r_tg - r_tc * r_cg) / den, C[(split, "TC", "x", key)].n)
    lat = {mod: lattice_share(im) for mod, im in imgs.items()}
    return C, lat, {k: float(np.mean(v)) for k, v in psnr.items()}


class Fixed:
    def __init__(self, r, n):
        self.r, self.n = float(r), n


def write_panel(out_dir, vd, V, masks, pc, pt):
    """Crop around the largest vessel-bearing organ: GT | control | texture, 2x zoom."""
    pick = max(masks, key=lambda t: (t[0] in ("liver", "kidney_left", "kidney_right"), t[2].sum()))
    ys, xs = np.nonzero(pick[2])
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h = 180
    y0, x0 = max(cy - h, 0), max(cx - h, 0)
    crop = lambda im: im[y0:y0 + 2 * h, x0:x0 + 2 * h]
    row = np.concatenate([crop(V["gt"]), crop(pc), crop(pt)], axis=1)
    row = cv2.resize((np.clip(row, 0, 1) * 255).astype(np.uint8), None, fx=2, fy=2,
                     interpolation=cv2.INTER_NEAREST)
    for k, lab in enumerate(("GT", "control (no texture)", "texture")):
        cv2.putText(row, lab, (10 + k * row.shape[1] // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out_dir) / f"{vd.parent.name}_{vd.name}_{pick[0]}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


def resolve_views(plan):
    root = Path(plan["data_root"])
    held = [root / h for h in plan["heldout"]]
    held_set = set(plan["heldout"])
    train = []
    for s in plan["subjects"]:
        vs = sorted(d for d in (root / s).glob("v*") if (d / "meta.json").exists()
                    and f"{s}/{d.name}" not in held_set)
        train += vs[: len(held) // len(plan["subjects"])]
    return held, train


def common_epochs(tex_run, ctrl_run):
    ep = lambda run: {p.name for p in (Path(run) / "checkpoints").glob("epoch_*.pt")}
    return sorted(ep(tex_run) & ep(ctrl_run))


def cmd_evaluate(args):
    plan = json.load(open(args.plan))
    T = plan["thresholds"]
    bank = TextureBank(plan["tex_dir"], preset("preflight_tex").data.tex_mm)
    held, train = resolve_views(plan)
    device = args.device or default_device()
    epochs = common_epochs(args.tex_run, args.ctrl_run)
    if not epochs:
        raise SystemExit("no epoch_*.pt common to both runs")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"held-out views: {len(held)}   train views scored: {len(train)}   device: {device}")

    if args.sweep:
        print("\nSWEEP (held-out M3 vessels registered: GT ceiling / tex / ctrl; train M1 fit; lattice %)")
        for e in epochs:
            C, lat, _ = evaluate_epoch(plan, Path(args.tex_run) / "checkpoints" / e,
                                       Path(args.ctrl_run) / "checkpoints" / e, device, bank, held, train,
                                       tex_preset=args.tex_preset)
            g = lambda *k: C[k].r
            print(f"   {e:14} M3 gt {g('heldout','M3','gt','all'):+.3f} tex {g('heldout','M3','tex','all'):+.3f} "
                  f"ctrl {g('heldout','M3','ctrl','all'):+.3f} | M2 {g('heldout','M2','tex','all'):+.3f} | "
                  f"M1 train tex {g('train','M1','tex','all'):+.3f} | lattice tex {lat['tex']:.2f} ctrl {lat['ctrl']:.2f}",
                  flush=True)

    e = args.epoch or epochs[-1]
    print(f"model under test: {args.tex_preset} ({args.tex_run})   control: preflight_ctrl ({args.ctrl_run})")
    C, lat, ps = evaluate_epoch(plan, Path(args.tex_run) / "checkpoints" / e,
                                Path(args.ctrl_run) / "checkpoints" / e, device, bank, held, train,
                                panels_dir=out / f"panels_{Path(e).stem}", tex_preset=args.tex_preset)
    g = lambda *k: C[k].r
    keys = ["all", "liver", "kidney_left", "kidney_right"]
    print(f"\n=== {e} ===")
    print(f"{'':34}" + "".join(f"{k:>14}" for k in keys))
    for split in ("heldout", "train"):
        for met, mod in (("M1", "tex"), ("M1", "ctrl"), ("M3", "gt"), ("M3", "tex"), ("M3", "ctrl"), ("M2", "tex")):
            row = f"{split:8} {met} {mod:5}" + " " * 16
            for k in keys:
                c = C.get((split, met, mod, k))
                row += f"{c.r:>+14.3f}" if c is not None and c.n > 100 else f"{'-':>14}"
            print(row)
    print(f"lattice % (held-out)  GT {lat['gt']:.2f}   tex {lat['tex']:.2f}   ctrl {lat['ctrl']:.2f}")
    print(f"organ PSNR (held-out) tex {ps['tex']:.2f} dB   ctrl {ps['ctrl']:.2f} dB")

    m3g, m3t, m3c = g("heldout", "M3", "gt", "all"), g("heldout", "M3", "tex", "all"), g("heldout", "M3", "ctrl", "all")
    d0 = m3g >= T["D0_min_M3_gt"]
    d1 = g("train", "M1", "tex", "all") >= T["D1_min_M1_train_tex"]
    gate = d0 and d1          # D2/D3 are ratios against the GT ceiling: meaningless unless both hold
    D = {
        "D0 views show a registered texture": (d0, f"M3_gt {m3g:+.3f} >= {T['D0_min_M3_gt']}"),
        "D1 tex model fits fine detail (train)": (d1, f"M1 {g('train','M1','tex','all'):+.3f} >= {T['D1_min_M1_train_tex']}"),
        "D2 vessels registered on NEW views": ((m3t >= T["D2_min_M3_ratio"] * m3g) if gate else None,
                                               f"M3_tex {m3t:+.3f} >= {T['D2_min_M3_ratio']}*{m3g:+.3f}"),
        "D3 attributable to the texture": ((m3c <= T["D3_max_M3_ctrl_ratio"] * m3g and
                                            g("heldout", "M2", "tex", "all") >= T["D3_min_M2"]) if gate else None,
                                           f"M3_ctrl {m3c:+.3f} <= {T['D3_max_M3_ctrl_ratio']}*{m3g:+.3f}, "
                                           f"M2 {g('heldout','M2','tex','all'):+.3f} >= {T['D3_min_M2']}"),
        "D4 no checkerboard lattice": (lat["tex"] <= T["D4_max_lattice_pct"] and lat["ctrl"] <= T["D4_max_lattice_pct"],
                                       f"tex {lat['tex']:.2f}% ctrl {lat['ctrl']:.2f}% <= {T['D4_max_lattice_pct']}%"),
        "D5 no quality regression": (ps["tex"] >= ps["ctrl"] - T["D5_max_psnr_drop_db"],
                                     f"PSNR tex {ps['tex']:.2f} >= ctrl {ps['ctrl']:.2f} - {T['D5_max_psnr_drop_db']}"),
    }
    meaning = {
        "D0 views show a registered texture": "data, not model: these subjects/views barely show vessels — re-run `plan` with other subjects",
        "D1 tex model fits fine detail (train)": "training, not texture: under-trained or broken — check train.log losses and samples/, train longer",
        "D2 vessels registered on NEW views": "the network cannot turn the texture buffer into vessels on unseen views — a full retrain will not either -> LIMITATION",
        "D3 attributable to the texture": "control does as well (vessels come from geometry memorisation), or the texture adds wrong detail — the buffer is not what helps",
        "D4 no checkerboard lattice": "architecture: resize upsampling did not remove the lattice",
        "D5 no quality regression": "the texture input costs overall quality",
    }
    print("\nDECISION MATRIX (thresholds pre-registered in plan.json)")
    for k, (flag, why) in D.items():
        print(f"   {'N/A ' if flag is None else ok(flag)}  {k:40} {why}")
        if flag is False:
            print(f"         -> {meaning[k]}")
    go = all(f is True for f, _ in D.values())
    if not d0:
        verdict = "INVALID TEST — these views do not show a registered texture (D0); re-run `plan` with other subjects"
    elif not d1:
        verdict = "INCONCLUSIVE — the texture model did not fit fine detail (D1); train longer or fix training, then re-evaluate"
    elif go:
        verdict = "GO — the full retrain with `texture` + resize upsampling is justified"
    else:
        verdict = "NO-GO — see the failed rows above for where it broke"
    print("\nPREFLIGHT VERDICT: " + verdict)
    json.dump(dict(epoch=e, metrics={"|".join(k): v.r for k, v in C.items()}, lattice=lat, psnr=ps,
                   decision={k: bool(f) for k, (f, _) in D.items()}, go=go),
              open(out / f"preflight_{Path(e).stem}.json", "w"), indent=2)
    print(f"panels: {out / ('panels_' + Path(e).stem)}   json: {out / ('preflight_' + Path(e).stem + '.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check")
    p.add_argument("--data-root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--view", default="v02")
    p.add_argument("--preset", default="preflight_tex", choices=["preflight_tex", "preflight_glint"])
    p.add_argument("--mesh-dir", default="data/meshes")
    p.add_argument("--tex-dir", default="data/renders/textures/tissue")
    p.add_argument("--a4-views", type=int, default=10)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("plan")
    p.add_argument("--data-root", required=True)
    p.add_argument("--tex-dir", default="data/renders/textures/tissue")
    p.add_argument("--n-subjects", type=int, default=2)
    p.add_argument("--scan", type=int, default=80)
    p.add_argument("--heldout", nargs="*", default=DEFAULT_HELDOUT)
    p.add_argument("--verify", type=int, default=6, help="candidates to verify the texture signal on")
    p.add_argument("--allow-any-split", action="store_true", help="local smoke tests only")
    p.add_argument("--out", default="results/preflight")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("evaluate")
    p.add_argument("--plan", default="results/preflight/plan.json")
    p.add_argument("--tex-run", default="results/training_runs/preflight_tex")
    p.add_argument("--tex-preset", default="preflight_tex", choices=["preflight_tex", "preflight_glint"],
                   help="preset of the model under test (the control is always preflight_ctrl)")
    p.add_argument("--ctrl-run", default="results/training_runs/preflight_ctrl")
    p.add_argument("--epoch", default=None, help="e.g. epoch_149.pt (default: last common)")
    p.add_argument("--sweep", action="store_true", help="also score every common checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/preflight/eval")
    p.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
