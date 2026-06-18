"""
analyze_dataset_split.py — Diagnose tissue coverage and compare split strategies.

WHY: tissue presence differs per subject (each subject's tissue_ids.json only lists the
organs that actually had meshes for that CT volume), so a naive random subject split can
leave a rare organ entirely out of val/test or skew its train coverage. This script does
NOT commit a split — it dumps the statistics we need to choose one well, and scores a few
candidate strategies so we can compare.

Run on the server (uses the project venv):

    venv/bin/python scripts/training/analyze_dataset_split.py \
        --data-root data/training_dataset --out results/split_analysis

    # also measure on-screen pixel coverage per tissue (slower; reads EXR IndexOB):
    venv/bin/python scripts/training/analyze_dataset_split.py \
        --data-root data/training_dataset --out results/split_analysis --coverage-views 2

Outputs (under --out):
    dataset_summary.json          overall stats + rare-tissue flags
    presence_matrix.csv           subjects × tissues, binary (from tissue_ids.json)
    tissue_summary.csv            per tissue: #subjects present, %, [pixel coverage]
    subject_summary.csv           per subject: #tissues, #views
    tissue_coocurrence.csv        tissue × tissue co-presence counts
    split_<name>.json             candidate subject→split assignment
    split_<name>_balance.csv      per-tissue counts in train/val/test for that candidate
    split_comparison.csv          one row per candidate: how balanced it is (pick from this)
    report.txt                    human-readable summary (paste this back to discuss)
"""
from __future__ import annotations

import sys
import csv
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr import read_render_exr


# ── scan ──────────────────────────────────────────────────────────────────────
def load_tissue_map(subj_dir: Path):
    """Return {tissue_name: pass_index} for a subject, or None."""
    for cand in (subj_dir / "tissue_ids.json", subj_dir.parent / "tissue_ids.json"):
        if cand.exists():
            try:
                return json.load(open(cand)).get("tissues", {})
            except Exception:
                pass
    return None


def scan_dataset(root: Path):
    subjects = {}
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        views = sorted(v for v in subj_dir.glob("v*") if (v / "meta.json").exists())
        if not views:
            continue
        tmap = load_tissue_map(subj_dir)
        if not tmap:
            print(f"[warn] {subj_dir.name}: no tissue_ids.json — skipped")
            continue
        subjects[subj_dir.name] = {"dir": subj_dir, "views": views, "tissues": tmap}
    return subjects


# ── pixel coverage (optional, sampled) ────────────────────────────────────────
def measure_coverage(subjects, n_views: int, max_subjects: int):
    """Mean on-screen pixel fraction per tissue, sampled over a few views per subject."""
    cover = defaultdict(list)
    names = sorted(subjects.keys())
    if max_subjects > 0:
        names = names[:max_subjects]
    for si, s in enumerate(names):
        info = subjects[s]
        inv = {int(v): k for k, v in info["tissues"].items()}   # id → name (per subject!)
        for vdir in info["views"][:n_views]:
            try:
                got = read_render_exr(str(vdir / "render.exr"), want=("indexob",))
            except Exception as e:
                print(f"[warn] coverage read failed {vdir}: {e}")
                continue
            if "indexob" not in got:
                continue
            ids = np.rint(got["indexob"]).astype(np.int64)
            total = ids.size
            uniq, counts = np.unique(ids, return_counts=True)
            present_frac = {int(u): c / total for u, c in zip(uniq, counts) if u > 0}
            for tid, name in inv.items():
                cover[name].append(present_frac.get(tid, 0.0))
        if (si + 1) % 50 == 0:
            print(f"[coverage] {si+1}/{len(names)} subjects")
    return {n: np.array(v, dtype=np.float64) for n, v in cover.items()}


# ── candidate splits ──────────────────────────────────────────────────────────
def split_random(subjects, val_frac, test_frac, seed):
    out = {}
    for s in subjects:
        h = int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) % 10000 / 10000.0
        out[s] = "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"
    return out


def split_iterative_stratified(subjects, vocab, val_frac, test_frac, seed):
    """
    Multi-label iterative stratification (Sechidis et al. 2011), labels = tissues present.
    Distributes subjects so each tissue is proportionally represented in train/val/test.
    """
    rng = np.random.default_rng(seed)
    splits = ["train", "val", "test"]
    target = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}

    subj_labels = {s: set(subjects[s]["tissues"].keys()) for s in subjects}
    n = len(subjects)
    # desired total per split, and desired per (split, label)
    desired = {sp: target[sp] * n for sp in splits}
    label_subj = {lab: {s for s in subjects if lab in subj_labels[s]} for lab in vocab}
    desired_l = {(sp, lab): target[sp] * len(label_subj[lab]) for sp in splits for lab in vocab}

    assign = {}
    remaining = set(subjects.keys())
    # subjects with no labels handled last by overall desired
    while remaining:
        # pick label with fewest remaining positive subjects (>0)
        rem_counts = {lab: len(label_subj[lab] & remaining) for lab in vocab}
        rem_counts = {k: v for k, v in rem_counts.items() if v > 0}
        if not rem_counts:
            break
        lab = min(rem_counts, key=lambda k: (rem_counts[k], -max(desired_l[(sp, k)] for sp in splits)))
        for s in sorted(label_subj[lab] & remaining):
            # split with largest desired_l for this label, tie → largest overall desired
            best = max(splits, key=lambda sp: (desired_l[(sp, lab)], desired[sp], rng.random()))
            assign[s] = best
            remaining.discard(s)
            desired[best] -= 1
            for l2 in subj_labels[s]:
                desired_l[(best, l2)] -= 1
    # leftover (label-less) subjects
    for s in sorted(remaining):
        best = max(splits, key=lambda sp: (desired[sp], rng.random()))
        assign[s] = best
        desired[best] -= 1
    return assign


# ── scoring ───────────────────────────────────────────────────────────────────
def score_split(assign, subjects, vocab, val_frac, test_frac):
    by_split = defaultdict(set)
    for s, sp in assign.items():
        by_split[sp].add(s)
    counts = {sp: len(by_split[sp]) for sp in ("train", "val", "test")}

    # per-tissue counts per split
    per_tissue = {}
    missing_val = missing_test = 0
    for lab in vocab:
        row = {}
        for sp in ("train", "val", "test"):
            row[sp] = sum(1 for s in by_split[sp] if lab in subjects[s]["tissues"])
        per_tissue[lab] = row
        if row["val"] == 0:  missing_val += 1
        if row["test"] == 0: missing_test += 1

    # deviation of each tissue's test/val fraction from target
    target = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    devs = []
    for lab in vocab:
        tot = sum(per_tissue[lab].values())
        if tot == 0:
            continue
        for sp in ("val", "test"):
            devs.append(abs(per_tissue[lab][sp] / tot - target[sp]))
    return {
        "counts": counts,
        "per_tissue": per_tissue,
        "tissues_missing_from_val": missing_val,
        "tissues_missing_from_test": missing_test,
        "mean_abs_frac_deviation": float(np.mean(devs)) if devs else float("nan"),
        "max_abs_frac_deviation": float(np.max(devs)) if devs else float("nan"),
    }


# ── writers ───────────────────────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/training_dataset")
    ap.add_argument("--out", default="results/split_analysis")
    ap.add_argument("--val-frac", type=float, default=0.075)
    ap.add_argument("--test-frac", type=float, default=0.075)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--coverage-views", type=int, default=0,
                    help="views/subject to measure pixel coverage (0 = skip; slow, reads EXR)")
    ap.add_argument("--coverage-max-subjects", type=int, default=0,
                    help="cap subjects for coverage sampling (0 = all)")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    subjects = scan_dataset(root)
    if not subjects:
        print(f"No subjects found under {root.resolve()}"); sys.exit(1)

    vocab = sorted({t for info in subjects.values() for t in info["tissues"]})
    n_sub = len(subjects)
    print(f"[scan] {n_sub} subjects, {len(vocab)} distinct tissues, "
          f"{sum(len(i['views']) for i in subjects.values())} views total")

    # presence matrix
    write_csv(out / "presence_matrix.csv", ["subject"] + vocab,
              [[s] + [int(t in subjects[s]["tissues"]) for t in vocab] for s in sorted(subjects)])

    # subject summary
    write_csv(out / "subject_summary.csv", ["subject", "n_tissues", "n_views"],
              [[s, len(subjects[s]["tissues"]), len(subjects[s]["views"])] for s in sorted(subjects)])

    # tissue presence counts
    pres_count = {t: sum(1 for s in subjects if t in subjects[s]["tissues"]) for t in vocab}

    # optional pixel coverage
    coverage = {}
    if args.coverage_views > 0:
        print(f"[coverage] sampling {args.coverage_views} view(s)/subject ...")
        coverage = measure_coverage(subjects, args.coverage_views, args.coverage_max_subjects)

    # tissue summary
    t_header = ["tissue", "n_subjects_present", "pct_subjects"]
    if coverage:
        t_header += ["mean_pixel_frac", "median_pixel_frac", "frac_views_visible"]
    t_rows = []
    for t in vocab:
        row = [t, pres_count[t], round(100 * pres_count[t] / n_sub, 2)]
        if coverage and t in coverage:
            arr = coverage[t]
            vis = arr[arr > 0]
            row += [round(float(arr.mean()), 6),
                    round(float(np.median(arr)), 6),
                    round(float((arr > 0).mean()), 4)]
        elif coverage:
            row += ["", "", ""]
        t_rows.append(row)
    write_csv(out / "tissue_summary.csv", t_header, t_rows)

    # co-occurrence
    cooc = np.zeros((len(vocab), len(vocab)), dtype=np.int64)
    idx = {t: i for i, t in enumerate(vocab)}
    for s in subjects:
        ts = [idx[t] for t in subjects[s]["tissues"]]
        for a in ts:
            for b in ts:
                cooc[a, b] += 1
    write_csv(out / "tissue_coocurrence.csv", ["tissue"] + vocab,
              [[vocab[i]] + cooc[i].tolist() for i in range(len(vocab))])

    # candidate splits
    candidates = {
        "random": split_random(subjects, args.val_frac, args.test_frac, args.seed),
        "stratified": split_iterative_stratified(subjects, vocab, args.val_frac, args.test_frac, args.seed),
    }
    comparison = []
    for name, assign in candidates.items():
        json.dump(assign, open(out / f"split_{name}.json", "w"), indent=2)
        sc = score_split(assign, subjects, vocab, args.val_frac, args.test_frac)
        write_csv(out / f"split_{name}_balance.csv",
                  ["tissue", "train", "val", "test"],
                  [[t, sc["per_tissue"][t]["train"], sc["per_tissue"][t]["val"],
                    sc["per_tissue"][t]["test"]] for t in vocab])
        comparison.append([name, sc["counts"]["train"], sc["counts"]["val"], sc["counts"]["test"],
                           sc["tissues_missing_from_val"], sc["tissues_missing_from_test"],
                           round(sc["mean_abs_frac_deviation"], 5), round(sc["max_abs_frac_deviation"], 5)])
    write_csv(out / "split_comparison.csv",
              ["candidate", "n_train", "n_val", "n_test", "tissues_missing_val",
               "tissues_missing_test", "mean_abs_frac_dev", "max_abs_frac_dev"],
              comparison)

    # dataset summary json
    rare = [t for t in vocab if pres_count[t] < max(1, int(0.05 * n_sub))]
    summary = {
        "n_subjects": n_sub,
        "n_tissues": len(vocab),
        "n_views_total": sum(len(i["views"]) for i in subjects.values()),
        "tissues_per_subject": {
            "min": min(len(i["tissues"]) for i in subjects.values()),
            "max": max(len(i["tissues"]) for i in subjects.values()),
            "mean": round(float(np.mean([len(i["tissues"]) for i in subjects.values()])), 2),
        },
        "rare_tissues_lt5pct": rare,
        "coverage_measured": bool(coverage),
    }
    json.dump(summary, open(out / "dataset_summary.json", "w"), indent=2)

    # report.txt + console
    lines = []
    lines.append(f"DATASET: {n_sub} subjects | {len(vocab)} tissues | "
                 f"{summary['n_views_total']} views")
    lines.append(f"tissues/subject: min={summary['tissues_per_subject']['min']} "
                 f"max={summary['tissues_per_subject']['max']} "
                 f"mean={summary['tissues_per_subject']['mean']}")
    lines.append("")
    lines.append("TISSUE PRESENCE (sorted, rarest first):")
    for t in sorted(vocab, key=lambda x: pres_count[x]):
        flag = "  <-- RARE" if t in rare else ""
        cov = ""
        if coverage and t in coverage:
            cov = f"  meanpix={coverage[t].mean():.4f}"
        lines.append(f"  {t:<34} {pres_count[t]:5d} subj ({100*pres_count[t]/n_sub:5.1f}%){cov}{flag}")
    lines.append("")
    lines.append("SPLIT CANDIDATE COMPARISON (lower missing / deviation = better):")
    lines.append(f"  {'candidate':<12} {'train':>6} {'val':>5} {'test':>5} "
                 f"{'miss_val':>9} {'miss_test':>10} {'mean_dev':>9} {'max_dev':>8}")
    for r in comparison:
        lines.append(f"  {r[0]:<12} {r[1]:>6} {r[2]:>5} {r[3]:>5} "
                     f"{r[4]:>9} {r[5]:>10} {r[6]:>9} {r[7]:>8}")
    report = "\n".join(lines)
    open(out / "report.txt", "w").write(report + "\n")
    print("\n" + report)
    print(f"\n[done] wrote analysis to {out.resolve()}")


if __name__ == "__main__":
    main()
