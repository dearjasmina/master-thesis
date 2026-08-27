"""
Plot test-set metrics comparison: RGB baseline vs G-buffer model.
RGB CSV is optional — if missing, only G-buffer is plotted.
Outputs figures/test/comparison_test.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
OUT  = HERE / "test" / "comparison_test.png"

GBUF_PATH = HERE / "test" / "metrics_test_gbuffer.csv"
RGB_PATH  = HERE / "test" / "metrics_test_rgb.csv"

gbuf = pd.read_csv(GBUF_PATH)
rgb  = pd.read_csv(RGB_PATH) if RGB_PATH.exists() else None

# ── palette ───────────────────────────────────────────────────────────────────
RGB_COL  = "#E07B54"
GBUF_COL = "#5B8DB8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

metrics = [
    ("psnr",  "PSNR (dB)",        True),
    ("ssim",  "SSIM",             True),
    ("lpips", "LPIPS (↓ better)", False),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
title = "Test-set metrics — RGB baseline vs G-buffer model" if rgb is not None \
        else "Test-set metrics — G-buffer model"
fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

has_rgb = rgb is not None

for ax, (col, ylabel, higher_better) in zip(axes, metrics):
    # aggregate: mean per subject, then distribution across subjects
    gbuf_subj = gbuf.groupby("subject")[col].mean()

    if has_rgb:
        rgb_subj = rgb.groupby("subject")[col].mean()
        positions = [1, 2]
        data      = [rgb_subj.values, gbuf_subj.values]
        colors    = [RGB_COL, GBUF_COL]
        labels    = ["RGB baseline", "G-buffer"]
    else:
        positions = [1]
        data      = [gbuf_subj.values]
        colors    = [GBUF_COL]
        labels    = ["G-buffer"]

    bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, linestyle="none", alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for flier, color in zip(bp["fliers"], colors):
        flier.set(markerfacecolor=color, markeredgecolor=color)
    for whisker, color in zip(bp["whiskers"], sum([[c, c] for c in colors], [])):
        whisker.set(color=color)
    for cap, color in zip(bp["caps"], sum([[c, c] for c in colors], [])):
        cap.set(color=color)

    # annotate medians
    for pos, d, color in zip(positions, data, colors):
        med = np.median(d)
        fmt = f"{med:.3f}" if col != "psnr" else f"{med:.2f} dB"
        ax.text(pos, med + (np.ptp(d) * 0.04), fmt,
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontweight="bold")
    ax.set_xlim(0.4, max(positions) + 0.6)

    arrow = "↑ better" if higher_better else "↓ better"
    ax.text(0.98, 0.02, arrow, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray", style="italic")

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
if not has_rgb:
    print("(RGB test CSV not found — plotted G-buffer only. Re-run once metrics_test_rgb.csv is ready.)")
