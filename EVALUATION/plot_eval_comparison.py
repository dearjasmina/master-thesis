"""
Plot eval metrics comparison: RGB baseline vs G-buffer model.
Outputs figures/eval/comparison_eval.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
OUT  = HERE / "eval" / "comparison_eval.png"

rgb   = pd.read_csv(HERE / "eval" / "metrics_history_rgb.csv")
gbuf  = pd.read_csv(HERE / "eval" / "metrics_history–gbuffer.csv")

rgb_train  = rgb[rgb.split == "train"]
rgb_val    = rgb[rgb.split == "val"]
gbuf_train = gbuf[gbuf.split == "train"]
gbuf_val   = gbuf[gbuf.split == "val"]

# ── palette ───────────────────────────────────────────────────────────────────
RGB_COL  = "#E07B54"   # warm orange  — RGB model
GBUF_COL = "#5B8DB8"   # steel blue   — G-buffer model
TRAIN_ALPHA = 0.35
VAL_ALPHA   = 1.0

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
    ("psnr",  "PSNR (dB)",             True),   # higher is better
    ("ssim",  "SSIM",                  True),
    ("lpips", "LPIPS (↓ better)",      False),  # lower is better
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Eval metrics — RGB baseline vs G-buffer model", fontsize=14, fontweight="bold", y=1.01)

for ax, (col, ylabel, higher_better) in zip(axes, metrics):
    # train curves (faded)
    ax.plot(rgb_train.epoch,  rgb_train[col],  color=RGB_COL,  alpha=TRAIN_ALPHA, lw=1.5, ls="--")
    ax.plot(gbuf_train.epoch, gbuf_train[col], color=GBUF_COL, alpha=TRAIN_ALPHA, lw=1.5, ls="--")

    # val curves (solid, main)
    ax.plot(rgb_val.epoch,  rgb_val[col],  color=RGB_COL,  alpha=VAL_ALPHA, lw=2.2, label="RGB — val")
    ax.plot(gbuf_val.epoch, gbuf_val[col], color=GBUF_COL, alpha=VAL_ALPHA, lw=2.2, label="G-buffer — val")

    # annotate final val values
    for df, col_c, name in [(rgb_val, RGB_COL, "RGB"), (gbuf_val, GBUF_COL, "G-buf")]:
        last = df.iloc[-1]
        val  = last[col]
        fmt  = f"{val:.3f}" if col != "psnr" else f"{val:.2f} dB"
        ax.annotate(fmt, xy=(last.epoch, val),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=col_c, fontweight="bold", va="center")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

# shared legend (train dashed / val solid)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=RGB_COL,  lw=2.2, label="RGB baseline — val"),
    Line2D([0], [0], color=GBUF_COL, lw=2.2, label="G-buffer — val"),
    Line2D([0], [0], color="gray",   lw=1.5, ls="--", alpha=0.6, label="train (dashed)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
