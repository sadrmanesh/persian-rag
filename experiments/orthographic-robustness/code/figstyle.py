"""Shared publication style: Okabe-Ito palette, fonts, Persian text shaping, export.

Author: Mohammad Naser Sadrmanesh
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# Vazirmatn is used for Persian labels. Override with VAZIRMATN_DIR, or place
# the TTFs in ../fonts (next to this code/ directory).
FONT_DIR = Path(os.environ["VAZIRMATN_DIR"]) if os.environ.get("VAZIRMATN_DIR") else (
    Path(__file__).resolve().parents[1] / "fonts"
)
for f in FONT_DIR.glob("Vazirmatn-*.ttf"):
    font_manager.fontManager.addfont(str(f))

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FA_FONT = font_manager.FontProperties(fname=str(FONT_DIR / "Vazirmatn-Regular.ttf"))
META = {"Author": "Mohammad Naser Sadrmanesh", "Title": None}


def fa(text):
    """Persian text for matplotlib (>= 3.11 shapes and orders RTL text natively)."""
    return text


def save(fig, path_stem, title):
    path_stem = Path(path_stem)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight",
                metadata={"Author": "Mohammad Naser Sadrmanesh", "Title": title, "Software": "matplotlib"})
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"Author": "Mohammad Naser Sadrmanesh", "Title": title, "Creator": "matplotlib"})
    plt.close(fig)
