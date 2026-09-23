"""Publication figures, redesigned with diverse panel types (Nature style)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

OUT = r"C:\Users\Arry\Desktop\NC\paper"
AVC, HC, NC = "#D55E00", "#0072B2", "#009E73"

plt.rcParams.update({
    "font.size": 8.5, "axes.linewidth": 0.9, "figure.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
    "font.family": "Palatino Linotype",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic",
    "mathtext.bf": "Palatino Linotype:bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.fontsize": 6.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.linewidth": 1.3,
})

def letter(ax, s):
    ax.text(-0.18, 1.15, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

def ecdf(ax, series, color, **kw):
    v = np.sort(series.dropna())
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=1.1, **kw)
