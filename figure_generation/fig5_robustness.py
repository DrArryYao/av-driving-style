"""Fig 5: Visual robustness summary (replaces Table 2)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\Arry\Desktop\NC\paper"
HERE = os.path.dirname(os.path.abspath(__file__))
AVC, HC, NC = "#D55E00", "#0072B2", "#009E73"
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.9, "figure.dpi": 200,
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
    "legend.fontsize": 6, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "lines.linewidth": 1.2,
})
L = lambda ax, s: ax.text(-0.18, 1.1, s, transform=ax.transAxes,
                          fontsize=10, fontweight="bold", va="top", ha="left")
SQ = lambda ax: ax.set_box_aspect(1)
WIDE = lambda ax: ax.set_box_aspect(0.5)


def fig5():
    p = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    sens = pd.read_csv(os.path.join(HERE, "p12_sensitivity.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.5))
    fig.subplots_adjust(hspace=0.15, wspace=0.55)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: Forest plot — all robustness checks (wide, 2:1)
    analyses = [
        ("aggregation (median)", -9.1, -14.2, "jerk diff (m s$^{-3}$)"),
        ("duration (11.3 s)", -10.1, -14.3, "jerk diff (m s$^{-3}$)"),
        ("threshold 0.2 m s$^{-1}$", 0.89, 0.75, "AV gain"),
        ("threshold 0.4 m s$^{-1}$", 0.64, 0.75, "AV gain"),
        ("energy (wheel model)", -54, -52, "VSP/km saving (%)"),
        ("vehicle (minivan)", -54, -52, "saving (%)"),
        ("leader noise (eq.)", 0.84, 1.24, "H|AV gain"),
    ]
    for i,(name, var, base, unit) in enumerate(analyses):
        ax1.plot([var, base], [i, i], color="#999", lw=2)
        ax1.scatter(var, i, color=AVC, s=35, zorder=3)
        ax1.scatter(base, i, color=HC, s=35, zorder=3)
    ax1.set_yticks(range(len(analyses)))
    ax1.set_yticklabels([a[0] for a in analyses], fontsize=6)
    ax1.invert_yaxis()
    ax1.set_xlabel("variation (red) vs baseline (blue)")
    ax1.text(0.02, 0.98, "blue = baseline", transform=ax1.transAxes,
             fontsize=5.5, color=HC, va="top")
    ax1.text(0.02, 0.93, "red = variation", transform=ax1.transAxes,
             fontsize=5.5, color=AVC, va="top")
    WIDE(ax1); L(ax1, "a")

    # b: Design effect (1:1)
    de_data = [0.94, 1.05, 1.09, 1.02]  # from cluster bootstrap
    de_labels = ["jerk", "aggr.", "brake", "VSP"]
    ax2.barh(range(4), de_data, color=["#8899aa"]*4, height=0.6)
    ax2.axvline(1, color="k", lw=0.8, ls="--")
    ax2.set_yticks(range(4)); ax2.set_yticklabels(de_labels, fontsize=6.5)
    ax2.set_xlabel("design effect (1 = independent)")
    ax2.set_xlim(0.8, 1.2)
    SQ(ax2); L(ax2, "b")

    # c: Threshold sensitivity (1:1)
    thresholds = [0.2, 0.3, 0.4]
    av_gains = [0.89, 0.75, 0.64]
    ax3.plot(thresholds, av_gains, "o-", color=AVC, ms=6, lw=1.5)
    ax3.axhline(0.95, color=HC, lw=1, ls="--", label="human baseline")
    ax3.fill_between(thresholds, av_gains, 0.95, alpha=0.15, color=AVC)
    ax3.set_xlabel("threshold (m s$^{-1}$)")
    ax3.set_ylabel("AV median gain")
    ax3.legend(fontsize=6)
    ax3.annotate("p < 10$^{-25}$ at\nall thresholds",
                 (0.3, 0.82), fontsize=6, ha="center", color="#333")
    SQ(ax3); L(ax3, "c")

    # d: Estimator bias (1:1)
    bias_data = pd.read_csv(os.path.join(HERE, "p02_gain_bias.csv"))
    s = bias_data[(bias_data.noise==0.15)&(bias_data.L==150)&(bias_data.g_true==1.0)]
    s = s.sort_values("sigma_d")
    ax4.axhspan(0.8, 0.9, alpha=0.15, color=AVC)
    ax4.axhline(1, color="k", lw=0.8, ls="--")
    ax4.plot(s.sigma_d, s.bias_ratio, "o-", color="#CC79A7", ms=4, lw=1.2)
    ax4.axvline(0.3, color="k", lw=0.8, ls=":")
    ax4.set_xscale("log")
    ax4.set_xlabel("disturbance (m s$^{-1}$)")
    ax4.set_ylabel("est. / true gain")
    ax4.text(0.02, 0.02, "0.83$\\times$ at all gains\n(contrasts unbiased)",
             transform=ax4.transAxes, fontsize=5.5, va="bottom")
    SQ(ax4); L(ax4, "d")

    # e: Controller sensitivity (1:1)
    base_e = 41.09
    for n_av, lab, c in [(12, "20%", "#56B4E9"), (30, "50%", "#E69F00"),
                          (60, "100%", AVC)]:
        sub = sens[sens.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).e.mean()
        sav = 100*(1-sub/base_e)
        ax5.scatter([lab]*len(sav), sav, color=c, s=15, alpha=0.5)
        ax5.scatter([lab], [sav.median()], color=AVC, s=35, zorder=3)
    ax5.set_xlabel("penetration")
    ax5.set_ylabel("energy saving (%)")
    SQ(ax5); L(ax5, "e")

    # f: Evidence summary heatmap (1:1)
    checks = ["style gap", "energy gap", "stability", "robustness"]
    criteria = ["magnitude", "significance", "generality", "bias direction"]
    evidence = np.array([
        [3, 3, 3, 3],  # style gap: large, significant, general, conservative
        [3, 3, 2, 3],  # energy gap: large, significant, regime-limited, conservative
        [3, 3, 2, 2],  # stability: large, significant, conditional, less clear
        [3, 3, 3, 3],  # robustness: large, significant, all checks pass, conservative
    ])
    im6 = ax6.imshow(evidence, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
    ax6.set_xticks(range(4)); ax6.set_xticklabels(criteria, fontsize=5.5, rotation=30)
    ax6.set_yticks(range(4)); ax6.set_yticklabels(checks, fontsize=6)
    for i in range(4):
        for j in range(4):
            label = ["", "weak", "moderate", "strong"][evidence[i,j]]
            ax6.text(j, i, label, ha="center", va="center", fontsize=6,
                     color="white" if evidence[i,j]>2 else "black",
                     fontweight="bold" if evidence[i,j]==3 else "normal")
    SQ(ax6); L(ax6, "f")

    fig.savefig(os.path.join(OUT, "fig5_robustness.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig5(); print("Fig 5: robustness summary (a=2:1 wide, b-f=1:1)")
