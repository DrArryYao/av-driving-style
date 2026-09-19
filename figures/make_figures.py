"""Publication figures (Nature style): boxed axes, bold lowercase panel
letters, high panel density. Manuscript Figs 1-4 + Extended Data 1-4."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"${OUT_DIR}"
# Okabe-Ito colour-blind-safe palette
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
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def ecdf(ax, series, color):
    v = np.sort(series)
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=1.1)


# ---------------- Fig 1: style ECDFs, 8 panels ----------------
def fig1():
    tk = pd.read_csv("${DATA_DIR}/tracks_full2.csv",
                     usecols=["is_av", "p95_abs_jerk", "mean_abs_accel",
                              "frac_aggr", "frac_hard_brake", "p99_abs_accel",
                              "p01_accel", "mean_vsp", "frac_high_vsp",
                              "std_speed"])
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.4))
    metrics = [("p95_abs_jerk", "95th p. |jerk| (m s$^{-3}$)", True),
               ("mean_abs_accel", "mean |accel.| (m s$^{-2}$)", True),
               ("p99_abs_accel", "99th p. accel. (m s$^{-2}$)", True),
               ("p01_accel", "1st p. accel. (m s$^{-2}$)", True),
               ("frac_aggr", "aggressive share", False),
               ("frac_hard_brake", "hard-brake share", False),
               ("mean_vsp", "mean VSP (kW t$^{-1}$)", True),
               ("std_speed", "speed s.d. (m s$^{-1}$)", True)]
    letters = "abcdefgh"
    for ax, (m, lab, logx), let in zip(axes.flat, metrics, letters):
        ecdf(ax, tk.loc[tk.is_av == 0, m], HC)
        ecdf(ax, tk.loc[tk.is_av == 1, m], AVC)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        letter(ax, let)
    axes.flat[0].set_ylabel("ECDF")
    axes.flat[4].set_ylabel("ECDF")
    axes.flat[0].legend(["Human vehicles", "Waymo AV"], fontsize=6.5,
                        loc="lower right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig1_ecdf.pdf"))
    plt.close(fig)


# ---------------- Fig 2: string stability, 4 panels ----------------
def fig3():
    wp = pd.read_csv("${DATA_DIR}/pairs_full.csv",
                     usecols=["follower_is_av", "mean_speed", "gain",
                              "leader_fluct_std"])
    ng = pd.read_csv("${DATA_DIR}/ngsim_pairs.csv",
                     usecols=["mean_speed", "gain", "leader_fluct_std"])
    wf = wp[wp.mean_speed > 10]
    nf = ng[ng.mean_speed > 10]
    strata = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 3.0)]
    mids = [np.mean(s) for s in strata]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    (ax1, ax2), (ax3, ax4) = axes

    for series, name, c in [(wf[wf.follower_is_av == 1], "Waymo AV", AVC),
                            (wf[wf.follower_is_av == 0], "Human (perceived)", HC),
                            (nf, "Human (NGSIM)", NC)]:
        med, q1, q3 = [], [], []
        for lo, hi in strata:
            s = series[(series.leader_fluct_std >= lo) & (series.leader_fluct_std < hi)]
            if len(s) >= 30:
                med.append(s.gain.median()); q1.append(s.gain.quantile(.25)); q3.append(s.gain.quantile(.75))
            else:
                med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        ax1.plot(mids, med, "o-", color=c, label=name, ms=4, lw=1.3)
        ax1.fill_between(mids, q1, q3, color=c, alpha=0.15, lw=0)
    ax1.axhline(1, color="k", lw=0.8, ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("leader disturbance (m s$^{-1}$)")
    ax1.set_ylabel("gain (median, IQR)")
    ax1.legend(loc="upper right")
    letter(ax1, "a")

    for series, name, c in [(wf[(wf.follower_is_av == 1) & (wf.leader_fluct_std >= 0.3)], "Waymo AV", AVC),
                            (nf[nf.leader_fluct_std >= 0.3], "Human (NGSIM)", NC)]:
        ecdf(ax2, series.gain, c)
        ax2.annotate(name, (0.05, 0.88 if c == AVC else 0.76),
                     xycoords="axes fraction", fontsize=6.5, color=c)
    ax2.axvline(1, color="k", lw=0.8, ls="--")
    ax2.set_xlim(0, 3)
    ax2.set_xlabel("gain, disturbances $\\geq$ 0.3 m s$^{-1}$")
    ax2.set_ylabel("ECDF")
    letter(ax2, "b")

    nm = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "p01_pairs.csv"))
    s = nm[(nm.mean_speed > 10) & (nm.l_fluct >= 0.3)]
    groups = [((0, 0), "g_raw", "human|human", HC),
              ((0, 1), "g_raw", "human|AV\n(raw)", "#88aacc"),
              ((0, 1), "g_nm", "human|AV\n(equalized)", "#1f4e79"),
              ((1, 0), "g_raw", "AV|human", AVC)]
    for i, ((f, l), mode, lab, c) in enumerate(groups):
        g = s[(s.f_av == f) & (s.l_av == l)][mode]
        rng2 = np.random.default_rng(1)
        boots = [np.median(rng2.choice(g.values, g.size)) for _ in range(1000)]
        med = np.median(g)
        lo_, hi_ = np.percentile(boots, [2.5, 97.5])
        ax3.errorbar([i], [med], yerr=[[med - lo_], [hi_ - med]],
                     fmt="o", color=c, capsize=3, ms=5)
    ax3.axhline(1, color="k", lw=0.8, ls="--")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([g[2] for g in groups], fontsize=7)
    ax3.set_ylabel("gain (median, 95% CI)")
    letter(ax3, "c")

    wf2 = wp[wp.mean_speed > 5]
    xedges = np.array([0.05, 0.15, 0.3, 0.6, 3.0])
    yedges = np.array([5, 10, 15, 20, 35])
    H_av = np.full((4, 4), np.nan)
    H_hu = np.full((4, 4), np.nan)
    for xi in range(4):
        for yi in range(4):
            msk = (wf2.leader_fluct_std >= xedges[xi]) & (wf2.leader_fluct_std < xedges[xi + 1]) & \
                (wf2.mean_speed >= yedges[yi]) & (wf2.mean_speed < yedges[yi + 1])
            a = wf2[msk & (wf2.follower_is_av == 1)].gain
            h = wf2[msk & (wf2.follower_is_av == 0)].gain
            if len(a) > 50:
                H_av[yi, xi] = a.median()
            if len(h) > 50:
                H_hu[yi, xi] = h.median()
    D = H_av - H_hu
    im = ax4.imshow(D, cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower",
                    aspect="auto")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels([".05-.15", ".15-.3", ".3-.6", ".6-3"], fontsize=6.5)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(["5-10", "10-15", "15-20", "20-35"], fontsize=6.5)
    ax4.set_xlabel("leader disturbance (m s$^{-1}$)")
    ax4.set_ylabel("speed (m s$^{-1}$)")
    for xi in range(4):
        for yi in range(4):
            if not np.isnan(D[yi, xi]):
                ax4.text(xi, yi, f"{D[yi, xi]:.2f}", ha="center", va="center",
                         fontsize=6.5, color="k")
    cb = fig.colorbar(im, ax=ax4, fraction=0.045)
    cb.set_label("AV $-$ human gain", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    letter(ax4, "d")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_string.pdf"))
    plt.close(fig)


# ---------------- Fig 3 (ms): simulation, 3 panels ----------------
def fig4():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "platoon_results.csv"))
    g = df.groupby("n_av").agg(e=("e_pos_kWh_100km", "mean"),
                               sd=("e_pos_kWh_100km", "std"),
                               vmean=("throughput_vmean", "mean"),
                               vstd=("fleet_v_std", "mean"),
                               stops=("stop_frac", "mean")).reset_index()
    g["p"] = 100 * g.n_av / 60
    base = g.e.iloc[0]
    sens = pd.read_csv(os.path.join(here, "p12_sensitivity.csv"))
    b = 41.09

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for n_av, in [(12,), (30,), (60,)]:
        sub = sens[sens.n_av == n_av].groupby(["jerk_cap", "t_close", "T_av"]).e.mean()
        sav = 100 * (1 - sub / b)
        ax1.scatter([100 * n_av / 60] * len(sav), sav, color="#999999",
                    s=10, alpha=0.5, zorder=2)
    ax1.errorbar(g.p, 100 * (1 - g.e / base), yerr=100 * g.sd / base,
                 fmt="o-", color=AVC, capsize=2, ms=4, lw=1.3, zorder=3)
    ax1.set_xlabel("AV penetration (%)")
    ax1.set_ylabel("energy saving (%)")
    letter(ax1, "a")

    ax2.plot(g.p, g.vmean, "o-", color=HC, ms=4)
    ax2.set_xlabel("AV penetration (%)")
    ax2.set_ylabel("mean speed (m s$^{-1}$)", color=HC)
    ax2b = ax2.twinx()
    ax2b.plot(g.p, 100 * g.stops, "s--", color=AVC, ms=3.5, lw=1.1)
    ax2b.set_ylabel("stopped time (%)", color=AVC)
    ax2b.tick_params(axis="y", labelcolor=AVC, labelsize=7)
    ax2.tick_params(axis="y", labelcolor=HC)
    letter(ax2, "b")

    ax3.plot(g.p, g.vstd, "o-", color="#666666", ms=4)
    ax3.set_xlabel("AV penetration (%)")
    ax3.set_ylabel("fleet speed s.d. (m s$^{-1}$)")
    letter(ax3, "c")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sim.pdf"))
    plt.close(fig)


# ---------------- Fig 4 (ms): measured energy, 4 panels ----------------
def fig6():
    tk = pd.read_csv("${DATA_DIR}/tracks_full2.csv",
                     usecols=["is_av", "mean_speed", "mean_vsp",
                              "sum_v", "sum_v3", "sum_va"])
    tk = tk[tk.mean_speed > 1].copy()
    m, g, rho, CdA, Crr = 1800.0, 9.81, 1.2, 0.70, 0.010
    tk["vsp_km"] = tk.mean_vsp / tk.mean_speed
    tk["e_km"] = (m * tk.sum_va + m * g * Crr * tk.sum_v
                  + 0.5 * rho * CdA * tk.sum_v3) / 1000.0 / (tk.sum_v / 1000.0)
    tk["regime"] = pd.cut(tk.mean_speed, [1, 5, 10, 15, 100],
                          labels=["0–5", "5–10", "10–15", ">15"])
    regs = ["0–5", "5–10", "10–15", ">15"]

    def med_ci(col):
        out = {}
        for r in regs:
            for key in [0, 1]:
                g_ = tk.loc[(tk.regime == r) & (tk.is_av == key), col]
                rng2 = np.random.default_rng(7)
                boots = [np.median(rng2.choice(g_.values, g_.size)) for _ in range(500)]
                out[(r, key)] = (np.median(g_), np.percentile(boots, [2.5, 97.5]))
        return out

    med_v = med_ci("vsp_km")
    med_e = med_ci("e_km")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    (ax1, ax2), (ax3, ax4) = axes
    x = np.arange(len(regs))

    for key, lab, c, off in [(0, "Human vehicles", HC, -0.18),
                             (1, "Waymo AV", AVC, 0.18)]:
        vals = [med_v[(r, key)][0] for r in regs]
        err = ([med_v[(r, key)][0] - med_v[(r, key)][1][0] for r in regs],
               [med_v[(r, key)][1][1] - med_v[(r, key)][0] for r in regs])
        ax1.bar(x + off, vals, width=0.34, color=c, label=lab,
                yerr=err, capsize=2, error_kw=dict(lw=0.8))
    ax1.set_xticks(x); ax1.set_xticklabels(regs)
    ax1.set_xlabel("mean speed (m s$^{-1}$)")
    ax1.set_ylabel("VSP per km (kJ t$^{-1}$ km$^{-1}$)")
    ax1.legend()
    letter(ax1, "a")

    w = 0.35
    for i, (col, lab, c) in enumerate([(med_v, "VSP proxy", "#8899aa"),
                                       (med_e, "wheel-level model", "#444444")]):
        rel = [100 * (col[(r, 1)][0] / col[(r, 0)][0] - 1) for r in regs]
        ax2.bar(x + (i - 0.5) * w, rel, width=w, color=c, label=lab)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(regs)
    ax2.set_xlabel("mean speed (m s$^{-1}$)")
    ax2.set_ylabel("AV relative change (%)")
    ax2.set_ylim(-80, 15)
    ax2.legend()
    letter(ax2, "b")

    for key, lab, c, off in [(0, "Human vehicles", HC, -0.18),
                             (1, "Waymo AV", AVC, 0.18)]:
        vals = [med_e[(r, key)][0] for r in regs]
        err = ([med_e[(r, key)][0] - med_e[(r, key)][1][0] for r in regs],
               [med_e[(r, key)][1][1] - med_e[(r, key)][0] for r in regs])
        ax3.bar(x + off, vals, width=0.34, color=c, label=lab,
                yerr=err, capsize=2, error_kw=dict(lw=0.8))
    ax3.set_xticks(x); ax3.set_xticklabels(regs)
    ax3.set_xlabel("mean speed (m s$^{-1}$)")
    ax3.set_ylabel("wheel energy (kWh per 100 km)")
    ax3.legend()
    letter(ax3, "c")

    m2, CdA2 = 2100.0, 0.95
    tk["e_km2"] = (m2 * tk.sum_va + m2 * g * Crr * tk.sum_v
                   + 0.5 * rho * CdA2 * tk.sum_v3) / 1000.0 / (tk.sum_v / 1000.0)
    sav1, sav2 = [], []
    for r in regs:
        h1 = tk[(tk.regime == r) & (tk.is_av == 0)].e_km.median()
        a1 = tk[(tk.regime == r) & (tk.is_av == 1)].e_km.median()
        h2 = tk[(tk.regime == r) & (tk.is_av == 0)].e_km2.median()
        a2 = tk[(tk.regime == r) & (tk.is_av == 1)].e_km2.median()
        sav1.append(100 * (a1 / h1 - 1))
        sav2.append(100 * (a2 / h2 - 1))
    ax4.scatter(sav1, sav2, c=[HC, AVC, AVC, AVC], s=45, zorder=3)
    lims = [min(sav1 + sav2) - 8, max(sav1 + sav2) + 8]
    ax4.plot(lims, lims, "k--", lw=0.8)
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_xlabel("saving, sedan parameters (%)")
    ax4.set_ylabel("saving, minivan parameters (%)")
    for i, r in enumerate(regs):
        ax4.annotate(r, (sav1[i], sav2[i]), textcoords="offset points",
                     xytext=(6, 3), fontsize=6.5)
    letter(ax4, "d")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_energy.pdf"))
    plt.close(fig)


# ------------- Extended Data figures -------------
def ed1():
    df = pd.read_csv("p02_gain_bias.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for noise, c in [(0.0, "#999999"), (0.05, "#56B4E9"), (0.15, "#CC79A7")]:
        s = df[(df.noise == noise) & (df.L == 150) & (df.g_true == 1.0)]
        ax1.plot(s.sigma_d, s.bias_ratio, "o-", color=c, ms=3.5, lw=1.1,
                 label=f"noise={noise}")
        s2 = df[(df.noise == noise) & (df.sigma_d == 0.6) & (df.g_true == 1.0)]
        ax2.plot(s2.L, s2.bias_ratio, "s-", color=c, ms=3.5, lw=1.1)
    ax1.axhline(1, color="k", lw=0.8, ls="--")
    ax1.axvline(0.3, color="k", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("true disturbance amplitude (m s$^{-1}$)")
    ax1.set_ylabel("estimated / true gain")
    ax1.legend()
    letter(ax1, "a")
    ax2.axhline(1, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("window length (steps)")
    ax2.set_ylabel("estimated / true gain")
    letter(ax2, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed1_bias.pdf")); plt.close(fig)


def ed2():
    df = pd.read_csv("platoon_probe.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for cfg, lab, c in [("human", "all human", HC), ("av", "all AV", AVC)]:
        s = df[df.config == cfg].sort_values("position")
        ax.plot(s.position, s.gain, "o-", color=c, ms=3.5, lw=1.3, label=lab)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("position in platoon")
    ax.set_ylabel("chain fluctuation gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed2_controller.pdf")); plt.close(fig)


def ed3():
    df = pd.read_csv("p12_sensitivity.csv")
    base = 41.09
    g = df.groupby(["n_av", "jerk_cap", "t_close", "T_av"]).e.mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for n_av, lab in [(12, "20%"), (30, "50%"), (60, "100%")]:
        sub = g[g.n_av == n_av]
        sav = 100 * (1 - sub.e / base)
        ax.scatter([lab] * len(sav), sav, color="#999999", s=12, alpha=0.6)
        ax.scatter([lab], [sav.median()], color=AVC, s=30, zorder=3)
    ax.set_xlabel("AV penetration")
    ax.set_ylabel("energy saving (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed3_sensitivity.pdf")); plt.close(fig)


def ed4():
    tk = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                     usecols=["is_av", "mean_speed", "p95_abs_jerk", "frac_aggr"])
    ng = pd.read_csv("${DATA_DIR}/ngsim_tracks.csv",
                     usecols=["mean_speed", "p95_abs_jerk", "frac_aggr"])
    w_h = tk[(tk.is_av == 0) & (tk.mean_speed > 10)]
    w_a = tk[(tk.is_av == 1) & (tk.mean_speed > 10)]
    ng_h = ng[ng.mean_speed > 10]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for s, lab, c in [(w_h, "Human, AV-perceived", HC),
                      (ng_h, "Human, camera (NGSIM)", NC),
                      (w_a, "Waymo AV", AVC)]:
        ecdf(ax1, s.p95_abs_jerk, c)
        ecdf(ax2, s.frac_aggr, c)
        ax1._proxy_labels = getattr(ax1, "_proxy_labels", []) + [(lab, c)]
    ax1.set_xscale("log")
    ax1.set_xlabel("95th p. $|jerk|$ (m s$^{-3}$)")
    ax2.set_xlabel("aggressive-driving share")
    ax1.set_ylabel("ECDF")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=1.2, label=l)
               for l, c in getattr(ax1, "_proxy_labels", [])]
    ax1.legend(handles=handles, fontsize=6)
    letter(ax1, "a"); letter(ax2, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed4_validation.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 done")
    fig3(); print("fig2(string) done")
    fig4(); print("fig3(sim) done")
    fig6(); print("fig4(energy) done")
    ed1(); ed2(); ed3(); ed4(); print("ED done")
