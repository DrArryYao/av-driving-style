"""Publication figures for the manuscript (PDF output for LaTeX).

fig1: driving-style ECDFs, AV vs in-scene humans (full data)
fig2: paired differences by speed regime with 95% CIs
fig3: disturbance-conditional string-stability gains (AV / WOMD humans / NGSIM)
fig4: platoon energy vs AV penetration
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"${OUT_DIR}"
AVC, HC, NC = "#c0392b", "#2471a3", "#7d3c98"  # AV red, human blue, NGSIM purple
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.bbox": "tight", "legend.frameon": False,
    "font.family": "Palatino Linotype",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic",
    "mathtext.bf": "Palatino Linotype:bold",
})

# ---------------- fig 1 ----------------
def fig1():
    w = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                    usecols=["is_av", "p95_abs_jerk", "mean_abs_accel",
                             "frac_aggr", "frac_hard_brake",
                             "mean_vsp", "frac_high_vsp"])
    metrics = [("p95_abs_jerk", "95th percentile |jerk| (m s$^{-3}$)", True),
               ("mean_abs_accel", "mean |acceleration| (m s$^{-2}$)", True),
               ("frac_aggr", "aggressive driving share", False),
               ("frac_hard_brake", "hard-braking share", False),
               ("mean_vsp", "mean VSP (kW t$^{-1}$)", True),
               ("frac_high_vsp", "high-power share", False)]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2))
    for ax, (m, lab, logx) in zip(axes.flat, metrics):
        for key, lab2, c in [(1, "Waymo AV", AVC), (0, "Human vehicles", HC)]:
            v = np.sort(w.loc[w.is_av == key, m].values)
            ax.plot(v, np.arange(1, v.size + 1) / v.size, color=c, lw=1.2, label=lab2)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("ECDF", fontsize=8)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_ecdf.pdf"))
    plt.close(fig)


# ---------------- fig 2 ----------------
def fig2():
    p = pd.read_csv("${DATA_DIR}/results_full/paired_scenarios.csv")
    regs = [(0, 5, "0–5"), (5, 10, "5–10"), (10, 15, "10–15"), (15, 100, ">15")]
    metrics = ["p95_abs_jerk", "frac_aggr", "frac_hard_brake", "frac_high_vsp"]
    labels = ["95th p. |jerk|", "aggressive share", "hard-brake share", "high-power share"]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1), sharey=False)
    for ax, m, lab in zip(axes, metrics, labels):
        xs, lo, hi, mid = [], [], [], []
        for i, (a, b, _) in enumerate(regs):
            sub = p[(p.av_speed >= a) & (p.av_speed < b)]
            if len(sub) < 30:
                continue
            d = (sub[f"av_{m}"] - sub[f"hdv_{m}"]) / sub[f"hdv_{m}"].replace(0, np.nan)
            d = d.dropna()
            xs.append(i); mid.append(d.median())
            ci = np.percentile(
                [np.median(np.random.default_rng(s).choice(d.values, d.size))
                 for s in range(1000)], [2.5, 97.5])
            lo.append(ci[0]); hi.append(ci[1])
        ax.errorbar(xs, mid, yerr=[np.array(mid) - np.array(lo),
                                   np.array(hi) - np.array(mid)],
                    fmt="o", color=AVC, capsize=2, ms=4, lw=1)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([r[2] for r in regs], fontsize=7)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("AV mean speed (m s$^{-1}$)", fontsize=7)
        if m == "p95_abs_jerk":
            ax.set_ylabel("relative paired difference", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_paired.pdf"))
    plt.close(fig)


# ---------------- fig 3 (string stability + composition, 3 panels) ---------
def fig3():
    wp = pd.read_csv("${DATA_DIR}/pairs_full.csv",
                     usecols=["follower_is_av", "mean_speed", "gain", "leader_fluct_std"])
    ng = pd.read_csv("${DATA_DIR}/ngsim_pairs.csv",
                     usecols=["mean_speed", "gain", "leader_fluct_std"])
    wf = wp[wp.mean_speed > 10]
    nf = ng[ng.mean_speed > 10]
    strata = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 3.0)]
    mids = [np.mean(s) for s in strata]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for series, name, c in [
            (wf[wf.follower_is_av == 1], "Waymo AV", AVC),
            (wf[wf.follower_is_av == 0], "Human (perceived)", HC),
            (nf, "Human (NGSIM, independent)", NC)]:
        med, q1, q3 = [], [], []
        for lo, hi in strata:
            s = series[(series.leader_fluct_std >= lo) & (series.leader_fluct_std < hi)]
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
            if len(s) >= 30:
                med[-1] = s.gain.median()
                q1[-1], q3[-1] = s.gain.quantile([.25, .75])
        ax1.plot(mids, med, "o-", color=c, label=name, ms=3.5, lw=1.1)
        ax1.fill_between(mids, q1, q3, color=c, alpha=0.15, lw=0)
    ax1.axhline(1, color="k", lw=0.7, ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("leader disturbance (m s$^{-1}$)", fontsize=7)
    ax1.set_ylabel("gain (median, IQR)", fontsize=7)
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=6)

    for series, name, c in [
            (wf[(wf.follower_is_av == 1) & (wf.leader_fluct_std >= 0.3)], "Waymo AV", AVC),
            (nf[nf.leader_fluct_std >= 0.3], "Human (NGSIM)", NC)]:
        v = np.sort(series.gain.values)
        ax2.plot(v, np.arange(1, v.size + 1) / v.size, color=c, lw=1.1, label=name)
    ax2.axvline(1, color="k", lw=0.7, ls="--")
    ax2.set_xlim(0, 3)
    ax2.set_xlabel("gain, disturbances ≥ 0.3 m s$^{-1}$", fontsize=7)
    ax2.set_ylabel("ECDF", fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6)

    # panel c: composition with measurement equalization
    import os as _os
    nm = pd.read_csv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "p01_pairs.csv"))
    s = nm[(nm.mean_speed > 10) & (nm.l_fluct >= 0.3)]
    groups = [((0, 0), "raw", "human|\nhuman", HC),
              ((0, 1), "raw", "human|AV\n(raw)", "#c39bd3"),
              ((0, 1), "nm", "human|AV\n(equalized)", "#8e44ad"),
              ((1, 0), "raw", "AV|\nhuman", AVC)]
    for i, ((f, l), mode, lab, c) in enumerate(groups):
        col = "g_raw" if mode == "raw" else "g_nm"
        g = s[(s.f_av == f) & (s.l_av == l)][col]
        rng2 = np.random.default_rng(1)
        boots = [np.median(rng2.choice(g.values, g.size)) for _ in range(1000)]
        med = np.median(g)
        lo_, hi_ = np.percentile(boots, [2.5, 97.5])
        ax3.errorbar([i], [med], yerr=[[med - lo_], [hi_ - med]],
                     fmt="o", color=c, capsize=2, ms=4)
    ax3.axhline(1, color="k", lw=0.7, ls="--")
    ax3.set_xticks(range(len(groups)))
    ax3.set_xticklabels([g[2] for g in groups], fontsize=6)
    ax3.tick_params(labelsize=7)
    ax3.set_ylabel("gain (median, 95% CI)", fontsize=7)
    ax3.set_xlabel("follower$|$leader type", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_string.pdf"))
    plt.close(fig)


# ---------------- fig 4 ----------------
def fig4():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "platoon_results.csv"))
    g = df.groupby("n_av").agg(e=("e_pos_kWh_100km", "mean"),
                               sd=("e_pos_kWh_100km", "std")).reset_index()
    g["p"] = 100 * g.n_av / 60
    base = g.e.iloc[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar(g.p, g.e, yerr=g.sd, fmt="o-", color=AVC, capsize=2, ms=4, lw=1.2)
    ax.set_xlabel("AV penetration (%)", fontsize=8)
    ax.set_ylabel("energy (kWh per 100 km)", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(g.p, 100 * (g.e / base - 1), "s--", color=HC, ms=3, lw=1)
    ax2.set_ylabel("relative change (%)", fontsize=8, color=HC)
    ax2.tick_params(axis="y", labelcolor=HC)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sim.pdf"))
    plt.close(fig)


# ---------------- fig 5: heterogeneity ----------------
def fig5():
    tk = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                     usecols=["scenario_id", "is_av"])
    p = pd.read_csv("${DATA_DIR}/results_full/paired_scenarios.csv")
    dens = tk.groupby("scenario_id").size().rename("n_veh")
    p = p.join(dens, on="scenario_id")
    p["dens"] = pd.cut(p.n_veh, [0, 10, 16, 100],
                       labels=["low (<10 veh)", "mid (10–16)", "high (>16)"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6),
                                   gridspec_kw={"width_ratios": [1, 1]})
    for m, lab, c in [("p95_abs_jerk", "95th p. |jerk|", "#c0392b"),
                      ("frac_aggr", "aggressive share", "#e67e22")]:
        xs, mid, lo, hi = [], [], [], []
        for i, cat in enumerate(["low (<10 veh)", "mid (10–16)", "high (>16)"]):
            sub = p[p.dens == cat]
            d = ((sub[f"av_{m}"] - sub[f"hdv_{m}"]) /
                 sub[f"hdv_{m}"].replace(0, np.nan)).dropna()
            xs.append(i); mid.append(d.median())
            ci = np.percentile([np.median(np.random.default_rng(s).choice(d.values, d.size))
                                for s in range(800)], [2.5, 97.5])
            lo.append(ci[0]); hi.append(ci[1])
        ax1.errorbar(xs, mid, yerr=[np.array(mid) - np.array(lo),
                                    np.array(hi) - np.array(mid)],
                     fmt="o--", color=c, capsize=2, ms=4, lw=1.1, label=lab)
    ax1.axhline(0, color="k", lw=0.7, ls=":")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["low (<10)", "mid (10–16)", "high (>16)"], fontsize=7)
    ax1.set_xlabel("surrounding traffic density (vehicles per scene)", fontsize=8)
    ax1.set_ylabel("relative paired difference", fontsize=8)
    ax1.legend(fontsize=7)

    # panel b: follower-leader types with measurement equalization
    import os as _os
    nm_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "p01_pairs.csv")
    nm = pd.read_csv(nm_path)
    s = nm[(nm.mean_speed > 10) & (nm.l_fluct >= 0.3)]
    groups = [((0, 0), "raw", "human|human", HC),
              ((0, 1), "raw", "human|AV\n(raw)", "#c39bd3"),
              ((0, 1), "nm", "human|AV\n(equalized)", "#8e44ad"),
              ((1, 0), "raw", "AV|human", AVC)]
    for i, ((f, l), mode, lab, c) in enumerate(groups):
        col = "g_raw" if mode == "raw" else "g_nm"
        g = s[(s.f_av == f) & (s.l_av == l)][col]
        rng2 = np.random.default_rng(1)
        boots = [np.median(rng2.choice(g.values, g.size)) for _ in range(1000)]
        med = np.median(g)
        lo_, hi_ = np.percentile(boots, [2.5, 97.5])
        ax2.errorbar([i], [med], yerr=[[med - lo_], [hi_ - med]],
                     fmt="o", color=c, capsize=3, ms=5)
        ax2.annotate(f"{g.size//1000}k", (i, med), textcoords="offset points",
                     xytext=(10, -2), fontsize=7, color=c)
    ax2.axhline(1, color="k", lw=0.7, ls=":")
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels([g[2] for g in groups], fontsize=7)
    ax2.set_ylabel("fluctuation gain (median, 95% CI)", fontsize=8)
    ax2.set_xlabel("follower$|$leader type (disturbances ≥ 0.3 m s$^{-1}$)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_heterogeneity.pdf"))
    plt.close(fig)


# ---------------- fig 6: measured energy proxy + physical model ----------------
def fig6():
    tk = pd.read_csv("${DATA_DIR}/tracks_full2.csv",
                     usecols=["is_av", "mean_speed", "mean_vsp",
                              "sum_v", "sum_v3", "sum_va"])
    tk = tk[tk.mean_speed > 1].copy()
    m, g, rho, CdA, Crr = 1800.0, 9.81, 1.2, 0.70, 0.010
    tk["e_whl_kJ"] = (m * tk.sum_va + m * g * Crr * tk.sum_v
                      + 0.5 * rho * CdA * tk.sum_v3) / 1000.0
    tk["vsp_km"] = tk.mean_vsp / tk.mean_speed
    tk["e_km"] = tk.e_whl_kJ / (tk.sum_v / 1000.0)
    tk["regime"] = pd.cut(tk.mean_speed, [1, 5, 10, 15, 100],
                          labels=["0–5", "5–10", "10–15", ">15"])
    regs = ["0–5", "5–10", "10–15", ">15"]

    def med_ci(col):
        out = {}
        for r in regs:
            for key in [0, 1]:
                g_ = tk.loc[(tk.regime == r) & (tk.is_av == key), col]
                rng2 = np.random.default_rng(7)
                boots = [np.median(rng2.choice(g_.values, g_.size))
                         for _ in range(500)]
                out[(r, key)] = (np.median(g_), np.percentile(boots, [2.5, 97.5]))
        return out

    med_v = med_ci("vsp_km")
    med_e = med_ci("e_km")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6))
    x = np.arange(len(regs))
    for key, lab, c, off in [(0, "Human vehicles", HC, -0.18),
                             (1, "Waymo AV", AVC, 0.18)]:
        vals = [med_v[(r, key)][0] for r in regs]
        err = ([med_v[(r, key)][0] - med_v[(r, key)][1][0] for r in regs],
               [med_v[(r, key)][1][1] - med_v[(r, key)][0] for r in regs])
        ax1.bar(x + off, vals, width=0.34, color=c, label=lab,
                yerr=err, capsize=2, error_kw=dict(lw=0.8))
    ax1.set_xticks(x); ax1.set_xticklabels(regs, fontsize=8)
    ax1.set_xlabel("mean speed (m s$^{-1}$)", fontsize=8)
    ax1.set_ylabel("VSP per km (kJ t$^{-1}$ km$^{-1}$)", fontsize=8)
    ax1.legend(fontsize=7)

    w = 0.35
    for i, (col, lab, c) in enumerate([(med_v, "VSP proxy", "#95a5a6"),
                                       (med_e, "wheel-level model", "#2c3e50")]):
        rel = [100 * (col[(r, 1)][0] / col[(r, 0)][0] - 1) for r in regs]
        ax2.bar(x + (i - 0.5) * w, rel, width=w, color=c, label=lab)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(regs, fontsize=8)
    ax2.set_xlabel("mean speed (m s$^{-1}$)", fontsize=8)
    ax2.set_ylabel("AV relative change (%)", fontsize=8)
    ax2.set_ylim(-80, 15)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_energy.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 done")
    fig3(); print("fig3 done")
    fig4(); print("fig4 done")
    fig6(); print("fig6 done")
    ed1(); print("ed1 done")
    ed2(); print("ed2 done")
    ed3(); print("ed3 done")
    ed4(); print("ed4 done")


# ------------- Extended Data figures (from existing CSVs) -------------
def ed1():
    """Estimator-bias calibration (null simulations)."""
    df = pd.read_csv("p02_gain_bias.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for noise, c in [(0.0, "#7f8c8d"), (0.05, "#2980b9"), (0.15, "#8e44ad")]:
        s = df[(df.noise == noise) & (df.L == 150) & (df.g_true == 1.0)]
        ax1.plot(s.sigma_d, s.bias_ratio, "o-", color=c, ms=3.5, lw=1.1,
                 label=f"noise={noise}")
        s2 = df[(df.noise == noise) & (df.sigma_d == 0.6) & (df.g_true == 1.0)]
        ax2.plot(s2.L, s2.bias_ratio, "s-", color=c, ms=3.5, lw=1.1)
    ax1.axhline(1, color="k", lw=0.7, ls="--")
    ax1.axvline(0.3, color="k", lw=0.7, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("true disturbance amplitude (m s$^{-1}$)", fontsize=7)
    ax1.set_ylabel("estimated / true gain", fontsize=7)
    ax1.tick_params(labelsize=7); ax1.legend(fontsize=6)
    ax2.axhline(1, color="k", lw=0.7, ls="--")
    ax2.set_xlabel("window length (steps)", fontsize=7)
    ax2.set_ylabel("estimated / true gain", fontsize=7)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed1_bias.pdf")); plt.close(fig)


def ed2():
    """Controller validation: chain gain decay (all-human vs all-AV)."""
    df = pd.read_csv("platoon_probe.csv")   # columns: config, position, gain
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for cfg, lab, c in [("human", "all human", HC), ("av", "all AV", AVC)]:
        s = df[df.config == cfg].sort_values("position")
        ax.plot(s.position, s.gain, "o-", color=c, ms=3.5, lw=1.1, label=lab)
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_xlabel("position in platoon", fontsize=7)
    ax.set_ylabel("chain fluctuation gain", fontsize=7)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed2_controller.pdf")); plt.close(fig)


def ed3():
    """Controller-parameter sensitivity of simulated savings."""
    df = pd.read_csv("p12_sensitivity.csv")
    base = 41.09
    g = df.groupby(["n_av", "jerk_cap", "t_close", "T_av"]).e.mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for n_av, lab in [(12, "20%"), (30, "50%"), (60, "100%")]:
        sub = g[g.n_av == n_av]
        sav = 100 * (1 - sub.e / base)
        ax.scatter([lab] * len(sav), sav, color="#5d6d7e", s=12, alpha=0.6)
        ax.scatter([lab], [sav.median()], color=AVC, s=30, zorder=3)
    ax.set_xlabel("AV penetration", fontsize=7)
    ax.set_ylabel("energy saving (%)", fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed3_sensitivity.pdf")); plt.close(fig)


def ed4():
    """Cross-validation: freeway style distributions, three sources."""
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
        for ax, m in [(ax1, "p95_abs_jerk"), (ax2, "frac_aggr")]:
            v = np.sort(s[m].values)
            ax.plot(v, np.arange(1, v.size + 1) / v.size, color=c, lw=1.1, label=lab)
    ax1.set_xscale("log")
    ax1.set_xlabel("95th p. $|jerk|$ (m s$^{-3}$)", fontsize=7)
    ax2.set_xlabel("aggressive-driving share", fontsize=7)
    for ax in (ax1, ax2):
        ax.set_ylabel("ECDF", fontsize=7); ax.tick_params(labelsize=7)
    ax1.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ed4_validation.pdf")); plt.close(fig)
