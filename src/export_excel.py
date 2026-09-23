"""Export all figure source data to a single Excel workbook.

Each figure gets one or more sheets with the exact data used for plotting.
Requires: openpyxl (pip install openpyxl)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\Arry\Desktop\NC\paper"
XLSX = os.path.join(OUT, "figure_source_data.xlsx")
AVC, HC, NC = "#D55E00", "#0072B2", "#009E73"


def load_tracks():
    return pd.read_csv("E:/av_style_data/tracks_full2.csv",
                       usecols=["is_av", "mean_speed", "p95_abs_jerk",
                                "frac_aggr", "frac_hard_brake", "mean_vsp",
                                "frac_high_vsp", "mean_abs_accel", "std_speed",
                                "sum_v", "sum_v3", "sum_va"])


def load_pairs():
    return pd.read_csv("E:/av_style_data/pairs_full.csv",
                       usecols=["follower_is_av", "mean_speed", "gain",
                                "leader_fluct_std"])


def load_paired():
    return pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")


def load_ngsim_tracks():
    return pd.read_csv("E:/av_style_data/ngsim_tracks.csv",
                       usecols=["mean_speed", "p95_abs_jerk", "frac_aggr"])


def load_ngsim_pairs():
    return pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                       usecols=["mean_speed", "gain", "leader_fluct_std"])


def med_ci(series, n=500):
    """Bootstrap median and 95% CI."""
    rng = np.random.default_rng(7)
    med = np.median(series)
    boots = [np.median(rng.choice(series.values, series.size)) for _ in range(n)]
    return med, np.percentile(boots, [2.5, 97.5])


def main():
    tk = load_tracks()
    tk = tk[tk.mean_speed > 1].copy()
    tk["vsp_km"] = tk.mean_vsp / tk.mean_speed
    m, g_, rho, CdA, Crr = 1800.0, 9.81, 1.2, 0.70, 0.010
    tk["e_km"] = ((m * tk.sum_va + m * g_ * Crr * tk.sum_v
                   + 0.5 * rho * CdA * tk.sum_v3) / 1000.0
                  / (tk.sum_v / 1000.0))
    tk["e_km2"] = ((2100.0 * tk.sum_va + 2100.0 * g_ * Crr * tk.sum_v
                    + 0.5 * rho * 0.95 * tk.sum_v3) / 1000.0
                   / (tk.sum_v / 1000.0))
    tk["regime"] = pd.cut(tk.mean_speed, [1, 5, 10, 15, 100],
                          labels=["0–5", "5–10", "10–15", ">15"])
    regs = ["0–5", "5–10", "10–15", ">15"]
    p = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    p["regime"] = pd.cut(p.av_speed, [0, 5, 10, 15, 100],
                         labels=["0–5", "5–10", "10–15", ">15"])
    wp = load_pairs()
    ng_p = load_ngsim_pairs()
    ng_t = load_ngsim_tracks()

    writer = pd.ExcelWriter(XLSX, engine="openpyxl")

    # ====== Fig 1 sheets ======
    # Fig1a: violin (subsample)
    np.random.seed(0)
    sub_h = tk[tk.is_av == 0].sample(min(5000, len(tk[tk.is_av == 0])), random_state=0)
    sub_a = tk[tk.is_av == 1].sample(min(5000, len(tk[tk.is_av == 1])), random_state=0)
    pd.DataFrame({
        "Human_jerk_p95": sub_h.p95_abs_jerk.clip(0, 50).values[:5000],
        "AV_jerk_p95": sub_a.p95_abs_jerk.clip(0, 50).values[:5000],
    }).to_excel(writer, sheet_name="Fig1a_violin", index=False)

    # Fig1b: dumbbell
    rows = []
    for i, r in enumerate(regs):
        sub = p[p.regime == r]
        if len(sub) < 30:
            continue
        dh = sub.hdv_p95_abs_jerk.median()
        da = sub.av_p95_abs_jerk.median()
        rows.append(dict(regime=r, human_median=dh, av_median=da,
                         reduction_pct=round(100 * (1 - da / dh))))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig1b_dumbbell", index=False)

    # Fig1c: heatmap
    metrics = ["p95_abs_jerk", "frac_aggr", "frac_hard_brake",
               "frac_high_vsp", "mean_abs_accel"]
    mlabels = ["|jerk| p95", "aggr. share", "hard brake", "high VSP", "mean |a|"]
    hm_rows = []
    for mi, m in enumerate(metrics):
        row = {"metric": mlabels[mi]}
        for ri, r in enumerate(regs):
            h = tk[(tk.regime == r) & (tk.is_av == 0)][m].median()
            a = tk[(tk.regime == r) & (tk.is_av == 1)][m].median()
            row[r] = round(100 * (a / h - 1)) if h > 0 else None
        hm_rows.append(row)
    pd.DataFrame(hm_rows).to_excel(writer, sheet_name="Fig1c_heatmap", index=False)

    # Fig1d: bar
    rows = []
    for r in regs:
        rows.append(dict(
            regime=r,
            human_share_pct=100 * tk[(tk.regime == r) & (tk.is_av == 0)].frac_aggr.median(),
            av_share_pct=100 * tk[(tk.regime == r) & (tk.is_av == 1)].frac_aggr.median()))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig1d_bar", index=False)

    # Fig1e: ECDF jerk
    for group, sub in [("Human", sub_h), ("AV", sub_a)]:
        v = np.sort(sub.p95_abs_jerk.dropna())
        pd.DataFrame({"value": v, "ECDF": np.arange(1, v.size + 1) / v.size}
                     ).to_excel(writer, sheet_name=f"Fig1e_ECDF_{group}", index=False)

    # Fig1f: ECDF VSP
    for group, sub in [("Human", sub_h), ("AV", sub_a)]:
        v = np.sort(sub.mean_vsp.dropna())
        pd.DataFrame({"value": v, "ECDF": np.arange(1, v.size + 1) / v.size}
                     ).to_excel(writer, sheet_name=f"Fig1f_ECDF_VSP_{group}", index=False)

    # Fig1g: forest
    rows = []
    for m, lab in [("p95_abs_jerk", "jerk"), ("frac_aggr", "aggr."),
                   ("frac_hard_brake", "brake"), ("frac_high_vsp", "VSP")]:
        d = (p[f"av_{m}"] - p[f"hdv_{m}"]) / p[f"hdv_{m}"].replace(0, np.nan)
        d = d.dropna()
        med = d.median()
        ci = np.percentile([np.median(np.random.default_rng(s).choice(d.values, d.size))
                            for s in range(500)], [2.5, 97.5])
        rows.append(dict(metric=lab, median=round(med, 3),
                         CI_lo=round(ci[0], 3), CI_hi=round(ci[1], 3)))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig1g_forest", index=False)

    # ====== Fig 2 sheets ======
    # Fig2a: gain vs disturbance
    wp = load_pairs()
    ng_p = load_ngsim_pairs()
    wf = wp[wp.mean_speed > 10]
    nf = ng_p[ng_p.mean_speed > 10]
    strata = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 3.0)]
    rows = []
    for series, name in [(wf[wf.follower_is_av == 1], "L4 AV"),
                         (wf[wf.follower_is_av == 0], "Human (perceived)"),
                         (nf, "Human (NGSIM)")]:
        for lo, hi in strata:
            s = series[(series.leader_fluct_std >= lo) & (series.leader_fluct_std < hi)]
            if len(s) >= 10:
                rows.append(dict(source=name,
                                 disturbance_lo=lo, disturbance_hi=hi,
                                 gain_median=round(s.gain.median(), 3),
                                 gain_IQR_lo=round(s.gain.quantile(.25), 3),
                                 gain_IQR_hi=round(s.gain.quantile(.75), 3),
                                 n=len(s)))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig2a_gain_dist", index=False)

    # Fig2b: violin
    for name, s in [("L4_AV", wf[(wf.follower_is_av == 1) & (wf.leader_fluct_std >= 0.3)].gain),
                    ("Human_perc", wf[(wf.follower_is_av == 0) & (wf.leader_fluct_std >= 0.3)].gain.sample(3000, random_state=0)),
                    ("NGSIM", nf[nf.leader_fluct_std >= 0.3].gain)]:
        pd.DataFrame({"gain": s.clip(0, 3)}).to_excel(
            writer, sheet_name=f"Fig2b_violin_{name}", index=False)

    # Fig2c: dumbbell composition
    nm = pd.read_csv(os.path.join(HERE, "p01_pairs.csv"))
    s = nm[(nm.mean_speed > 10) & (nm.l_fluct >= 0.3)]
    groups = [((0, 0), "g_raw", "H|H"), ((0, 1), "g_raw", "H|AV raw"),
              ((0, 1), "g_nm", "H|AV equalized"), ((1, 0), "g_raw", "AV|H")]
    rows = []
    for (f, l), mode, lab in groups:
        g = s[(s.f_av == f) & (s.l_av == l)][mode]
        rows.append(dict(type=lab, gain_median=round(np.median(g), 3), n=len(g)))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig2c_composition", index=False)

    # Fig2d: heatmap
    wf2 = wp[wp.mean_speed > 5]
    xe = [0.05, 0.15, 0.3, 0.6, 3.0]
    ye = [5, 10, 15, 20, 35]
    rows = []
    for xi in range(4):
        for yi in range(4):
            msk = ((wf2.leader_fluct_std >= xe[xi]) & (wf2.leader_fluct_std < xe[xi + 1]) &
                   (wf2.mean_speed >= ye[yi]) & (wf2.mean_speed < ye[yi + 1]))
            a = wf2[msk & (wf2.follower_is_av == 1)].gain
            h = wf2[msk & (wf2.follower_is_av == 0)].gain
            if len(a) > 50 and len(h) > 50:
                rows.append(dict(disturbance=f"{xe[xi]:.2f}-{xe[xi+1]:.2f}",
                                 speed=f"{ye[yi]}-{ye[yi+1]}",
                                 diff_AV_minus_human=round(a.median() - h.median(), 3)))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig2d_heatmap", index=False)

    # Fig2g: ECDF by source
    for name, s in [("L4_AV", wf[(wf.follower_is_av == 1) & (wf.leader_fluct_std >= 0.3)].gain),
                    ("NGSIM_human", nf[nf.leader_fluct_std >= 0.3].gain)]:
        v = np.sort(s.clip(0, 3))
        pd.DataFrame({"gain": v, "ECDF": np.arange(1, v.size + 1) / v.size}
                     ).to_excel(writer, sheet_name=f"Fig2g_ECDF_{name}", index=False)

    # Fig2h: automation spectrum bar
    rows = []
    for lab, s in [("Human (NGSIM)", nf[nf.leader_fluct_std >= 0.3].gain),
                   ("Comm. ACC", None),  # filled below from CSV
                   ("L4 AV", wf[(wf.follower_is_av == 1) & (wf.leader_fluct_std >= 0.3)].gain)]:
        if s is not None:
            rows.append(dict(system=lab, median_gain=round(s.median(), 3),
                             unstable_frac=round((s > 1).mean(), 3)))
    jrc = pd.read_csv("E:/av_style_data/jrc_acc/jrc_acc_gains.csv")
    rows.insert(1, dict(system="Comm. ACC", median_gain=round(jrc.gain.median(), 3),
                        unstable_frac=round((jrc.gain > 1).mean(), 3)))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig2h_spectrum", index=False)

    # Fig2i: ACC gain vs speed
    jrc[["mean_speed", "gain"]].to_excel(writer, sheet_name="Fig2i_ACC_speed", index=False)

    # ====== Fig 3 (simulation) sheets ======
    sim = pd.read_csv(os.path.join(HERE, "platoon_results.csv"))
    sim_g = sim.groupby("n_av").agg(e=("e_pos_kWh_100km", "mean"),
                                    sd=("e_pos_kWh_100km", "std"),
                                    vmean=("throughput_vmean", "mean"),
                                    vstd=("fleet_v_std", "mean"),
                                    stops=("stop_frac", "mean")).reset_index()
    sim_g["p"] = 100 * sim_g.n_av / 60
    sim_g["saving_pct"] = 100 * (1 - sim_g.e / sim_g.e.iloc[0])
    sim_g[["p", "saving_pct", "vmean", "vstd", "stops"]].to_excel(
        writer, sheet_name="Fig3a_savings", index=False)

    traces = pd.read_csv(os.path.join(HERE, "sim_speed_traces.csv"))
    for pen in [0, 50]:
        tr = traces[traces.pen == pen]
        for vi in [5, 15, 25, 35, 45, 55]:
            sub = tr[tr.veh == vi]
            if len(sub) > 0:
                pd.DataFrame({"t": sub.t.values[::5], "v": sub.v.values[::5]}
                             ).to_excel(writer, sheet_name=f"Fig3{'bc'}_trace_p{pen}_v{vi}", index=False)

    # ====== Fig 4 (energy) sheets ======
    def med_ci(col):
        out = {}
        for r in regs:
            for key in [0, 1]:
                g2 = tk.loc[(tk.regime == r) & (tk.is_av == key), col]
                rng2 = np.random.default_rng(7)
                boots = [np.median(rng2.choice(g2.values, g2.size)) for _ in range(500)]
                out[(r, key)] = (np.median(g2), np.percentile(boots, [2.5, 97.5]))
        return out
    med_v = med_ci("vsp_km")
    med_e = med_ci("e_km")
    med_e2 = {}
    for r in regs:
        for key in [0, 1]:
            g2 = tk.loc[(tk.regime == r) & (tk.is_av == key), "e_km2"]
            med_e2[(r, key)] = np.median(g2)

    rows = []
    for r in regs:
        rows.append(dict(regime=r,
                         vsp_human=round(med_v[(r, 0)][0], 2),
                         vsp_AV=round(med_v[(r, 1)][0], 2),
                         vsp_change_pct=round(100 * (med_v[(r, 1)][0] / med_v[(r, 0)][0] - 1)),
                         wheel_human=round(med_e[(r, 0)][0], 2),
                         wheel_AV=round(med_e[(r, 1)][0], 2),
                         wheel_change_pct=round(100 * (med_e[(r, 1)][0] / med_e[(r, 0)][0] - 1)),
                         wheel_sedan_saving=round(100 * (med_e[(r, 1)][0] / med_e[(r, 0)][0] - 1)),
                         wheel_minivan_saving=round(100 * (med_e2[(r, 1)] / med_e2[(r, 0)] - 1))))
    pd.DataFrame(rows).to_excel(writer, sheet_name="Fig4_energy", index=False)

    # ====== Fig 5 (robustness) sheets ======
    robust = pd.read_csv(os.path.join(HERE, "p12_sensitivity.csv"))
    robust["sav"] = 100 * (1 - robust.e / 41.09)
    robust_g = robust.groupby("n_av").sav.agg(["median", "min", "max", "std"]).reset_index()
    robust_g["penetration"] = 100 * robust_g.n_av / 60
    robust_g[["penetration", "median", "min", "max", "std"]].to_excel(
        writer, sheet_name="Fig5_sensitivity", index=False)

    # Fig5d: bias curve
    bias = pd.read_csv(os.path.join(HERE, "p02_gain_bias.csv"))
    s = bias[(bias.noise == 0.15) & (bias.L == 150) & (bias.g_true == 1.0)].sort_values("sigma_d")
    s[["sigma_d", "bias_ratio"]].to_excel(writer, sheet_name="Fig5d_bias", index=False)

    # ====== ED data ======
    # ED1: bias calibration (full)
    bias_all = pd.read_csv(os.path.join(HERE, "p02_gain_bias.csv"))
    bias_all.to_excel(writer, sheet_name="ED1_bias_full", index=False)

    # ED2: chain gains
    chain = pd.read_csv(os.path.join(HERE, "platoon_probe.csv"))
    chain.to_excel(writer, sheet_name="ED2_chain_gains", index=False)

    # ED4: perception validation — use jerk series from each source
    tk_full = tk  # already loaded with is_av, mean_speed, p95_abs_jerk
    for name, s in [("Human_perc", tk_full[(tk_full.is_av == 0) & (tk_full.mean_speed > 10)].p95_abs_jerk),
                    ("AV", tk_full[(tk_full.is_av == 1) & (tk_full.mean_speed > 10)].p95_abs_jerk)]:
        v = np.sort(s.dropna())
        pd.DataFrame({"value": v, "ECDF": np.arange(1, v.size + 1) / v.size}
                     ).to_excel(writer, sheet_name=f"ED4_ECDF_{name}", index=False)
    ng_t_jerk = load_ngsim_tracks()
    v = np.sort(ng_t_jerk[ng_t_jerk.mean_speed > 10].p95_abs_jerk.dropna())
    pd.DataFrame({"value": v, "ECDF": np.arange(1, v.size + 1) / v.size}
                 ).to_excel(writer, sheet_name="ED4_ECDF_NGSIM", index=False)

    writer.close()
    print(f"Excel saved: {XLSX}")
    print(f"Sheets: {len(pd.ExcelFile(XLSX).sheet_names)}")
    for s in pd.ExcelFile(XLSX).sheet_names:
        print(f"  {s}")


if __name__ == "__main__":
    main()
