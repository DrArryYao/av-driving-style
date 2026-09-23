"""Pilot paired analysis: AV vs in-scene human-driven vehicles (HDV).

Design: within each scenario the AV shares road, weather and traffic state
with the HDVs it drives among, so scenario-level pairing removes those
confounders. For each scenario we compare the AV's metrics with the mean of
its HDVs and report paired differences with bootstrap CIs + Wilcoxon tests,
overall and stratified by scenario speed regime.

Usage: python analyze_pilot.py <tracks.csv> <outdir>
"""
import sys
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    "p95_abs_jerk", "mean_abs_accel", "frac_aggr", "frac_hard_brake",
    "mean_vsp", "frac_high_vsp", "std_speed",
]
BOOT_N = 3000
RNG = np.random.default_rng(42)


def paired_table(df):
    """Scenario-level paired comparison (AV vs mean of HDVs in same scenario)."""
    rows = []
    av = df[df.is_av == 1].groupby("scenario_id")
    hdv = df[df.is_av == 0].groupby("scenario_id")
    common = sorted(set(av.groups) & set(hdv.groups))
    for sid in common:
        a = av.get_group(sid)
        h = hdv.get_group(sid)
        # speed regime of the scenario (AV mean speed as regime proxy)
        regime = a["mean_speed"].mean()
        row = {"scenario_id": sid, "av_speed": regime, "n_hdv": len(h)}
        for m in METRICS:
            row[f"av_{m}"] = a[m].mean()
            row[f"hdv_{m}"] = h[m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(p, metric):
    diff = p[f"av_{metric}"] - p[f"hdv_{metric}"]
    boot = np.array([
        RNG.choice(diff.values, diff.size, replace=True).mean() for _ in range(BOOT_N)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        pval = wilcoxon(diff).pvalue
    except ValueError:
        pval = np.nan
    return {
        "metric": metric,
        "av_mean": p[f"av_{metric}"].mean(),
        "hdv_mean": p[f"hdv_{metric}"].mean(),
        "paired_diff": diff.mean(),
        "ci_lo": lo, "ci_hi": hi,
        "rel_diff_%": 100 * diff.mean() / max(abs(p[f"hdv_{metric}"].mean()), 1e-9),
        "wilcoxon_p": pval,
        "n_scenarios": len(diff),
    }


def main():
    csv_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    p = paired_table(df)
    p.to_csv(os.path.join(outdir, "paired_scenarios.csv"), index=False)

    summary = pd.DataFrame([summarize(p, m) for m in METRICS])
    summary.to_csv(os.path.join(outdir, "summary_overall.csv"), index=False)
    print("=== OVERALL (paired by scenario) ===")
    print(summary.to_string(index=False))

    # stratify by scenario speed regime
    bins = [0, 5, 10, 15, 100]
    labels = ["0-5 m/s (stop-go)", "5-10 m/s (urban)", "10-15 m/s (arterial)", ">15 m/s (freeway)"]
    p["regime"] = pd.cut(p["av_speed"], bins=bins, labels=labels)
    for lab in labels:
        sub = p[p["regime"] == lab]
        if len(sub) < 30:
            continue
        s = pd.DataFrame([summarize(sub, m) for m in METRICS])
        safe = lab.split()[0].replace(">", "gt").replace("<", "lt")
        s.to_csv(os.path.join(outdir, f"summary_{safe}.csv"), index=False)
        print(f"\n=== REGIME {lab}  (n={len(sub)}) ===")
        print(s.to_string(index=False))

    # --- figure 1: ECDFs of key metrics AV vs HDV (track level) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_metrics = ["p95_abs_jerk", "mean_abs_accel", "frac_aggr",
                    "frac_hard_brake", "mean_vsp", "frac_high_vsp"]
    for ax, m in zip(axes.flat, plot_metrics):
        for key, lab, c in [(1, "Waymo AV", "tab:red"), (0, "Human vehicles", "tab:blue")]:
            v = df[df.is_av == key][m]
            xs = np.sort(v)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, label=f"{lab} (n={xs.size})", color=c, lw=1.5)
        ax.set_xlabel(m); ax.set_ylabel("ECDF"); ax.legend(fontsize=8)
    fig.suptitle("Driving style: Waymo AV vs surrounding human vehicles (track level)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_ecdf.png"), dpi=150)

    # --- figure 2: paired differences with CIs by regime ---
    fig, ax = plt.subplots(figsize=(9, 5))
    regs = [l for l in labels if (p["regime"] == l).sum() >= 30]
    ys = np.arange(len(regs))
    for i, m in [(0, "p95_abs_jerk"), (1, "frac_aggr")]:
        xs, errs = [], []
        for lab in regs:
            sub = p[p["regime"] == lab]
            d = (sub[f"av_{m}"] - sub[f"hdv_{m}"]).mean()
            se = (sub[f"av_{m}"] - sub[f"hdv_{m}"]).std() / np.sqrt(len(sub))
            xs.append(d); errs.append(1.96 * se)
        ax.errorbar(xs, ys + (0.12 if i else -0.12), xerr=errs, fmt="o",
                    label=m, capsize=3)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(regs)
    ax.set_xlabel("Paired difference  AV - HDV (negative = AV smoother)")
    ax.legend(); ax.set_title("In-scenario paired differences by speed regime")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_paired.png"), dpi=150)
    print(f"\nFigures and CSVs written to {outdir}")


if __name__ == "__main__":
    main()
