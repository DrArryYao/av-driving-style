"""Expanded Extended Data figures: each ≥6 panels with diverse types."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

OUT = r"${OUT_DIR}"
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
    ax.text(-0.15, 1.08, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

def ecdf(ax, series, color, **kw):
    v = np.sort(series.dropna())
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=1.1, **kw)


HERE = os.path.dirname(os.path.abspath(__file__))


# ============ ED Fig 1: Estimator calibration, 6 panels (2x3) ============
def ed1():
    df = pd.read_csv(os.path.join(HERE, "p02_gain_bias.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: bias vs disturbance
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s = df[(df.noise==noise)&(df.L==150)&(df.g_true==1.0)]
        ax1.plot(s.sigma_d, s.bias_ratio, "o-", color=c, ms=3.5, lw=1.1,
                 label=f"$\\sigma$={noise}")
    ax1.axhline(1,color="k",lw=0.8,ls="--"); ax1.axvline(0.3,color="k",lw=0.8,ls=":")
    ax1.set_xscale("log"); ax1.set_xlabel("disturbance (m s$^{-1}$)")
    ax1.set_ylabel("est./true"); ax1.legend(fontsize=6)
    letter(ax1,"a")

    # b: bias vs window length
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s2 = df[(df.noise==noise)&(df.sigma_d==0.6)&(df.g_true==1.0)]
        ax2.plot(s2.L, s2.bias_ratio, "s-", color=c, ms=3.5, lw=1.1)
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xlabel("window (steps)"); ax2.set_ylabel("est./true")
    letter(ax2,"b")

    # c: bias vs true gain
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s3 = df[(df.noise==noise)&(df.L==150)&(df.sigma_d==0.6)]
        ax3.plot(s3.g_true, s3.bias_ratio, "D-", color=c, ms=4, lw=1.1)
    ax3.axhline(1,color="k",lw=0.8,ls="--")
    ax3.set_xlabel("true gain"); ax3.set_ylabel("est./true")
    ax3.set_ylim(0.6,1.4)
    letter(ax3,"c")

    # d: IQR width vs disturbance
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s4 = df[(df.noise==noise)&(df.L==150)&(df.g_true==1.0)]
        ax4.plot(s4.sigma_d, s4.iqr_hi-s4.iqr_lo, "v-", color=c, ms=3.5, lw=1.1)
    ax4.axvline(0.3,color="k",lw=0.8,ls=":")
    ax4.set_xscale("log"); ax4.set_xlabel("disturbance (m s$^{-1}$)")
    ax4.set_ylabel("IQR width")
    letter(ax4,"d")

    # e: violin plot of estimated/true by noise level (at sigma_d=0.6)
    data_e = []
    labels_e = []
    for noise in [0.0, 0.05, 0.15]:
        s5 = df[(df.noise==noise)&(df.L==150)&(df.sigma_d==0.6)]
        for gt in s5.g_true.unique():
            vals = s5[s5.g_true==gt].bias_ratio.values
            data_e.append(vals)
            labels_e.append(f"$\\sigma$={noise}\nG={gt:.1f}")
    parts = ax5.violinplot(data_e[:6], positions=range(1,min(7,len(data_e)+1)),
                           showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#56B4E9"); pc.set_alpha(0.5)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xticks(range(1,min(7,len(data_e)+1)))
    ax5.set_xticklabels(labels_e[:6], fontsize=5.5)
    ax5.set_ylabel("est./true")
    letter(ax5,"e")

    # f: bias heatmap (noise x disturbance x window)
    hm = np.full((3, 4), np.nan)
    noises = [0.0, 0.05, 0.15]
    for ni, noise in enumerate(noises):
        for di, sd in enumerate([0.05, 0.15, 0.3, 1.0]):
            s6 = df[(df.noise==noise)&(df.sigma_d==sd)&(df.L==150)&(df.g_true==1.0)]
            if len(s6) > 0:
                hm[ni, di] = s6.bias_ratio.iloc[0]
    im = ax6.imshow(hm, cmap="RdYlBu_r", vmin=0.7, vmax=1.3, aspect="auto")
    ax6.set_xticks(range(4)); ax6.set_xticklabels([".05",".15",".3","1.0"], fontsize=6.5)
    ax6.set_yticks(range(3)); ax6.set_yticklabels(["0",".05",".15"], fontsize=6.5)
    ax6.set_xlabel("disturbance (m s$^{-1}$)")
    ax6.set_ylabel("noise $\\sigma$ (m s$^{-1}$)")
    for ni in range(3):
        for di in range(4):
            if not np.isnan(hm[ni,di]):
                ax6.text(di,ni,f"{hm[ni,di]:.2f}",ha="center",va="center",fontsize=6.5)
    cb = fig.colorbar(im, ax=ax6, fraction=0.045)
    cb.set_label("est./true", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed1_bias.pdf")); plt.close(fig)


# ============ ED Fig 2: Controller validation, 6 panels (2x3) ============
def ed2():
    df = pd.read_csv(os.path.join(HERE, "platoon_probe.csv"))
    traces = pd.read_csv(os.path.join(HERE, "sim_speed_traces.csv"))
    energy = pd.read_csv(os.path.join(HERE, "sim_per_vehicle_energy.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: chain gain decay
    for cfg,lab,c in [("human","all human",HC),("av","all AV",AVC)]:
        s = df[df.config==cfg].sort_values("position")
        ax1.plot(s.position, s.gain, "o-", color=c, ms=3.5, lw=1.3, label=lab)
    ax1.axhline(1,color="k",lw=0.8,ls="--")
    ax1.set_xlabel("position"); ax1.set_ylabel("chain gain")
    ax1.legend()
    letter(ax1,"a")

    # b: leader speed profile
    lead = traces[(traces.pen==0)&(traces.veh==5)]
    ax2.plot(lead.t[::10], lead.v[::10], color="#333", lw=0.8)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("speed (m s$^{-1}$)")
    ax2.set_title("input wave", fontsize=7)
    letter(ax2,"b")

    # c: vehicle 35 response
    v35_0 = traces[(traces.pen==0)&(traces.veh==35)]
    v35_50 = traces[(traces.pen==50)&(traces.veh==35)]
    ax3.plot(v35_0.t[::5], v35_0.v[::5], color=HC, lw=0.8, alpha=0.8, label="human")
    ax3.plot(v35_50.t[::5], v35_50.v[::5], color=AVC, lw=0.8, alpha=0.8, label="AV")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("speed (m s$^{-1}$)")
    ax3.legend(fontsize=6)
    letter(ax3,"c")

    # d: vehicle 55 response (further downstream)
    v55_0 = traces[(traces.pen==0)&(traces.veh==55)]
    v55_50 = traces[(traces.pen==50)&(traces.veh==55)]
    ax4.plot(v55_0.t[::5], v55_0.v[::5], color=HC, lw=0.8, alpha=0.8, label="human")
    ax4.plot(v55_50.t[::5], v55_50.v[::5], color=AVC, lw=0.8, alpha=0.8, label="AV")
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("speed (m s$^{-1}$)")
    ax4.legend(fontsize=6)
    letter(ax4,"d")

    # e: speed std of each vehicle position (human vs AV platoon)
    for pen,lab,c in [(0,"0% AV",HC),(50,"50% AV",AVC)]:
        stds = []
        for vi in [5,15,25,35,45,55]:
            sub = traces[(traces.pen==pen)&(traces.veh==vi)]
            stds.append(sub.v.std())
        ax5.plot([5,15,25,35,45,55], stds, "o-", color=c, ms=4, lw=1.3, label=lab)
    ax5.set_xlabel("vehicle position"); ax5.set_ylabel("speed s.d. (m s$^{-1}$)")
    ax5.legend()
    letter(ax5,"e")

    # f: per-vehicle energy violin
    e0 = energy[energy.pen==0]
    e50 = energy[energy.pen==50]
    parts = ax6.violinplot([e0.e_kwh.clip(0,60), e50.e_kwh.clip(0,60)],
                           positions=[1,2], showmedians=True, widths=0.7)
    for pc,c in zip(parts["bodies"],[HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax6.set_xticks([1,2]); ax6.set_xticklabels(["0%","50%"],fontsize=7)
    ax6.set_ylabel("kWh/100km")
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed2_controller.pdf")); plt.close(fig)


# ============ ED Fig 3: Sensitivity, 6 panels (2x3) ============
def ed3():
    df = pd.read_csv(os.path.join(HERE, "p12_sensitivity.csv"))
    base = 41.09
    df["sav"] = 100 * (1 - df.e / base)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: savings vs penetration
    for n_av,lab in [(12,"20%"),(30,"50%"),(60,"100%")]:
        sub = df[df.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).sav.mean()
        ax1.scatter([lab]*len(sub), sub, color="#999", s=12, alpha=0.6)
        ax1.scatter([lab],[sub.median()],color=AVC,s=30,zorder=3)
    ax1.set_xlabel("penetration"); ax1.set_ylabel("saving (%)")
    letter(ax1,"a")

    # b: vs jerk cap at 50%
    sub50 = df[df.n_av==30]
    for jc in [1.0,1.5,2.0]:
        s = sub50[sub50.jerk_cap==jc].sav
        ax2.scatter([jc]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax2.scatter([jc],[s.median()],color=AVC,s=35,zorder=3)
    ax2.set_xlabel("jerk cap (m s$^{-3}$)"); ax2.set_ylabel("saving at 50% (%)")
    letter(ax2,"b")

    # c: vs anticipation
    for tc in [0.0,0.3,0.55]:
        s = sub50[sub50.t_close==tc].sav
        ax3.scatter([tc]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax3.scatter([tc],[s.median()],color=AVC,s=35,zorder=3)
    ax3.set_xlabel("anticipation gain"); ax3.set_ylabel("saving at 50% (%)")
    letter(ax3,"c")

    # d: vs headway
    for T in [1.1,1.4,1.7]:
        s = sub50[sub50.T_av==T].sav
        ax4.scatter([T]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax4.scatter([T],[s.median()],color=AVC,s=35,zorder=3)
    ax4.set_xlabel("headway T (s)"); ax4.set_ylabel("saving at 50% (%)")
    letter(ax4,"d")

    # e: interaction heatmap (jerk_cap x T_av at t_close=0.3, 50% pen)
    hm = np.full((3,3), np.nan)
    jcs = [1.0,1.5,2.0]; Ts = [1.1,1.4,1.7]
    for ji,jc in enumerate(jcs):
        for ti,T in enumerate(Ts):
            s = sub50[(sub50.jerk_cap==jc)&(sub50.T_av==T)&(sub50.t_close==0.3)].sav
            if len(s)>0: hm[ji,ti] = s.mean()
    im = ax5.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax5.set_xticks(range(3)); ax5.set_xticklabels(Ts, fontsize=7)
    ax5.set_yticks(range(3)); ax5.set_yticklabels(jcs, fontsize=7)
    ax5.set_xlabel("headway T (s)"); ax5.set_ylabel("jerk cap")
    for ji in range(3):
        for ti in range(3):
            if not np.isnan(hm[ji,ti]):
                ax5.text(ti,ji,f"{hm[ji,ti]:.1f}",ha="center",va="center",fontsize=7)
    cb = fig.colorbar(im, ax=ax5, fraction=0.045)
    cb.set_label("saving (%)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    letter(ax5,"e")

    # f: savings distribution (all 27 configs × 3 penetrations)
    data_f = []; labels_f = []
    for n_av,lab,c in [(12,"20%",NC),(30,"50%","#E69F00"),(60,"100%",AVC)]:
        sub = df[df.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).sav.mean()
        data_f.append(sub.values); labels_f.append(lab)
    parts = ax6.violinplot(data_f, positions=[1,2,3], showmedians=True, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor("#56B4E9"); pc.set_alpha(0.5)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax6.set_xticks([1,2,3]); ax6.set_xticklabels(labels_f, fontsize=7)
    ax6.set_ylabel("saving (%)"); ax6.set_xlabel("penetration")
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed3_sensitivity.pdf")); plt.close(fig)


# ============ ED Fig 4: Perception validation, 6 panels (2x3) ============
def ed4():
    tk = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                     usecols=["is_av","mean_speed","p95_abs_jerk","frac_aggr",
                              "mean_abs_accel"])
    ng = pd.read_csv("${DATA_DIR}/ngsim_tracks.csv",
                     usecols=["mean_speed","p95_abs_jerk","frac_aggr"])
    w_h = tk[(tk.is_av==0)&(tk.mean_speed>10)]
    w_a = tk[(tk.is_av==1)&(tk.mean_speed>10)]
    ng_h = ng[ng.mean_speed>10]

    colors = [(w_h,HC,"Human (AV-perc.)"),(ng_h,NC,"Human (NGSIM)"),
              (w_a,AVC,"Waymo AV")]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.4))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: jerk ECDF
    for s,c,_ in colors:
        ecdf(ax1, s.p95_abs_jerk, c)
    ax1.set_xscale("log"); ax1.set_xlabel("95th p.$|jerk|$ (m s$^{-3}$)")
    ax1.set_ylabel("ECDF")
    h = [Line2D([0],[0],color=c,lw=1.2,label=l) for _,c,l in colors]
    ax1.legend(handles=h, fontsize=6)
    letter(ax1,"a")

    # b: aggressive share ECDF
    for s,c,_ in colors:
        ecdf(ax2, s.frac_aggr, c)
    ax2.set_xlabel("aggressive share"); ax2.set_ylabel("ECDF")
    letter(ax2,"b")

    # c: median jerk by speed bin
    bins = [(10,15),(15,20),(20,25),(25,35)]
    bl = ["10-15","15-20","20-25","25-35"]
    x = np.arange(len(bins))
    for (s,c,_),off in zip(colors,[-0.22,0,0.22]):
        vals = []
        for lo,hi in bins:
            sub = s[(s.mean_speed>=lo)&(s.mean_speed<hi)]
            vals.append(sub.p95_abs_jerk.median() if len(sub)>10 else np.nan)
        ax3.bar(x+off, vals, width=0.25, color=c)
    ax3.set_xticks(x); ax3.set_xticklabels(bl, fontsize=7)
    ax3.set_xlabel("speed (m s$^{-1}$)"); ax3.set_ylabel("median jerk")
    letter(ax3,"c")

    # d: mean |accel| ECDF (only WOMD sources, NGSIM lacks this col)
    for s,c,_ in colors:
        if "mean_abs_accel" in s.columns:
            ecdf(ax4, s.mean_abs_accel.clip(0,4), c)
    ax4.set_xlabel("mean $|a|$ (m s$^{-2}$)"); ax4.set_ylabel("ECDF")
    letter(ax4,"d")

    # e: scatter — mean speed vs jerk (subsampled)
    np.random.seed(0)
    for s,c,lab in colors:
        sub = s.sample(min(3000, len(s)), random_state=0)
        ax5.scatter(sub.mean_speed, sub.p95_abs_jerk.clip(0,50),
                   s=1.5, alpha=0.2, color=c, rasterized=True)
    ax5.set_xlabel("mean speed (m s$^{-1}$)"); ax5.set_ylabel("95th p. jerk")
    ax5.set_ylim(0,50)
    letter(ax5,"e")

    # f: summary bar — key metric ratios
    metrics_f = [("p95_abs_jerk","jerk"),("frac_aggr","aggr.")]
    ypos = np.arange(len(metrics_f))
    for i,(m,lab) in enumerate(metrics_f):
        vals = [s[m].median() for s,_,_ in colors]
        norm = [v/vals[0] if vals[0]>0 else 1 for v in vals]
        for j,(v,c) in enumerate(zip(norm,[HC,NC,AVC])):
            ax6.barh(i*3+j, v, color=c, height=0.7)
            ax6.text(v+0.02, i*3+j, f"{vals[j]:.2f}", va="center", fontsize=6)
    ax6.set_yticks([0,1,2,3,4,5])
    ax6.set_yticklabels(["jerk:P","jerk:N","jerk:AV","agr:P","agr:N","agr:AV"],
                         fontsize=6)
    ax6.axvline(1,color=AVC,lw=0.8,ls="--")
    ax6.set_xlabel("normalized to AV-perceived")
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed4_validation.pdf")); plt.close(fig)


if __name__ == "__main__":
    ed1(); print("ED Fig 1 done (6: line+line+line+line+violin+heatmap)")
    ed2(); print("ED Fig 2 done (6: line+line+line+line+line+violin)")
    ed3(); print("ED Fig 3 done (6: scatter+scatter+scatter+scatter+heatmap+violin)")
    ed4(); print("ED Fig 4 done (6: ECDF+ECDF+bar+ECDF+scatter+bar)")
