"""Redesigned Fig 3 (simulation, 6 panels) and ED Figs 1-4 (more panels)."""
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


# ================= Fig 3: Simulation, 6 panels (2x3) =================
def fig4():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "platoon_results.csv"))
    g = df.groupby("n_av").agg(
        e=("e_pos_kWh_100km","mean"), sd=("e_pos_kWh_100km","std"),
        vmean=("throughput_vmean","mean"), vstd=("fleet_v_std","mean"),
        stops=("stop_frac","mean")).reset_index()
    g["p"] = 100*g.n_av/60; base = g.e.iloc[0]
    sens = pd.read_csv(os.path.join(here,"p12_sensitivity.csv")); b = 41.09
    traces = pd.read_csv(os.path.join(here,"sim_speed_traces.csv"))
    energy = pd.read_csv(os.path.join(here,"sim_per_vehicle_energy.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: energy savings
    for n_av in [12,30,60]:
        sub = sens[sens.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).e.mean()
        sav = 100*(1-sub/b)
        ax1.scatter([100*n_av/60]*len(sav), sav, color="#999", s=10, alpha=0.5, zorder=2)
    ax1.errorbar(g.p, 100*(1-g.e/base), yerr=100*g.sd/base,
                 fmt="o-",color=AVC,capsize=2,ms=4,lw=1.3,zorder=3)
    ax1.set_xlabel("AV penetration (%)"); ax1.set_ylabel("energy saving (%)")
    letter(ax1,"a")

    # b: capacity + stops (dual axis)
    ax2.plot(g.p,g.vmean,"o-",color=HC,ms=4)
    ax2.set_xlabel("AV penetration (%)"); ax2.set_ylabel("mean speed (m s$^{-1}$)",color=HC)
    ax2.tick_params(axis="y",labelcolor=HC)
    ax2b = ax2.twinx()
    ax2b.plot(g.p,100*g.stops,"s--",color=AVC,ms=3.5,lw=1.1)
    ax2b.set_ylabel("stopped time (%)",color=AVC)
    ax2b.tick_params(axis="y",labelcolor=AVC,labelsize=7)
    letter(ax2,"b")

    # c: fleet instability
    ax3.plot(g.p,g.vstd,"o-",color="#666",ms=4)
    ax3.set_xlabel("AV penetration (%)"); ax3.set_ylabel("fleet speed s.d. (m s$^{-1}$)")
    letter(ax3,"c")

    # d: speed traces at 0% penetration
    tr0 = traces[traces.pen==0]
    colors_d = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    for j, vi in enumerate([5,15,25,35,45,55]):
        sub = tr0[tr0.veh==vi]
        ax4.plot(sub.t[::5], sub.v[::5], color=colors_d[j], lw=0.7, alpha=0.8)
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("speed (m s$^{-1}$)")
    ax4.set_title("0% AV (stop-and-go waves)", fontsize=7, fontweight="bold")
    letter(ax4,"d")

    # e: speed traces at 50% penetration
    tr50 = traces[traces.pen==50]
    for j, vi in enumerate([5,15,25,35,45,55]):
        sub = tr50[tr50.veh==vi]
        ax5.plot(sub.t[::5], sub.v[::5], color=colors_d[j], lw=0.7, alpha=0.8)
    ax5.set_xlabel("time (s)"); ax5.set_ylabel("speed (m s$^{-1}$)")
    ax5.set_title("50% AV (damped)", fontsize=7, fontweight="bold")
    letter(ax5,"e")

    # f: per-vehicle energy distribution
    e0 = energy[(energy.pen==0)]
    e50 = energy[energy.pen==50 & (energy.is_av==False)] if len(energy[energy.pen==50]) else energy[energy.pen==50]
    parts = ax6.violinplot([
        e0.e_kwh.clip(0, 60),
        energy[energy.pen==50].e_kwh.clip(0, 60)],
        positions=[1,2], showmedians=True, widths=0.7)
    for pc,c in zip(parts["bodies"], [HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax6.set_xticks([1,2]); ax6.set_xticklabels(["0% AV","50% AV"],fontsize=7)
    ax6.set_ylabel("per-veh. energy (kWh/100km)")
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"fig4_sim.pdf")); plt.close(fig)


# ================= ED Fig 1: Estimator bias, 4 panels =================
def ed1():
    df = pd.read_csv("p02_gain_bias.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    (ax1,ax2),(ax3,ax4) = axes

    # a: bias vs disturbance
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s = df[(df.noise==noise)&(df.L==150)&(df.g_true==1.0)]
        ax1.plot(s.sigma_d, s.bias_ratio, "o-", color=c, ms=3.5, lw=1.1,
                 label=f"$\\sigma$={noise}")
    ax1.axhline(1,color="k",lw=0.8,ls="--"); ax1.axvline(0.3,color="k",lw=0.8,ls=":")
    ax1.set_xscale("log"); ax1.set_xlabel("true disturbance (m s$^{-1}$)")
    ax1.set_ylabel("est./true gain"); ax1.legend(fontsize=6)
    letter(ax1,"a")

    # b: bias vs window length
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s2 = df[(df.noise==noise)&(df.sigma_d==0.6)&(df.g_true==1.0)]
        ax2.plot(s2.L, s2.bias_ratio, "s-", color=c, ms=3.5, lw=1.1,
                 label=f"$\\sigma$={noise}")
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xlabel("window length (steps)"); ax2.set_ylabel("est./true gain")
    ax2.legend(fontsize=6)
    letter(ax2,"b")

    # c: bias vs true gain (gain-independence)
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s3 = df[(df.noise==noise)&(df.L==150)&(df.sigma_d==0.6)]
        ax3.plot(s3.g_true, s3.bias_ratio, "D-", color=c, ms=4, lw=1.1,
                 label=f"$\\sigma$={noise}")
    ax3.axhline(1,color="k",lw=0.8,ls="--")
    ax3.set_xlabel("true gain"); ax3.set_ylabel("est./true gain")
    ax3.set_ylim(0.6, 1.4)
    letter(ax3,"c")

    # d: estimator dispersion (IQR width) vs disturbance
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s4 = df[(df.noise==noise)&(df.L==150)&(df.g_true==1.0)]
        width = s4.iqr_hi - s4.iqr_lo
        ax4.plot(s4.sigma_d, width, "v-", color=c, ms=3.5, lw=1.1,
                 label=f"$\\sigma$={noise}")
    ax4.axvline(0.3,color="k",lw=0.8,ls=":")
    ax4.set_xscale("log")
    ax4.set_xlabel("true disturbance (m s$^{-1}$)")
    ax4.set_ylabel("IQR width of est./true")
    letter(ax4,"d")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed1_bias.pdf")); plt.close(fig)


# ================= ED Fig 2: Controller validation, 3 panels =================
def ed2():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here,"platoon_probe.csv"))
    traces = pd.read_csv(os.path.join(here,"sim_speed_traces.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # a: chain gain decay
    for cfg,lab,c in [("human","all human",HC),("av","all AV",AVC)]:
        s = df[df.config==cfg].sort_values("position")
        ax1.plot(s.position, s.gain, "o-", color=c, ms=3.5, lw=1.3, label=lab)
    ax1.axhline(1,color="k",lw=0.8,ls="--")
    ax1.set_xlabel("position in platoon"); ax1.set_ylabel("chain gain")
    ax1.legend()
    letter(ax1,"a")

    # b: leader speed profile (input signal)
    lead = traces[(traces.pen==0)&(traces.veh==5)]
    ax2.plot(lead.t[::10], lead.v[::10], color="#333", lw=0.8)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("speed (m s$^{-1}$)")
    ax2.set_title("input wave profile", fontsize=7)
    letter(ax2,"b")

    # c: AV vs human speed response (vehicle 35)
    v35_0 = traces[(traces.pen==0)&(traces.veh==35)]
    v35_50 = traces[(traces.pen==50)&(traces.veh==35)]
    ax3.plot(v35_0.t[::5], v35_0.v[::5], color=HC, lw=0.8, alpha=0.8, label="human follower")
    ax3.plot(v35_50.t[::5], v35_50.v[::5], color=AVC, lw=0.8, alpha=0.8, label="AV follower")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("speed (m s$^{-1}$)")
    ax3.legend(fontsize=6)
    ax3.set_title("response of vehicle 35", fontsize=7)
    letter(ax3,"c")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed2_controller.pdf")); plt.close(fig)


# ================= ED Fig 3: Sensitivity, 4 panels =================
def ed3():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here,"p12_sensitivity.csv"))
    base = 41.09
    df["sav"] = 100 * (1 - df.e / base)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    (ax1,ax2),(ax3,ax4) = axes

    # a: savings vs penetration
    for n_av,lab in [(12,"20%"),(30,"50%"),(60,"100%")]:
        sub = df[df.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).sav.mean()
        ax1.scatter([lab]*len(sub), sub, color="#999", s=12, alpha=0.6)
        ax1.scatter([lab], [sub.median()], color=AVC, s=30, zorder=3)
    ax1.set_xlabel("AV penetration"); ax1.set_ylabel("energy saving (%)")
    letter(ax1,"a")

    # b: savings vs jerk cap (at 50%)
    sub50 = df[df.n_av==30]
    for jc in [1.0, 1.5, 2.0]:
        s = sub50[sub50.jerk_cap==jc].sav
        ax2.scatter([jc]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax2.scatter([jc],[s.median()],color=AVC,s=35,zorder=3)
    ax2.set_xlabel("jerk cap (m s$^{-3}$)"); ax2.set_ylabel("saving at 50% (%)")
    letter(ax2,"b")

    # c: savings vs anticipation gain (at 50%)
    for tc in [0.0, 0.3, 0.55]:
        s = sub50[sub50.t_close==tc].sav
        ax3.scatter([tc]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax3.scatter([tc],[s.median()],color=AVC,s=35,zorder=3)
    ax3.set_xlabel("anticipation gain"); ax3.set_ylabel("saving at 50% (%)")
    letter(ax3,"c")

    # d: savings vs desired headway (at 50%)
    for T in [1.1, 1.4, 1.7]:
        s = sub50[sub50.T_av==T].sav
        ax4.scatter([T]*len(s), s, color="#56B4E9", s=15, alpha=0.5)
        ax4.scatter([T],[s.median()],color=AVC,s=35,zorder=3)
    ax4.set_xlabel("desired headway T (s)"); ax4.set_ylabel("saving at 50% (%)")
    letter(ax4,"d")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed3_sensitivity.pdf")); plt.close(fig)


# ================= ED Fig 4: Perception validation, 4 panels =================
def ed4():
    tk = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                     usecols=["is_av","mean_speed","p95_abs_jerk","frac_aggr","mean_vsp"])
    ng = pd.read_csv("${DATA_DIR}/ngsim_tracks.csv",
                     usecols=["mean_speed","p95_abs_jerk","frac_aggr"])
    w_h = tk[(tk.is_av==0)&(tk.mean_speed>10)]
    w_a = tk[(tk.is_av==1)&(tk.mean_speed>10)]
    ng_h = ng[ng.mean_speed>10]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    (ax1,ax2),(ax3,ax4) = axes
    colors = [(w_h,HC,"Human (AV-perceived)"),
              (ng_h,NC,"Human (NGSIM camera)"),
              (w_a,AVC,"Waymo AV")]

    # a: jerk ECDF
    for s,c,_ in colors:
        v = np.sort(s.p95_abs_jerk.dropna())
        ax1.plot(v, np.arange(1,v.size+1)/v.size, color=c, lw=1.1)
    ax1.set_xscale("log"); ax1.set_xlabel("95th p.$|jerk|$ (m s$^{-3}$)")
    ax1.set_ylabel("ECDF")
    handles = [Line2D([0],[0],color=c,lw=1.2,label=l) for _,c,l in colors]
    ax1.legend(handles=handles, fontsize=6)
    letter(ax1,"a")

    # b: aggressive share ECDF
    for s,c,_ in colors:
        v = np.sort(s.frac_aggr.dropna())
        ax2.plot(v, np.arange(1,v.size+1)/v.size, color=c, lw=1.1)
    ax2.set_xlabel("aggressive share"); ax2.set_ylabel("ECDF")
    letter(ax2,"b")

    # c: median jerk by speed bin (3 sources)
    bins = [(10,15),(15,20),(20,25),(25,35)]
    blabels = ["10-15","15-20","20-25","25-35"]
    x = np.arange(len(bins))
    for (s,c,lab),off in zip(colors,[-0.22,0,0.22]):
        vals = []
        for lo,hi in bins:
            sub = s[(s.mean_speed>=lo)&(s.mean_speed<hi)]
            vals.append(sub.p95_abs_jerk.median() if len(sub)>10 else np.nan)
        ax3.bar(x+off, vals, width=0.25, color=c, label=lab.split("(")[0].strip())
    ax3.set_xticks(x); ax3.set_xticklabels(blabels, fontsize=7)
    ax3.set_xlabel("speed (m s$^{-1}$)"); ax3.set_ylabel("median 95th p. jerk")
    ax3.legend(fontsize=6)
    letter(ax3,"c")

    # d: summary comparison table-like bar
    metrics_ed = [("p95_abs_jerk","jerk"),("frac_aggr","aggr. share")]
    for i,(m,lab) in enumerate(metrics_ed):
        vals = [s[m].median() for s,_,_ in colors]
        norm = [v/vals[0] if vals[0]>0 else 1 for v in vals]
        ypos = [i*3+j for j in range(3)]
        ax4.barh(ypos, norm, color=[HC,NC,AVC], height=0.7)
        for j,v in enumerate(norm):
            ax4.text(v+0.02, i*3+j, f"{vals[j]:.2f}", va="center", fontsize=6.5)
    ax4.set_yticks([0,1,3,4]); ax4.set_yticklabels(["jerk: H-perc","jerk: NGSIM","aggr: H-perc","aggr: NGSIM"], fontsize=6.5)
    ax4.axvline(1,color=AVC,lw=0.8,ls="--")
    ax4.set_xlabel("normalized to human (AV-perceived)")
    letter(ax4,"d")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"ed4_validation.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig4(); print("Fig 3 done (6 panels: scatter+dual+line+traces+traces+violin)")
    ed1(); print("ED Fig 1 done (4 panels: line+line+line+line)")
    ed2(); print("ED Fig 2 done (3 panels: line+line+line)")
    ed3(); print("ED Fig 3 done (4 panels: scatter+scatter+scatter+scatter)")
    ed4(); print("ED Fig 4 done (4 panels: ECDF+ECDF+bar+bar)")
