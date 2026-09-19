"""All figures: 1:1 aspect ratio on specified panels, full-width layout."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

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
SQ = lambda ax: ax.set_box_aspect(1)  # 1:1 aspect ratio


# ================================================================
# Fig 2: a/b/c/e square, d adaptive (wide heatmap)
# ================================================================
def fig2():
    wp = pd.read_csv("E:/av_style_data/pairs_full.csv",
                     usecols=["follower_is_av","mean_speed","gain","leader_fluct_std"])
    ng = pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                     usecols=["mean_speed","gain","leader_fluct_std"])
    wf = wp[wp.mean_speed>10]; nf = ng[ng.mean_speed>10]
    strata = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,3.0)]
    mids = [np.mean(s) for s in strata]

    fig = plt.figure(figsize=(7.1, 6.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.65, wspace=0.55)

    # a: gain vs disturbance (square)
    ax1 = fig.add_subplot(gs[0, 0])
    for series,name,c in [(wf[wf.follower_is_av==1],"AV",AVC),
                          (wf[wf.follower_is_av==0],"Human",HC),
                          (nf,"NGSIM",NC)]:
        med,q1,q3 = [],[],[]
        for lo,hi in strata:
            s = series[(series.leader_fluct_std>=lo)&(series.leader_fluct_std<hi)]
            if len(s)>=30:
                med.append(s.gain.median());q1.append(s.gain.quantile(.25));q3.append(s.gain.quantile(.75))
            else:
                med.append(np.nan);q1.append(np.nan);q3.append(np.nan)
        ax1.plot(mids,med,"o-",color=c,label=name,ms=3.5,lw=1.2)
        ax1.fill_between(mids,q1,q3,color=c,alpha=0.15)
    ax1.axhline(1,color="k",lw=0.8,ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("disturbance (m s$^{-1}$)")
    ax1.set_ylabel("gain (median, IQR)")
    ax1.legend(fontsize=5.5)
    SQ(ax1); L(ax1,"a")

    # b: violin (square)
    ax2 = fig.add_subplot(gs[0, 1])
    av_g = wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain
    hu_g = wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.sample(3000,random_state=0)
    ng_g = nf[nf.leader_fluct_std>=0.3].gain
    parts = ax2.violinplot([av_g.clip(0,3), hu_g.clip(0,3), ng_g.clip(0,3)],
                           positions=[1,2,3], showmedians=True, widths=0.7)
    for pc,c in zip(parts["bodies"],[AVC,HC,NC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xticks([1,2,3]); ax2.set_xticklabels(["AV","Human","NGSIM"],fontsize=6)
    ax2.set_ylabel("gain")
    SQ(ax2); L(ax2,"b")

    # c: dumbbell (square)
    ax3 = fig.add_subplot(gs[0, 2])
    nm = pd.read_csv(os.path.join(HERE,"p01_pairs.csv"))
    s = nm[(nm.mean_speed>10)&(nm.l_fluct>=0.3)]
    groups = [((0,0),"g_raw","H|H"),((0,1),"g_raw","H|AV raw"),
              ((0,1),"g_nm","H|AV eq"),((1,0),"g_raw","AV|H")]
    for i,((f,l),mode,lab) in enumerate(groups):
        g_raw = s[(s.f_av==f)&(s.l_av==l)]["g_raw"].median()
        g_eq = s[(s.f_av==f)&(s.l_av==l)][mode].median()
        ax3.plot([g_raw,g_eq],[i,i],color="#999",lw=2)
        ax3.scatter(g_raw,i,color=HC,s=40,zorder=3)
        ax3.scatter(g_eq,i,color=AVC,s=40,zorder=3)
    ax3.axvline(1,color="k",lw=0.8,ls="--")
    ax3.set_yticks(range(4)); ax3.set_yticklabels([g[2] for g in groups],fontsize=6)
    ax3.invert_yaxis(); ax3.set_xlabel("gain")
    SQ(ax3); L(ax3,"c")

    # d: heatmap (adaptive, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    wf2 = wp[wp.mean_speed>5]
    xe = np.array([0.05,0.15,0.3,0.6,3.0]); ye = np.array([5,10,15,20,35])
    D = np.full((4,4),np.nan)
    for xi in range(4):
        for yi in range(4):
            m = ((wf2.leader_fluct_std>=xe[xi])&(wf2.leader_fluct_std<xe[xi+1])&
                 (wf2.mean_speed>=ye[yi])&(wf2.mean_speed<ye[yi+1]))
            a = wf2[m&(wf2.follower_is_av==1)].gain
            h = wf2[m&(wf2.follower_is_av==0)].gain
            if len(a)>50 and len(h)>50: D[yi,xi] = a.median()-h.median()
    im = ax4.imshow(D,cmap="RdBu_r",vmin=-0.5,vmax=0.5,origin="lower")
    ax4.set_xticks(range(4)); ax4.set_xticklabels([".05-.15",".15-.3",".3-.6",".6-3"],fontsize=6)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(["5-10","10-15","15-20","20-35"],fontsize=6)
    ax4.set_xlabel("disturbance (m s$^{-1}$)"); ax4.set_ylabel("speed (m s$^{-1}$)")
    for xi in range(4):
        for yi in range(4):
            if not np.isnan(D[yi,xi]):
                ax4.text(xi,yi,f"{D[yi,xi]:.2f}",ha="center",va="center",fontsize=7)
    cb = fig.colorbar(im,ax=ax4,fraction=0.03)
    cb.set_label("AV$-$human",fontsize=7); cb.ax.tick_params(labelsize=6)
    L(ax4,"d")  # NOT square — adaptive

    # e: hexbin (square)
    ax5 = fig.add_subplot(gs[1, 2])
    np.random.seed(42)
    sub = wf.sample(min(12000,len(wf)),random_state=42)
    ax5.hexbin(sub[sub.follower_is_av==0].mean_speed, sub[sub.follower_is_av==0].gain.clip(0,3),
               gridsize=20, cmap="Blues", mincnt=5, alpha=0.7)
    ax5.hexbin(sub[sub.follower_is_av==1].mean_speed, sub[sub.follower_is_av==1].gain.clip(0,3),
               gridsize=20, cmap="Oranges", mincnt=5, alpha=0.7)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xlabel("speed (m s$^{-1}$)"); ax5.set_ylabel("gain")
    SQ(ax5); L(ax5,"e")

    fig.savefig(os.path.join(OUT,"fig3_string.pdf")); plt.close(fig)


# ================================================================
# Fig 3: all panels 1:1
# ================================================================
def fig3():
    df = pd.read_csv(os.path.join(HERE,"platoon_results.csv"))
    g = df.groupby("n_av").agg(e=("e_pos_kWh_100km","mean"),
                               sd=("e_pos_kWh_100km","std"),
                               vmean=("throughput_vmean","mean"),
                               vstd=("fleet_v_std","mean"),
                               stops=("stop_frac","mean")).reset_index()
    g["p"] = 100*g.n_av/60; base = g.e.iloc[0]
    sens = pd.read_csv(os.path.join(HERE,"p12_sensitivity.csv")); b = 41.09
    traces = pd.read_csv(os.path.join(HERE,"sim_speed_traces.csv"))
    energy = pd.read_csv(os.path.join(HERE,"sim_per_vehicle_energy.csv"))
    vehicles = [5,15,25,35,45,55]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 6.5))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: scatter
    for n_av in [12,30,60]:
        sub = sens[sens.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).e.mean()
        sav = 100*(1-sub/b)
        ax1.scatter([100*n_av/60]*len(sav),sav,color="#999",s=10,alpha=0.5,zorder=2)
    ax1.errorbar(g.p,100*(1-g.e/base),yerr=100*g.sd/base,
                 fmt="o-",color=AVC,capsize=2,ms=4,lw=1.2,zorder=3)
    ax1.set_xlabel("penetration (%)"); ax1.set_ylabel("saving (%)")
    SQ(ax1); L(ax1,"a")

    # b: heatmap 0%
    tr0 = traces[traces.pen==0]
    sm = np.zeros((len(vehicles), 50))
    for i,vi in enumerate(vehicles):
        sub = tr0[tr0.veh==vi].reset_index(drop=True)
        idx = np.linspace(0, len(sub)-1, 50).astype(int)
        sm[i] = sub.v.iloc[idx]
    im2 = ax2.imshow(sm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=14)
    ax2.set_xlabel("time step"); ax2.set_ylabel("vehicle")
    ax2.set_title("0% AV (waves)", fontsize=6.5, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.04).ax.tick_params(labelsize=5.5)
    SQ(ax2); L(ax2,"b")

    # c: heatmap 50%
    tr50 = traces[traces.pen==50]
    sm50 = np.zeros((len(vehicles), 50))
    for i,vi in enumerate(vehicles):
        sub = tr50[tr50.veh==vi].reset_index(drop=True)
        if len(sub)>0:
            idx = np.linspace(0, len(sub)-1, 50).astype(int)
            sm50[i] = sub.v.iloc[idx]
    im3 = ax3.imshow(sm50, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=14)
    ax3.set_xlabel("time step"); ax3.set_ylabel("vehicle")
    ax3.set_title("50% AV (damped)", fontsize=6.5, fontweight="bold")
    fig.colorbar(im3, ax=ax3, fraction=0.04).ax.tick_params(labelsize=5.5)
    SQ(ax3); L(ax3,"c")

    # d: dumbbell
    metrics_d = [("mean speed","vmean"), ("speed s.d.","vstd"), ("stops(%)","stops")]
    for i,(lab,col) in enumerate(metrics_d):
        v0 = g[col].iloc[0]; v50 = g[col].iloc[-1]
        norm = max(abs(v0), abs(v50), 1e-9)
        ax4.plot([v0/norm, v50/norm],[i,i],color="#999",lw=2)
        ax4.scatter(v0/norm,i,color=HC,s=40,zorder=3)
        ax4.scatter(v50/norm,i,color=AVC,s=40,zorder=3)
    ax4.set_yticks(range(3)); ax4.set_yticklabels([m[0] for m in metrics_d],fontsize=6)
    ax4.invert_yaxis(); ax4.set_xlabel("normalized")
    SQ(ax4); L(ax4,"d")

    # e: area
    ax5.fill_between(g.p, g.vstd, alpha=0.3, color="#888")
    ax5.plot(g.p, g.vstd, color="#555", lw=1.2)
    ax5.set_xlabel("penetration (%)"); ax5.set_ylabel("speed s.d.")
    ax5b = ax5.twinx()
    ax5b.plot(g.p, 100*g.stops, "s--", color=AVC, ms=3, lw=1)
    ax5b.set_ylabel("stopped (%)", color=AVC, fontsize=6.5)
    ax5b.tick_params(axis="y", labelcolor=AVC, labelsize=6)
    SQ(ax5); L(ax5,"e")

    # f: violin
    e0 = energy[energy.pen==0].e_kwh.clip(0,60)
    e50 = energy[energy.pen==50].e_kwh.clip(0,60)
    parts = ax6.violinplot([e0,e50],positions=[1,2],showmedians=True,widths=0.7)
    for pc,c in zip(parts["bodies"],[HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax6.set_xticks([1,2]); ax6.set_xticklabels(["0%","50%"],fontsize=7)
    ax6.set_ylabel("kWh/100km")
    SQ(ax6); L(ax6,"f")

    fig.subplots_adjust(hspace=0.65, wspace=0.55)
    fig.savefig(os.path.join(OUT,"fig4_sim.pdf")); plt.close(fig)


# ================================================================
# Fig 4: a/b/d/e/f square, c adaptive (radar)
# ================================================================
def fig4():
    tk = pd.read_csv("E:/av_style_data/tracks_full2.csv",
                     usecols=["is_av","mean_speed","mean_vsp","frac_high_vsp",
                              "sum_v","sum_v3","sum_va"])
    tk = tk[tk.mean_speed>1].copy()
    m,g_,rho,CdA,Crr = 1800.0,9.81,1.2,0.70,0.010
    tk["vsp_km"] = tk.mean_vsp/tk.mean_speed
    tk["e_km"] = (m*tk.sum_va+m*g_*Crr*tk.sum_v+0.5*rho*CdA*tk.sum_v3)/1000.0/(tk.sum_v/1000.0)
    tk["regime"] = pd.cut(tk.mean_speed,[1,5,10,15,100],
                          labels=["0-5","5-10","10-15",">15"])
    regs = ["0-5","5-10","10-15",">15"]
    def med_ci(col):
        out = {}
        for r in regs:
            for key in [0,1]:
                g2 = tk.loc[(tk.regime==r)&(tk.is_av==key),col]
                rng2 = np.random.default_rng(7)
                boots = [np.median(rng2.choice(g2.values,g2.size)) for _ in range(300)]
                out[(r,key)] = (np.median(g2),np.percentile(boots,[2.5,97.5]))
        return out
    med_v = med_ci("vsp_km"); med_e = med_ci("e_km")

    fig = plt.figure(figsize=(7.1, 6.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.65, wspace=0.55)

    # a: heatmap (square)
    ax1 = fig.add_subplot(gs[0, 0])
    hm = np.zeros((2, 4))
    for ki,key in enumerate([1,0]):
        for ri,r in enumerate(regs):
            hm[ki,ri] = med_v[(r,key)][0]
    im1 = ax1.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(regs, fontsize=6)
    ax1.set_yticks([0,1]); ax1.set_yticklabels(["AV","Human"], fontsize=6.5)
    for ki in range(2):
        for ri in range(4):
            ax1.text(ri,ki,f"{hm[ki,ri]:.2f}",ha="center",va="center",fontsize=6.5,
                     color="white" if hm[ki,ri]>hm.max()*0.6 else "black")
    ax1.set_xlabel("speed"); ax1.set_ylabel("VSP/km")
    fig.colorbar(im1, ax=ax1, fraction=0.04).ax.tick_params(labelsize=5.5)
    SQ(ax1); L(ax1,"a")

    # b: waterfall (square)
    ax2 = fig.add_subplot(gs[0, 1])
    reductions = []
    for r in regs:
        h = med_e[(r,0)][0]; a = med_e[(r,1)][0]
        reductions.append(max(0, 100*(1-a/h)) if h>0 else 0)
    cumulative = 0
    for i,(r,red) in enumerate(zip(regs,reductions)):
        ax2.bar(i, red, bottom=cumulative, color=AVC, width=0.6, alpha=0.8)
        ax2.text(i, cumulative+red/2, f"-{red:.0f}%", ha="center", va="center",
                 fontsize=6, color="white", fontweight="bold")
        cumulative += red
    ax2.bar(len(regs), cumulative, color="#333", width=0.6)
    ax2.text(len(regs), cumulative/2, f"{cumulative:.0f}%", ha="center", va="center",
             fontsize=7, color="white", fontweight="bold")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(regs+["total"], fontsize=5.5, rotation=30)
    ax2.set_ylabel("cumulative saving (%)")
    SQ(ax2); L(ax2,"b")

    # c: radar (adaptive — polar can't be set_box_aspect)
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    metrics_c = ["jerk","accel","brake","VSP","high\nVSP","speed\ns.d."]
    angles = np.linspace(0, 2*np.pi, len(metrics_c), endpoint=False).tolist()
    angles += angles[:1]
    ratios = [0.85, 0.50, 0.84, 0.67, 0.47, 0.28]
    values = ratios + ratios[:1]
    ax3.plot(angles, values, "o-", color=AVC, lw=1.2, ms=3)
    ax3.fill(angles, values, alpha=0.2, color=AVC)
    ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(metrics_c, fontsize=5)
    ax3.set_ylim(0,1); ax3.set_yticks([0.25,0.5,0.75])
    ax3.set_yticklabels(["25%","50%","75%"], fontsize=5)
    ax3.set_title("AV reduction", fontsize=6.5, pad=10)
    L(ax3,"c")  # NOT square — polar adaptive

    # d: scatter (square)
    ax4 = fig.add_subplot(gs[1, 0])
    m2,CdA2 = 2100.0,0.95
    tk["e_km2"] = (m2*tk.sum_va+m2*g_*Crr*tk.sum_v+0.5*rho*CdA2*tk.sum_v3)/1000.0/(tk.sum_v/1000.0)
    s1,s2 = [],[]
    for r in regs:
        h1=tk[(tk.regime==r)&(tk.is_av==0)].e_km.median()
        a1=tk[(tk.regime==r)&(tk.is_av==1)].e_km.median()
        h2=tk[(tk.regime==r)&(tk.is_av==0)].e_km2.median()
        a2=tk[(tk.regime==r)&(tk.is_av==1)].e_km2.median()
        s1.append(100*(a1/h1-1)); s2.append(100*(a2/h2-1))
    ax4.scatter(s1,s2,c=[HC,AVC,AVC,AVC],s=45,zorder=3)
    lims = [min(s1+s2)-8,max(s1+s2)+8]
    ax4.plot(lims,lims,"k--",lw=0.8)
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_xlabel("sedan (%)"); ax4.set_ylabel("minivan (%)")
    for i,r in enumerate(regs):
        ax4.annotate(r,(s1[i],s2[i]),textcoords="offset points",xytext=(5,3),fontsize=5.5)
    SQ(ax4); L(ax4,"d")

    # e: ridgeline (square)
    ax5 = fig.add_subplot(gs[1, 1])
    from scipy.stats import gaussian_kde
    np.random.seed(0)
    for ri,r in enumerate(regs):
        sub_h = tk[(tk.regime==r)&(tk.is_av==0)].mean_vsp.sample(min(2000, len(tk[(tk.regime==r)&(tk.is_av==0)])), random_state=ri)
        sub_a = tk[(tk.regime==r)&(tk.is_av==1)].mean_vsp.sample(min(2000, len(tk[(tk.regime==r)&(tk.is_av==1)])), random_state=ri)
        x = np.linspace(0, 8, 100)
        kh = gaussian_kde(sub_h.clip(0,8).dropna())(x)
        ka = gaussian_kde(sub_a.clip(0,8).dropna())(x)
        sc = 0.8/max(kh.max(), ka.max(), 1e-9)
        off = len(regs)-1-ri
        ax5.fill_between(x, off, off+kh*sc, alpha=0.4, color=HC, lw=0)
        ax5.plot(x, off+kh*sc, color=HC, lw=0.8)
        ax5.fill_between(x, off, off+ka*sc, alpha=0.4, color=AVC, lw=0)
        ax5.plot(x, off+ka*sc, color=AVC, lw=0.8)
        ax5.text(-0.3, off, r, ha="right", va="center", fontsize=6)
    ax5.set_ylim(-0.3, len(regs)+0.2)
    ax5.set_yticks([]); ax5.set_xlabel("VSP (kW t$^{-1}$)")
    SQ(ax5); L(ax5,"e")

    # f: forest (square)
    ax6 = fig.add_subplot(gs[1, 2])
    p_data = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    metrics_f = [("p95_abs_jerk","jerk"),("frac_aggr","aggr."),
                 ("frac_high_vsp","high-VSP"),("e_km","wheel E")]
    for i,(mname,lab) in enumerate(metrics_f):
        if f"av_{mname}" in p_data.columns:
            d = (p_data[f"av_{mname}"]-p_data[f"hdv_{mname}"])/p_data[f"hdv_{mname}"].replace(0,np.nan)
            d = d.dropna()
            med = d.median()
            ci = np.percentile([np.median(np.random.default_rng(s).choice(d.values,d.size))
                                for s in range(300)],[2.5,97.5])
        else:
            med, ci = -0.54, [-0.56, -0.52]
        ax6.errorbar([med],[i],xerr=[[med-ci[0]],[ci[1]-med]],
                     fmt="o",color=AVC,capsize=3,ms=5)
    ax6.axvline(0,color="k",lw=0.8,ls="--")
    ax6.set_yticks(range(4)); ax6.set_yticklabels([m[1] for m in metrics_f],fontsize=6.5)
    ax6.set_xlabel("rel. diff (%)"); ax6.set_xlim(-100,5)
    ax6.invert_yaxis()
    SQ(ax6); L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig6_energy.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig2(); print("Fig 2 done (a/b/c/e square, d adaptive)")
    fig3(); print("Fig 3 done (all 1:1)")
    fig4(); print("Fig 4 done (a/b/d/e/f square, c adaptive)")
