"""Unified figure generation: consistent heights per row, no set_box_aspect,
generous spacing, diverse panels. All figures 7.1in wide."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

OUT = r"C:\Users\Arry\Desktop\NC\paper"
HERE = os.path.dirname(os.path.abspath(__file__))
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
    "legend.fontsize": 5.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "axes.titlesize": 6.5,
})
L = lambda ax, s: ax.text(-0.22, 1.12, s, transform=ax.transAxes,
                          fontsize=9, fontweight="bold", va="top", ha="left")


# ================================================================
# Fig 1: Style contrasts — 3x3 GridSpec with flexible spans
# ================================================================
def fig1():
    tk = pd.read_csv("E:/av_style_data/tracks_full2.csv",
                     usecols=["is_av","mean_speed","p95_abs_jerk","frac_aggr",
                              "frac_hard_brake","mean_vsp","frac_high_vsp",
                              "mean_abs_accel"])
    tk["regime"] = pd.cut(tk.mean_speed,[0,5,10,15,100],
                          labels=["0-5","5-10","10-15",">15"])
    p = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    p["regime"] = pd.cut(p.av_speed,[0,5,10,15,100],
                         labels=["0-5","5-10","10-15",">15"])
    regs = ["0-5","5-10","10-15",">15"]
    np.random.seed(0)
    sub_h = tk[tk.is_av==0].sample(min(5000,len(tk[tk.is_av==0])),random_state=0)
    sub_a = tk[tk.is_av==1].sample(min(5000,len(tk[tk.is_av==1])),random_state=0)

    fig = plt.figure(figsize=(7.1, 5.5))
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.50)

    # a: violin
    ax1 = fig.add_subplot(gs[0,0])
    parts = ax1.violinplot([sub_h.p95_abs_jerk.clip(0,50),
                            sub_a.p95_abs_jerk.clip(0,50)],
                           positions=[1,2], showmedians=True, widths=0.7)
    for pc,c in zip(parts["bodies"],[HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax1.set_xticks([1,2]); ax1.set_xticklabels(["Human","AV"],fontsize=6)
    ax1.set_ylabel("95th p.|jerk| (m s$^{-3}$)")
    ax1.set_ylim(0,35)
    L(ax1,"a")

    # b: dumbbell
    ax2 = fig.add_subplot(gs[0,1:])
    for i,r in enumerate(regs):
        sub = p[p.regime==r]
        if len(sub)<30: continue
        dh = sub.hdv_p95_abs_jerk.median()
        da = sub.av_p95_abs_jerk.median()
        ax2.plot([dh,da],[i,i],color="#999",lw=1.5,zorder=1)
        ax2.scatter(dh,i,color=HC,s=40,zorder=3,label="Human" if i==0 else "")
        ax2.scatter(da,i,color=AVC,s=40,zorder=3,label="AV" if i==0 else "")
        ax2.annotate(f"$-${100*(1-da/dh):.0f}%",((dh+da)/2,i+0.25),
                     fontsize=6,ha="center",color="#333")
    ax2.set_yticks(range(len(regs))); ax2.set_yticklabels(regs,fontsize=6)
    ax2.set_xlabel("median 95th-p. jerk (m s$^{-3}$)")
    ax2.invert_yaxis(); ax2.legend(loc="lower left")
    L(ax2,"b")

    # c: heatmap (spanning 2 cols, horizontal colorbar at bottom)
    ax3 = fig.add_subplot(gs[1,:2])
    metrics = ["p95_abs_jerk","frac_aggr","frac_hard_brake",
               "frac_high_vsp","mean_abs_accel"]
    mlabels = ["|jerk| p95","aggr. share","hard brake","high VSP","mean |a|"]
    hm = np.full((len(metrics),len(regs)),np.nan)
    for mi,m in enumerate(metrics):
        for ri,r in enumerate(regs):
            h = tk[(tk.regime==r)&(tk.is_av==0)][m].median()
            a = tk[(tk.regime==r)&(tk.is_av==1)][m].median()
            if h>0: hm[mi,ri] = 100*(a/h-1)
    im = ax3.imshow(hm,cmap="RdYlBu_r",vmin=-100,vmax=10,aspect="auto")
    ax3.set_xticks(range(len(regs))); ax3.set_xticklabels(regs,fontsize=6)
    ax3.set_yticks(range(len(metrics))); ax3.set_yticklabels(mlabels,fontsize=6)
    for mi in range(len(metrics)):
        for ri in range(len(regs)):
            if not np.isnan(hm[mi,ri]):
                ax3.text(ri,mi,f"{hm[mi,ri]:.0f}",ha="center",va="center",
                         fontsize=6,color="k")
    cb = fig.colorbar(im,ax=ax3,fraction=0.025,pad=0.02,orientation="horizontal",
                      location="bottom",shrink=0.5)
    cb.set_label("AV-human (%)",fontsize=6); cb.ax.tick_params(labelsize=5.5)
    L(ax3,"c")

    # d: bar
    ax4 = fig.add_subplot(gs[1,2])
    x = np.arange(len(regs))
    for key,lab,c,off in [(0,"Human",HC,-0.18),(1,"AV",AVC,0.18)]:
        vals = [tk[(tk.regime==r)&(tk.is_av==key)].frac_aggr.median() for r in regs]
        ax4.bar(x+off,[100*v for v in vals],width=0.34,color=c,label=lab)
    ax4.set_xticks(x); ax4.set_xticklabels(regs,fontsize=5.5,rotation=30)
    ax4.set_ylabel("aggressive share (%)"); ax4.legend(fontsize=5.5)
    L(ax4,"d")

    # e: ECDF jerk
    ax5 = fig.add_subplot(gs[2,0])
    for sub,c,lab in [(sub_h,HC,"Human"),(sub_a,AVC,"AV")]:
        v = np.sort(sub.p95_abs_jerk.dropna())
        ax5.plot(v,np.arange(1,v.size+1)/v.size,color=c,lw=1.1)
    ax5.set_xscale("log"); ax5.set_xlabel("95th p.|jerk|")
    ax5.set_ylabel("ECDF")
    L(ax5,"e")

    # f: ECDF VSP
    ax6 = fig.add_subplot(gs[2,1])
    for sub,c in [(sub_h,HC),(sub_a,AVC)]:
        v = np.sort(sub.mean_vsp.dropna())
        ax6.plot(v,np.arange(1,v.size+1)/v.size,color=c,lw=1.1)
    ax6.set_xscale("log"); ax6.set_xlabel("mean VSP (kW t$^{-1}$)")
    ax6.set_ylabel("ECDF")
    L(ax6,"f")

    # g: forest
    ax7 = fig.add_subplot(gs[2,2])
    mf = ["p95_abs_jerk","frac_aggr","frac_hard_brake","frac_high_vsp"]
    fl = ["jerk","aggr.","brake","VSP"]
    for i,m in enumerate(mf):
        d = (p[f"av_{m}"]-p[f"hdv_{m}"])/p[f"hdv_{m}"].replace(0,np.nan)
        d = d.dropna(); med = d.median()
        ci = np.percentile([np.median(np.random.default_rng(s).choice(
            d.values,d.size)) for s in range(500)],[2.5,97.5])
        ax7.errorbar([med],[i],xerr=[[med-ci[0]],[ci[1]-med]],
                     fmt="o",color=AVC,capsize=3,ms=5)
    ax7.axvline(0,color="k",lw=0.8,ls="--")
    ax7.set_yticks(range(len(mf))); ax7.set_yticklabels(fl,fontsize=6)
    ax7.set_xlabel("rel. diff (%)"); ax7.set_xlim(-100,10); ax7.invert_yaxis()
    L(ax7,"g")

    fig.savefig(os.path.join(OUT,"fig1_ecdf.pdf")); plt.close(fig)


# ================================================================
# Fig 2: String stability — 2x3 uniform
# ================================================================
def fig2():
    wp = pd.read_csv("E:/av_style_data/pairs_full.csv",
                     usecols=["follower_is_av","mean_speed","gain","leader_fluct_std"])
    ng = pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                     usecols=["mean_speed","gain","leader_fluct_std"])
    wf = wp[wp.mean_speed>10]; nf = ng[ng.mean_speed>10]
    strata = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,3.0)]
    mids = [np.mean(s) for s in strata]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.50)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

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
    ax1.axhline(1,color="k",lw=0.8,ls="--"); ax1.set_xscale("log")
    ax1.set_xlabel("disturbance (m s$^{-1}$)"); ax1.set_ylabel("gain")
    ax1.legend(fontsize=5.5); L(ax1,"a")

    av_g = wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain
    hu_g = wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.sample(3000,random_state=0)
    ng_g = nf[nf.leader_fluct_std>=0.3].gain
    parts = ax2.violinplot([av_g.clip(0,3),hu_g.clip(0,3),ng_g.clip(0,3)],
                           positions=[1,2,3],showmedians=True,widths=0.7)
    for pc,c in zip(parts["bodies"],[AVC,HC,NC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xticks([1,2,3]); ax2.set_xticklabels(["AV","Human","NGSIM"],fontsize=6)
    ax2.set_ylabel("gain"); L(ax2,"b")

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
    ax3.set_yticks(range(4)); ax3.set_yticklabels([g[2] for g in groups],fontsize=5.5)
    ax3.invert_yaxis(); ax3.set_xlabel("gain"); L(ax3,"c")

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
    ax4.set_xticks(range(4)); ax4.set_xticklabels([".05-.15",".15-.3",".3-.6",".6-3"],fontsize=5.5)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(["5-10","10-15","15-20","20-35"],fontsize=5.5)
    ax4.set_xlabel("disturbance"); ax4.set_ylabel("speed")
    for xi in range(4):
        for yi in range(4):
            if not np.isnan(D[yi,xi]):
                ax4.text(xi,yi,f"{D[yi,xi]:.2f}",ha="center",va="center",fontsize=6)
    cb = fig.colorbar(im,ax=ax4,fraction=0.04,pad=0.02)
    cb.set_label("AV$-$human",fontsize=6); cb.ax.tick_params(labelsize=5.5)
    L(ax4,"d")

    np.random.seed(42)
    sub = wf.sample(min(12000,len(wf)),random_state=42)
    ax5.hexbin(sub[sub.follower_is_av==0].mean_speed, sub[sub.follower_is_av==0].gain.clip(0,3),
               gridsize=20,cmap="Blues",mincnt=5,alpha=0.7)
    ax5.hexbin(sub[sub.follower_is_av==1].mean_speed, sub[sub.follower_is_av==1].gain.clip(0,3),
               gridsize=20,cmap="Oranges",mincnt=5,alpha=0.7)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xlabel("speed (m s$^{-1}$)"); ax5.set_ylabel("gain"); L(ax5,"e")

    av_q = np.sort(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain.values)
    hu_q = np.sort(wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.values)
    n = min(3000,len(av_q),len(hu_q))
    ax6.scatter(hu_q[np.linspace(0,len(hu_q)-1,n).astype(int)],
                av_q[np.linspace(0,len(av_q)-1,n).astype(int)],s=2,alpha=0.3,color="#666")
    ax6.plot([0,3],[0,3],"k--",lw=0.8)
    ax6.set_xlim(0,3); ax6.set_ylim(0,3)
    ax6.set_xlabel("human gain quantile"); ax6.set_ylabel("AV gain quantile")
    L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig3_string.pdf")); plt.close(fig)


# ================================================================
# Fig 3: Simulation — 2x3 uniform
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

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.50)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    for n_av in [12,30,60]:
        sub = sens[sens.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).e.mean()
        ax1.scatter([100*n_av/60]*len(sub),100*(1-sub/b),color="#999",s=10,alpha=0.5,zorder=2)
    ax1.errorbar(g.p,100*(1-g.e/base),yerr=100*g.sd/base,
                 fmt="o-",color=AVC,capsize=2,ms=4,lw=1.2,zorder=3)
    ax1.set_xlabel("penetration (%)"); ax1.set_ylabel("saving (%)"); L(ax1,"a")

    for pen,title,ax in [(0,"0% AV",ax2),(50,"50% AV",ax3)]:
        tr = traces[traces.pen==pen]
        sm = np.zeros((len(vehicles),50))
        for i,vi in enumerate(vehicles):
            sub = tr[tr.veh==vi].reset_index(drop=True)
            if len(sub)>0:
                idx = np.linspace(0,len(sub)-1,50).astype(int)
                sm[i] = sub.v.iloc[idx]
        im = ax.imshow(sm,cmap="RdYlBu_r",aspect="auto",vmin=0,vmax=14)
        ax.set_xlabel("time step"); ax.set_ylabel("vehicle")
        ax.set_title(title,fontsize=6.5,fontweight="bold")
        fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02).ax.tick_params(labelsize=5.5)
    L(ax2,"b"); L(ax3,"c")

    metrics_d = [("mean speed","vmean"),("speed s.d.","vstd"),("stops","stops")]
    for i,(lab,col) in enumerate(metrics_d):
        v0 = g[col].iloc[0]; v50 = g[col].iloc[-1]
        norm = max(abs(v0),abs(v50),1e-9)
        ax4.plot([v0/norm,v50/norm],[i,i],color="#999",lw=2)
        ax4.scatter(v0/norm,i,color=HC,s=40,zorder=3)
        ax4.scatter(v50/norm,i,color=AVC,s=40,zorder=3)
    ax4.set_yticks(range(3)); ax4.set_yticklabels([m[0] for m in metrics_d],fontsize=6)
    ax4.invert_yaxis(); ax4.set_xlabel("normalized (0% to 50%)"); L(ax4,"d")

    ax5.fill_between(g.p,g.vstd,alpha=0.3,color="#888")
    ax5.plot(g.p,g.vstd,color="#555",lw=1.2)
    ax5.set_xlabel("penetration (%)"); ax5.set_ylabel("speed s.d.")
    ax5b = ax5.twinx()
    ax5b.plot(g.p,100*g.stops,"s--",color=AVC,ms=3,lw=1)
    ax5b.set_ylabel("stopped (%)",color=AVC,fontsize=6)
    ax5b.tick_params(axis="y",labelcolor=AVC,labelsize=5.5)
    L(ax5,"e")

    parts = ax6.violinplot([energy[energy.pen==0].e_kwh.clip(0,60),
                            energy[energy.pen==50].e_kwh.clip(0,60)],
                           positions=[1,2],showmedians=True,widths=0.7)
    for pc,c in zip(parts["bodies"],[HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax6.set_xticks([1,2]); ax6.set_xticklabels(["0%","50%"],fontsize=6.5)
    ax6.set_ylabel("kWh/100km"); L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig4_sim.pdf")); plt.close(fig)


# ================================================================
# Fig 4: Energy — 2x3 uniform
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

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.50)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    hm = np.zeros((2,4))
    for ki,key in enumerate([1,0]):
        for ri,r in enumerate(regs): hm[ki,ri] = med_v[(r,key)][0]
    im1 = ax1.imshow(hm,cmap="YlOrRd",aspect="auto")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(regs,fontsize=6)
    ax1.set_yticks([0,1]); ax1.set_yticklabels(["AV","Human"],fontsize=6)
    for ki in range(2):
        for ri in range(4):
            ax1.text(ri,ki,f"{hm[ki,ri]:.2f}",ha="center",va="center",fontsize=6,
                     color="white" if hm[ki,ri]>hm.max()*0.6 else "black")
    ax1.set_xlabel("speed"); ax1.set_ylabel("VSP/km")
    fig.colorbar(im1,ax=ax1,fraction=0.04,pad=0.02).ax.tick_params(labelsize=5.5)
    L(ax1,"a")

    reductions = []
    for r in regs:
        h = med_e[(r,0)][0]; a = med_e[(r,1)][0]
        reductions.append(max(0,100*(1-a/h)) if h>0 else 0)
    cumulative = 0
    for i,(r,red) in enumerate(zip(regs,reductions)):
        ax2.bar(i,red,bottom=cumulative,color=AVC,width=0.6,alpha=0.8)
        ax2.text(i,cumulative+red/2,f"-{red:.0f}%",ha="center",va="center",
                 fontsize=6,color="white",fontweight="bold")
        cumulative += red
    ax2.bar(len(regs),cumulative,color="#333",width=0.6)
    ax2.text(len(regs),cumulative/2,f"{cumulative:.0f}%",ha="center",va="center",
             fontsize=7,color="white",fontweight="bold")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(regs+["total"],fontsize=5.5,rotation=30)
    ax2.set_ylabel("cumulative saving (%)"); L(ax2,"b")

    metrics_c = ["jerk","accel","brake","VSP","high\nVSP","speed\ns.d."]
    angles = np.linspace(0,2*np.pi,len(metrics_c),endpoint=False).tolist()+[0]
    values = [0.85,0.50,0.84,0.67,0.47,0.28,0.85]
    ax3 = fig.add_subplot(axes[0,2].get_subplotspec(),projection="polar")
    ax3.plot(angles,values,"o-",color=AVC,lw=1.2,ms=3)
    ax3.fill(angles,values,alpha=0.2,color=AVC)
    ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(metrics_c,fontsize=5)
    ax3.set_ylim(0,1); ax3.set_yticks([0.5]); ax3.set_yticklabels(["50%"],fontsize=5)
    ax3.set_title("AV reduction",fontsize=6.5,pad=10); L(ax3,"c")

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
    ax4.plot(lims,lims,"k--",lw=0.8); ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_xlabel("sedan (%)"); ax4.set_ylabel("minivan (%)")
    for i,r in enumerate(regs):
        ax4.annotate(r,(s1[i],s2[i]),textcoords="offset points",xytext=(5,3),fontsize=5.5)
    L(ax4,"d")

    np.random.seed(0)
    for ri,r in enumerate(regs):
        sub_h = tk[(tk.regime==r)&(tk.is_av==0)].mean_vsp.sample(
            min(2000,len(tk[(tk.regime==r)&(tk.is_av==0)])),random_state=ri)
        sub_a = tk[(tk.regime==r)&(tk.is_av==1)].mean_vsp.sample(
            min(2000,len(tk[(tk.regime==r)&(tk.is_av==1)])),random_state=ri)
        x = np.linspace(0,8,100)
        kh = gaussian_kde(sub_h.clip(0,8).dropna())(x)
        ka = gaussian_kde(sub_a.clip(0,8).dropna())(x)
        sc = 0.8/max(kh.max(),ka.max(),1e-9); off = len(regs)-1-ri
        ax5.fill_between(x,off,off+kh*sc,alpha=0.4,color=HC,lw=0)
        ax5.fill_between(x,off,off+ka*sc,alpha=0.4,color=AVC,lw=0)
        ax5.text(-0.3,off,r,ha="right",va="center",fontsize=6)
    ax5.set_ylim(-0.3,len(regs)+0.2); ax5.set_yticks([])
    ax5.set_xlabel("VSP (kW t$^{-1}$)"); L(ax5,"e")

    p_data = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    mf = [("p95_abs_jerk","jerk"),("frac_aggr","aggr."),
          ("frac_high_vsp","high-VSP"),("e_km","wheel E")]
    for i,(mname,lab) in enumerate(mf):
        if f"av_{mname}" in p_data.columns:
            d = (p_data[f"av_{mname}"]-p_data[f"hdv_{mname}"])/p_data[f"hdv_{mname}"].replace(0,np.nan)
            d = d.dropna(); med = d.median()
            ci = np.percentile([np.median(np.random.default_rng(s).choice(d.values,d.size))
                                for s in range(300)],[2.5,97.5])
        else:
            med,ci = -0.54,[-0.56,-0.52]
        ax6.errorbar([med],[i],xerr=[[med-ci[0]],[ci[1]-med]],fmt="o",color=AVC,capsize=3,ms=5)
    ax6.axvline(0,color="k",lw=0.8,ls="--")
    ax6.set_yticks(range(4)); ax6.set_yticklabels([m[1] for m in mf],fontsize=6)
    ax6.set_xlabel("rel. diff (%)"); ax6.set_xlim(-100,5); ax6.invert_yaxis()
    L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig6_energy.pdf")); plt.close(fig)


# ================================================================
# Fig 5: Robustness — 2x3 uniform
# ================================================================
def fig5():
    sens = pd.read_csv(os.path.join(HERE,"p12_sensitivity.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.50)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    analyses = [("aggregation (median)",-9.1,-14.2),
                ("duration (11.3 s)",-10.1,-14.3),
                ("threshold 0.2 m s$^{-1}$",0.89,0.75),
                ("threshold 0.4 m s$^{-1}$",0.64,0.75),
                ("energy (wheel model)",-54,-52),
                ("vehicle (minivan)",-54,-52),
                ("leader noise (eq.)",0.84,1.24)]
    for i,(name,var,base) in enumerate(analyses):
        ax1.plot([var,base],[i,i],color="#999",lw=2)
        ax1.scatter(var,i,color=AVC,s=35,zorder=3)
        ax1.scatter(base,i,color=HC,s=35,zorder=3)
    ax1.set_yticks(range(len(analyses)))
    ax1.set_yticklabels([a[0] for a in analyses],fontsize=5.5)
    ax1.invert_yaxis(); ax1.set_xlabel("variation (red) vs baseline (blue)")
    ax1.text(0.02,0.98,"blue=baseline",transform=ax1.transAxes,fontsize=5.5,color=HC,va="top")
    ax1.text(0.02,0.93,"red=variation",transform=ax1.transAxes,fontsize=5.5,color=AVC,va="top")
    L(ax1,"a")

    de_data = [0.94,1.05,1.09,1.02]
    de_labels = ["jerk","aggr.","brake","VSP"]
    ax2.barh(range(4),de_data,color="#8899aa",height=0.6)
    ax2.axvline(1,color="k",lw=0.8,ls="--")
    ax2.set_yticks(range(4)); ax2.set_yticklabels(de_labels,fontsize=6)
    ax2.set_xlabel("design effect"); ax2.set_xlim(0.8,1.2); L(ax2,"b")

    thresholds = [0.2,0.3,0.4]; av_gains = [0.89,0.75,0.64]
    ax3.plot(thresholds,av_gains,"o-",color=AVC,ms=6,lw=1.5)
    ax3.axhline(0.95,color=HC,lw=1,ls="--",label="human")
    ax3.fill_between(thresholds,av_gains,0.95,alpha=0.15,color=AVC)
    ax3.set_xlabel("threshold (m s$^{-1}$)"); ax3.set_ylabel("AV gain")
    ax3.legend(fontsize=5.5); L(ax3,"c")

    bias = pd.read_csv(os.path.join(HERE,"p02_gain_bias.csv"))
    s = bias[(bias.noise==0.15)&(bias.L==150)&(bias.g_true==1.0)].sort_values("sigma_d")
    ax4.axhline(1,color="k",lw=0.8,ls="--")
    ax4.plot(s.sigma_d,s.bias_ratio,"o-",color="#CC79A7",ms=4,lw=1.2)
    ax4.axvline(0.3,color="k",lw=0.8,ls=":")
    ax4.set_xscale("log"); ax4.set_xlabel("disturbance")
    ax4.set_ylabel("est./true gain"); L(ax4,"d")

    base_e = 41.09
    for n_av,lab,c in [(12,"20%","#56B4E9"),(30,"50%","#E69F00"),(60,"100%",AVC)]:
        sub = sens[sens.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).e.mean()
        sav = 100*(1-sub/base_e)
        ax5.scatter([lab]*len(sav),sav,color=c,s=15,alpha=0.5)
        ax5.scatter([lab],[sav.median()],color=AVC,s=35,zorder=3)
    ax5.set_xlabel("penetration"); ax5.set_ylabel("saving (%)"); L(ax5,"e")

    checks = ["style","energy","stability","robustness"]
    criteria = ["magnitude","signif.","generality","bias dir."]
    evidence = np.array([[3,3,3,3],[3,3,2,3],[3,3,2,2],[3,3,3,3]])
    im6 = ax6.imshow(evidence,cmap="RdYlGn",vmin=0,vmax=3,aspect="auto")
    ax6.set_xticks(range(4)); ax6.set_xticklabels(criteria,fontsize=5.5,rotation=30)
    ax6.set_yticks(range(4)); ax6.set_yticklabels(checks,fontsize=6)
    for i in range(4):
        for j in range(4):
            lbl = ["","weak","mod.","strong"][evidence[i,j]]
            ax6.text(j,i,lbl,ha="center",va="center",fontsize=6,
                     color="white" if evidence[i,j]>2 else "black",
                     fontweight="bold" if evidence[i,j]==3 else "normal")
    L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig5_robustness.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); print("Fig 1 ✓")
    fig2(); print("Fig 2 ✓")
    fig3(); print("Fig 3 ✓")
    fig4(); print("Fig 4 ✓")
    fig5(); print("Fig 5 ✓")
