"""Expand Fig 2 and Fig 4 to 6 panels each with diverse types."""
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
    ax.text(-0.15, 1.08, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

def ecdf(ax, series, color, **kw):
    v = np.sort(series.dropna())
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=1.1, **kw)


# ============ Fig 2: 6 panels (2x3) ============
def fig3():
    wp = pd.read_csv("E:/av_style_data/pairs_full.csv",
                     usecols=["follower_is_av","mean_speed","gain",
                              "leader_fluct_std","gain_fft"])
    ng = pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                     usecols=["mean_speed","gain","leader_fluct_std"])
    wf = wp[wp.mean_speed>10]; nf = ng[ng.mean_speed>10]
    strata = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,3.0)]
    mids = [np.mean(s) for s in strata]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.4))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: gain vs disturbance (line)
    for series,name,c in [(wf[wf.follower_is_av==1],"Waymo AV",AVC),
                          (wf[wf.follower_is_av==0],"Human (perc.)",HC),
                          (nf,"Human (NGSIM)",NC)]:
        med,q1,q3 = [],[],[]
        for lo,hi in strata:
            s = series[(series.leader_fluct_std>=lo)&(series.leader_fluct_std<hi)]
            if len(s)>=30:
                med.append(s.gain.median());q1.append(s.gain.quantile(.25));q3.append(s.gain.quantile(.75))
            else:
                med.append(np.nan);q1.append(np.nan);q3.append(np.nan)
        ax1.plot(mids,med,"o-",color=c,label=name,ms=4,lw=1.3)
        ax1.fill_between(mids,q1,q3,color=c,alpha=0.15,lw=0)
    ax1.axhline(1,color="k",lw=0.8,ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("leader disturbance (m s$^{-1}$)")
    ax1.set_ylabel("gain (median, IQR)")
    ax1.legend(loc="upper right", fontsize=6)
    letter(ax1,"a")

    # b: gain ECDF at >=0.3
    for series,name,c in [(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)],
                           "AV",AVC),
                          (nf[nf.leader_fluct_std>=0.3],"NGSIM",NC)]:
        ecdf(ax2, series.gain, c)
        ax2.annotate(name,(0.05,0.88 if c==AVC else 0.76),
                     xycoords="axes fraction",fontsize=6.5,color=c)
    ax2.axvline(1,color="k",lw=0.8,ls="--")
    ax2.set_xlim(0,3)
    ax2.set_xlabel("gain, dist.$\\geq$0.3")
    ax2.set_ylabel("ECDF")
    letter(ax2,"b")

    # c: composition bars
    nm = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "p01_pairs.csv"))
    s = nm[(nm.mean_speed>10)&(nm.l_fluct>=0.3)]
    groups = [((0,0),"g_raw","H|H",HC),
              ((0,1),"g_raw","H|AV\nraw","#88aacc"),
              ((0,1),"g_nm","H|AV\neq.","#1f4e79"),
              ((1,0),"g_raw","AV|H",AVC)]
    for i,((f,l),mode,lab,c) in enumerate(groups):
        g = s[(s.f_av==f)&(s.l_av==l)][mode]
        rng2 = np.random.default_rng(1)
        boots = [np.median(rng2.choice(g.values,g.size)) for _ in range(1000)]
        med = np.median(g); lo_,hi_ = np.percentile(boots,[2.5,97.5])
        ax3.bar(i,med,width=0.6,color=c,
                yerr=[[med-lo_],[hi_-med]],capsize=3,error_kw=dict(lw=0.8))
        ax3.text(i,hi_+0.03,f"{med:.2f}",ha="center",fontsize=6.5,color=c)
    ax3.axhline(1,color="k",lw=0.8,ls="--")
    ax3.set_xticks(range(4));ax3.set_xticklabels([g[2] for g in groups],fontsize=6.5)
    ax3.set_ylabel("gain (median, 95% CI)")
    letter(ax3,"c")

    # d: heatmap
    wf2 = wp[wp.mean_speed>5]
    xedges = np.array([0.05,0.15,0.3,0.6,3.0]); yedges = np.array([5,10,15,20,35])
    D = np.full((4,4),np.nan)
    for xi in range(4):
        for yi in range(4):
            msk = ((wf2.leader_fluct_std>=xedges[xi])&(wf2.leader_fluct_std<xedges[xi+1])&
                   (wf2.mean_speed>=yedges[yi])&(wf2.mean_speed<yedges[yi+1]))
            a = wf2[msk&(wf2.follower_is_av==1)].gain
            h = wf2[msk&(wf2.follower_is_av==0)].gain
            if len(a)>50 and len(h)>50: D[yi,xi] = a.median()-h.median()
    im = ax4.imshow(D,cmap="RdBu_r",vmin=-0.5,vmax=0.5,origin="lower",aspect="auto")
    ax4.set_xticks(range(4));ax4.set_xticklabels([".05-.15",".15-.3",".3-.6",".6-3"],fontsize=6.5)
    ax4.set_yticks(range(4));ax4.set_yticklabels(["5-10","10-15","15-20","20-35"],fontsize=6.5)
    ax4.set_xlabel("disturbance (m s$^{-1}$)")
    ax4.set_ylabel("speed (m s$^{-1}$)")
    for xi in range(4):
        for yi in range(4):
            if not np.isnan(D[yi,xi]):
                ax4.text(xi,yi,f"{D[yi,xi]:.2f}",ha="center",va="center",fontsize=6.5)
    cb = fig.colorbar(im,ax=ax4,fraction=0.045)
    cb.set_label("AV-human",fontsize=7); cb.ax.tick_params(labelsize=6.5)
    letter(ax4,"d")

    # e: hexbin scatter — gain vs speed (all freeway pairs, subsampled)
    np.random.seed(42)
    sub = wf.sample(min(15000, len(wf)), random_state=42)
    ax5.hexbin(sub[sub.follower_is_av==0].mean_speed,
               sub[sub.follower_is_av==0].gain.clip(0,3),
               gridsize=25, cmap="Blues", mincnt=5, alpha=0.7, label="Human")
    ax5.hexbin(sub[sub.follower_is_av==1].mean_speed,
               sub[sub.follower_is_av==1].gain.clip(0,3),
               gridsize=25, cmap="Oranges", mincnt=5, alpha=0.7, label="AV")
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xlabel("follower speed (m s$^{-1}$)")
    ax5.set_ylabel("fluctuation gain")
    ax5.set_ylim(0,3)
    letter(ax5,"e")

    # f: Q-Q plot AV vs human gains
    av_g = np.sort(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain.values)
    hu_g = np.sort(wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.values)
    # subsample to equal length
    n = min(3000, len(av_g), len(hu_g))
    av_q = av_g[np.linspace(0, len(av_g)-1, n).astype(int)]
    hu_q = hu_g[np.linspace(0, len(hu_g)-1, n).astype(int)]
    ax6.scatter(hu_q, av_q, s=2, alpha=0.3, color="#666")
    lims = [0, min(3, max(av_q.max(), hu_q.max()))]
    ax6.plot(lims, lims, "k--", lw=0.8, label="identity")
    ax6.set_xlim(0,3); ax6.set_ylim(0,3)
    ax6.set_xlabel("human gain quantile")
    ax6.set_ylabel("AV gain quantile")
    ax6.legend(fontsize=6)
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"fig3_string.pdf")); plt.close(fig)


# ============ Fig 4: 6 panels (2x3) ============
def fig6():
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
                boots = [np.median(rng2.choice(g2.values,g2.size)) for _ in range(500)]
                out[(r,key)] = (np.median(g2),np.percentile(boots,[2.5,97.5]))
        return out

    med_v = med_ci("vsp_km"); med_e = med_ci("e_km")
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.4))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes
    x = np.arange(len(regs))

    # a: VSP bars
    for key,lab,c,off in [(0,"Human",HC,-0.18),(1,"AV",AVC,0.18)]:
        vals = [med_v[(r,key)][0] for r in regs]
        err = ([med_v[(r,key)][0]-med_v[(r,key)][1][0] for r in regs],
               [med_v[(r,key)][1][1]-med_v[(r,key)][0] for r in regs])
        ax1.bar(x+off,vals,width=0.34,color=c,label=lab,yerr=err,capsize=2,error_kw=dict(lw=0.8))
    ax1.set_xticks(x);ax1.set_xticklabels(regs,fontsize=7)
    ax1.set_xlabel("speed (m s$^{-1}$)");ax1.set_ylabel("VSP/km (kJ t$^{-1}$)")
    ax1.legend(); letter(ax1,"a")

    # b: relative change grouped bar
    w = 0.35
    for i,(col,lab,c) in enumerate([(med_v,"VSP","#8899aa"),(med_e,"wheel","#444")]):
        rel = [100*(col[(r,1)][0]/col[(r,0)][0]-1) for r in regs]
        ax2.bar(x+(i-0.5)*w,rel,width=w,color=c,label=lab)
    ax2.axhline(0,color="k",lw=0.8)
    ax2.set_xticks(x);ax2.set_xticklabels(regs,fontsize=7)
    ax2.set_xlabel("speed (m s$^{-1}$)");ax2.set_ylabel("AV change (%)")
    ax2.set_ylim(-80,15);ax2.legend(); letter(ax2,"b")

    # c: absolute wheel energy bars
    for key,lab,c,off in [(0,"Human",HC,-0.18),(1,"AV",AVC,0.18)]:
        vals = [med_e[(r,key)][0] for r in regs]
        err = ([med_e[(r,key)][0]-med_e[(r,key)][1][0] for r in regs],
               [med_e[(r,key)][1][1]-med_e[(r,key)][0] for r in regs])
        ax3.bar(x+off,vals,width=0.34,color=c,label=lab,yerr=err,capsize=2,error_kw=dict(lw=0.8))
    ax3.set_xticks(x);ax3.set_xticklabels(regs,fontsize=7)
    ax3.set_xlabel("speed (m s$^{-1}$)");ax3.set_ylabel("kWh/100km")
    ax3.legend(); letter(ax3,"c")

    # d: sedan vs minivan scatter
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
    ax4.set_xlim(lims);ax4.set_ylim(lims)
    ax4.set_xlabel("sedan saving (%)");ax4.set_ylabel("minivan saving (%)")
    for i,r in enumerate(regs):
        ax4.annotate(r,(s1[i],s2[i]),textcoords="offset points",xytext=(6,3),fontsize=6.5)
    letter(ax4,"d")

    # e: ECDF of VSP (speed-matched subset)
    np.random.seed(0)
    mid_h = tk[(tk.is_av==0)&(tk.mean_speed>5)&(tk.mean_speed<15)].sample(
        min(8000, len(tk[(tk.is_av==0)&(tk.mean_speed>5)&(tk.mean_speed<15)])), random_state=0)
    mid_a = tk[(tk.is_av==1)&(tk.mean_speed>5)&(tk.mean_speed<15)].sample(
        min(8000, len(tk[(tk.is_av==1)&(tk.mean_speed>5)&(tk.mean_speed<15)])), random_state=0)
    ecdf(ax5, mid_h.mean_vsp.clip(0,15), HC)
    ecdf(ax5, mid_a.mean_vsp.clip(0,15), AVC)
    ax5.set_xlabel("VSP (kW t$^{-1}$)")
    ax5.set_ylabel("ECDF")
    ax5.text(0.95, 0.35, f"Human med.={mid_h.mean_vsp.median():.1f}\nAV med.={mid_a.mean_vsp.median():.1f}",
             transform=ax5.transAxes, fontsize=6.5, ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
    letter(ax5,"e")

    # f: high-VSP share vs speed (scatter + trend)
    np.random.seed(1)
    sub = tk.sample(min(12000, len(tk)), random_state=1)
    ax6.scatter(sub[sub.is_av==0].mean_speed, 100*sub[sub.is_av==0].frac_high_vsp,
               s=1.5, alpha=0.15, color=HC, rasterized=True)
    ax6.scatter(sub[sub.is_av==1].mean_speed, 100*sub[sub.is_av==1].frac_high_vsp,
               s=1.5, alpha=0.15, color=AVC, rasterized=True)
    # add median trend lines
    for key,c in [(0,HC),(1,AVC)]:
        sk = tk[tk.is_av==key]
        speeds = np.arange(2, 30, 2)
        meds = [sk[(sk.mean_speed>=s-1)&(sk.mean_speed<s+1)].frac_high_vsp.median()*100
                for s in speeds]
        ax6.plot(speeds, meds, color=c, lw=1.5)
    ax6.set_xlabel("mean speed (m s$^{-1}$)")
    ax6.set_ylabel("high-VSP share (%)")
    ax6.set_xlim(0, 30)
    letter(ax6,"f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT,"fig6_energy.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig3(); print("Fig 2 done (6 panels: line+ECDF+bar+heatmap+hexbin+QQ)")
    fig6(); print("Fig 4 done (6 panels: bar+grouped+bar+scatter+ECDF+scatter)")
