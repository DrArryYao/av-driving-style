"""Fig 2 expanded to 9 panels (3x3): original 6 + 3 ACC comparison panels."""
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
ACC_C = "#CC79A7"  # commercial ACC (pink)
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
    "legend.fontsize": 5.5, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "lines.linewidth": 1.2,
})
L = lambda ax, s: ax.text(-0.20, 1.15, s, transform=ax.transAxes,
                          fontsize=9, fontweight="bold", va="top", ha="left")


def fig2_9panel():
    wp = pd.read_csv("E:/av_style_data/pairs_full.csv",
                     usecols=["follower_is_av","mean_speed","gain","leader_fluct_std"])
    ng = pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                     usecols=["mean_speed","gain","leader_fluct_std"])
    jrc = pd.read_csv(os.path.join(HERE, "E:/av_style_data/jrc_acc/jrc_acc_gains.csv"))
    wf = wp[wp.mean_speed>10]; nf = ng[ng.mean_speed>10]
    strata = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,3.0)]
    mids = [np.mean(s) for s in strata]

    fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.5))
    fig.subplots_adjust(hspace=0.40, wspace=0.50)
    (ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9) = axes

    # a: gain vs disturbance (4 sources including ACC)
    for series,name,c in [(wf[wf.follower_is_av==1],"L4 AV",AVC),
                          (wf[wf.follower_is_av==0],"Human (perc.)",HC),
                          (nf,"Human (NGSIM)",NC),
                          (jrc,"Comm. ACC",ACC_C)]:
        med,q1,q3 = [],[],[]
        for lo,hi in strata:
            s = series[(series.leader_fluct_std>=lo)&(series.leader_fluct_std<hi)]
            if len(s)>=10:
                med.append(s.gain.median());q1.append(s.gain.quantile(.25));q3.append(s.gain.quantile(.75))
            else:
                med.append(np.nan);q1.append(np.nan);q3.append(np.nan)
        ax1.plot(mids,med,"o-",color=c,label=name,ms=3.5,lw=1.2)
        ax1.fill_between(mids,q1,q3,color=c,alpha=0.15)
    ax1.axhline(1,color="k",lw=0.8,ls="--"); ax1.set_xscale("log")
    ax1.set_xlabel("disturbance (m s$^{-1}$)"); ax1.set_ylabel("gain")
    ax1.legend(fontsize=5); L(ax1,"a")

    # b: violin — gain by group at >=0.3
    av_g = wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain
    hu_g = wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.sample(3000,random_state=0)
    ng_g = nf[nf.leader_fluct_std>=0.3].gain
    acc_g = jrc[jrc.leader_fluct_std>=0.05].gain  # ACC has fewer at >=0.3
    parts = ax2.violinplot([av_g.clip(0,3),hu_g.clip(0,3),ng_g.clip(0,3),acc_g.clip(0,3)],
                           positions=[1,2,3,4],showmedians=True,widths=0.7)
    for pc,c in zip(parts["bodies"],[AVC,HC,NC,ACC_C]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["L4\nAV","Human\n(perceived)","NGSIM\n(camera)","Comm.\nACC"],fontsize=5.5)
    ax2.set_ylabel("gain"); L(ax2,"b")

    # c: dumbbell — composition
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

    # d: heatmap
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

    # e: hexbin
    np.random.seed(42)
    sub = wf.sample(min(12000,len(wf)),random_state=42)
    ax5.hexbin(sub[sub.follower_is_av==0].mean_speed, sub[sub.follower_is_av==0].gain.clip(0,3),
               gridsize=20,cmap="Blues",mincnt=5,alpha=0.7)
    ax5.hexbin(sub[sub.follower_is_av==1].mean_speed, sub[sub.follower_is_av==1].gain.clip(0,3),
               gridsize=20,cmap="Oranges",mincnt=5,alpha=0.7)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xlabel("speed (m s$^{-1}$)"); ax5.set_ylabel("gain"); L(ax5,"e")

    # f: QQ plot
    av_q = np.sort(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain.values)
    hu_q = np.sort(wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.values)
    n = min(3000,len(av_q),len(hu_q))
    ax6.scatter(hu_q[np.linspace(0,len(hu_q)-1,n).astype(int)],
                av_q[np.linspace(0,len(av_q)-1,n).astype(int)],s=2,alpha=0.3,color="#666")
    ax6.plot([0,3],[0,3],"k--",lw=0.8)
    ax6.set_xlim(0,3); ax6.set_ylim(0,3)
    ax6.set_xlabel("human gain quantile"); ax6.set_ylabel("AV gain quantile"); L(ax6,"f")

    # g: NEW — ACC vs L4 vs Human gain ECDF
    for series,lab,c in [(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain,"L4 AV",AVC),
                          (nf[nf.leader_fluct_std>=0.3].gain,"Human (NGSIM)",NC),
                          (jrc.gain,"Comm. ACC",ACC_C)]:
        v = np.sort(series.clip(0,3))
        ax7.plot(v,np.arange(1,v.size+1)/v.size,color=c,lw=1.1,label=lab)
    ax7.axvline(1,color="k",lw=0.8,ls="--")
    ax7.set_xlabel("gain"); ax7.set_ylabel("ECDF"); ax7.legend(fontsize=5)
    L(ax7,"g")

    # h: NEW — automation quality spectrum bar
    cats = ["Human\n(NGSIM)","Comm.\nACC","L4\nAV"]
    gains = [nf[nf.leader_fluct_std>=0.3].gain.median(),
             jrc.gain.median(),
             wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain.median()]
    fracs = [(nf[nf.leader_fluct_std>=0.3].gain>1).mean(),
             (jrc.gain>1).mean(),
             (wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain>1).mean()]
    x = np.arange(3)
    bars = ax8.bar(x,gains,color=[NC,ACC_C,AVC],width=0.65)
    for i,(g,f) in enumerate(zip(gains,fracs)):
        ax8.text(i,g+0.02,f"{g:.2f}",ha="center",fontsize=7,fontweight="bold")
        ax8.text(i,g/2,f"{f:.0%}\nunstable",ha="center",fontsize=5,color="white")
    ax8.axhline(1,color="k",lw=0.8,ls="--")
    ax8.set_xticks(x); ax8.set_xticklabels(cats,fontsize=6)
    ax8.set_ylabel("median gain"); ax8.set_ylim(0,1.4)
    L(ax8,"h")

    # i: NEW — gain vs speed for ACC
    ax9.scatter(jrc.mean_speed,jrc.gain.clip(0,3),s=8,alpha=0.4,color=ACC_C,edgecolors="none",label="Comm. ACC")
    # add median trend
    speeds = np.arange(20,40,3)
    meds = [jrc[(jrc.mean_speed>=s-1.5)&(jrc.mean_speed<s+1.5)].gain.median()
            for s in speeds]
    ax9.plot(speeds,meds,color=ACC_C,lw=1.5,label="Comm. ACC")
    ax9.axhline(1,color="k",lw=0.8,ls="--")
    # add L4 for comparison
    av_wf = wf[wf.follower_is_av==1]
    av_speeds = np.arange(10,30,3)
    av_meds = [av_wf[(av_wf.mean_speed>=s-1.5)&(av_wf.mean_speed<s+1.5)].gain.median()
               for s in av_speeds]
    ax9.plot(av_speeds,av_meds,color=AVC,lw=1.5,label="L4 AV")
    ax9.set_xlabel("speed (m s$^{-1}$)"); ax9.set_ylabel("gain")
    ax9.legend(fontsize=5.5)
    L(ax9,"i")

    fig.savefig(os.path.join(OUT,"fig3_string.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig2_9panel(); print("Fig 2: 9 panels done")
