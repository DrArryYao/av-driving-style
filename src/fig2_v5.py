"""Fig 2 redesigned: 6 panels in 2x3, panel a 2:1, rest 1:1."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SQ = lambda ax: ax.set_box_aspect(1)
WIDE = lambda ax: ax.set_box_aspect(0.5)  # 2:1 (width:height)


def fig2():
    wp = pd.read_csv("E:/av_style_data/pairs_full.csv",
                     usecols=["follower_is_av","mean_speed","gain","leader_fluct_std"])
    ng = pd.read_csv("E:/av_style_data/ngsim_pairs.csv",
                     usecols=["mean_speed","gain","leader_fluct_std"])
    wf = wp[wp.mean_speed>10]; nf = ng[ng.mean_speed>10]
    strata = [(0.05,0.15),(0.15,0.3),(0.3,0.6),(0.6,3.0)]
    mids = [np.mean(s) for s in strata]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.5))
    fig.subplots_adjust(hspace=0.15, wspace=0.55)
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: gain vs disturbance (2:1 wide)
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
    WIDE(ax1); L(ax1,"a")

    # b: violin (1:1)
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

    # c: dumbbell (1:1)
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

    # d: heatmap (1:1)
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
    ax4.set_xlabel("disturbance"); ax4.set_ylabel("speed")
    for xi in range(4):
        for yi in range(4):
            if not np.isnan(D[yi,xi]):
                ax4.text(xi,yi,f"{D[yi,xi]:.2f}",ha="center",va="center",fontsize=6.5)
    cb = fig.colorbar(im,ax=ax4,fraction=0.04)
    cb.set_label("AV$-$human",fontsize=7); cb.ax.tick_params(labelsize=6)
    SQ(ax4); L(ax4,"d")

    # e: hexbin (1:1)
    np.random.seed(42)
    sub = wf.sample(min(12000,len(wf)),random_state=42)
    ax5.hexbin(sub[sub.follower_is_av==0].mean_speed, sub[sub.follower_is_av==0].gain.clip(0,3),
               gridsize=20, cmap="Blues", mincnt=5, alpha=0.7)
    ax5.hexbin(sub[sub.follower_is_av==1].mean_speed, sub[sub.follower_is_av==1].gain.clip(0,3),
               gridsize=20, cmap="Oranges", mincnt=5, alpha=0.7)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xlabel("speed (m s$^{-1}$)"); ax5.set_ylabel("gain")
    SQ(ax5); L(ax5,"e")

    # f: QQ plot (1:1)
    av_q = np.sort(wf[(wf.follower_is_av==1)&(wf.leader_fluct_std>=0.3)].gain.values)
    hu_q = np.sort(wf[(wf.follower_is_av==0)&(wf.leader_fluct_std>=0.3)].gain.values)
    n = min(3000, len(av_q), len(hu_q))
    av_s = av_q[np.linspace(0,len(av_q)-1,n).astype(int)]
    hu_s = hu_q[np.linspace(0,len(hu_q)-1,n).astype(int)]
    ax6.scatter(hu_s, av_s, s=2, alpha=0.3, color="#666")
    ax6.plot([0,3],[0,3],"k--",lw=0.8,label="identity")
    ax6.set_xlim(0,3); ax6.set_ylim(0,3)
    ax6.set_xlabel("human gain quantile"); ax6.set_ylabel("AV gain quantile")
    ax6.legend(fontsize=5.5)
    SQ(ax6); L(ax6,"f")

    fig.savefig(os.path.join(OUT,"fig3_string.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig2(); print("Fig 2: 6 panels (a=2:1 wide, b-f=1:1)")
