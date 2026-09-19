"""ED figures with 1:1 aspect ratio on specified panels."""
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
SQ = lambda ax: ax.set_box_aspect(1)


def ed1():
    """a adaptive (wide heatmap), b/c/d/e square."""
    df = pd.read_csv(os.path.join(HERE, "p02_gain_bias.csv"))
    fig = plt.figure(figsize=(7.1, 6.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.65)

    # a: wide heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    noises = [0.0, 0.05, 0.15]
    dists = [0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
    hm = np.full((3,6), np.nan)
    for ni,noise in enumerate(noises):
        for di,sd in enumerate(dists):
            s = df[(df.noise==noise)&(df.sigma_d==sd)&(df.L==150)&(df.g_true==1.0)]
            if len(s)>0: hm[ni,di] = s.bias_ratio.iloc[0]
    im = ax1.imshow(hm, cmap="RdYlBu_r", vmin=0.7, vmax=1.3, aspect="auto")
    ax1.set_xticks(range(6)); ax1.set_xticklabels([str(d) for d in dists], fontsize=6)
    ax1.set_yticks(range(3)); ax1.set_yticklabels([str(n) for n in noises], fontsize=6.5)
    ax1.set_xlabel("disturbance"); ax1.set_ylabel("noise")
    for ni in range(3):
        for di in range(6):
            if not np.isnan(hm[ni,di]):
                ax1.text(di,ni,f"{hm[ni,di]:.2f}",ha="center",va="center",fontsize=6.5)
    cb = fig.colorbar(im, ax=ax1, fraction=0.03)
    cb.set_label("est./true", fontsize=7); cb.ax.tick_params(labelsize=6)
    L(ax1,"a")  # adaptive

    # b: violin (square)
    ax2 = fig.add_subplot(gs[0, 2])
    data = []
    for gt in [0.5, 0.8, 1.0, 1.2]:
        s = df[(df.noise==0.15)&(df.sigma_d==0.6)&(df.L==150)&(df.g_true==gt)]
        data.append(s.bias_ratio.values)
    parts = ax2.violinplot(data, positions=range(1,5), showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#56B4E9"); pc.set_alpha(0.5)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax2.axhline(1,color="k",lw=0.8,ls="--")
    ax2.set_xticks(range(1,5)); ax2.set_xticklabels(["0.5","0.8","1.0","1.2"],fontsize=6.5)
    ax2.set_xlabel("true gain"); ax2.set_ylabel("est./true")
    SQ(ax2); L(ax2,"b")

    # c: area (square)
    ax3 = fig.add_subplot(gs[1, 0])
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s = df[(df.noise==noise)&(df.L==150)&(df.g_true==1.0)].sort_values("sigma_d")
        width = s.iqr_hi - s.iqr_lo
        ax3.fill_between(s.sigma_d, width, alpha=0.3, color=c)
        ax3.plot(s.sigma_d, width, color=c, lw=1.2, label=f"$\\sigma$={noise}")
    ax3.axvline(0.3,color="k",lw=0.8,ls=":")
    ax3.set_xscale("log")
    ax3.set_xlabel("disturbance"); ax3.set_ylabel("IQR width")
    ax3.legend(fontsize=5.5)
    SQ(ax3); L(ax3,"c")

    # d: dumbbell (square)
    ax4 = fig.add_subplot(gs[1, 1])
    for noise,c in [(0.0,"#999"),(0.05,"#56B4E9"),(0.15,"#CC79A7")]:
        s = df[(df.noise==noise)&(df.sigma_d==0.6)&(df.L==150)]
        gts = [0.5,1.0,1.2]
        vals = [s[s.g_true==g].bias_ratio.iloc[0] for g in gts if len(s[s.g_true==g])>0]
        ax4.plot(range(len(vals)), vals, "o-", color=c, ms=4, lw=1.2,
                 label=f"$\\sigma$={noise}")
    ax4.axhline(1,color="k",lw=0.8,ls="--")
    ax4.set_xticks(range(3)); ax4.set_xticklabels(["0.5","1.0","1.2"],fontsize=6.5)
    ax4.set_xlabel("true gain"); ax4.set_ylabel("est./true")
    ax4.legend(fontsize=5.5)
    SQ(ax4); L(ax4,"d")

    # e: hexbin (square)
    ax5 = fig.add_subplot(gs[1, 2])
    sub = df[(df.noise==0.15)&(df.g_true==1.0)]
    ax5.hexbin(sub.sigma_d, sub.bias_ratio, gridsize=15, cmap="YlOrRd", mincnt=3)
    ax5.axhline(1,color="k",lw=0.8,ls="--")
    ax5.set_xscale("log")
    ax5.set_xlabel("disturbance"); ax5.set_ylabel("est./true")
    SQ(ax5); L(ax5,"e")

    fig.savefig(os.path.join(OUT,"ed1_bias.pdf")); plt.close(fig)


def ed2():
    """All 1:1."""
    df = pd.read_csv(os.path.join(HERE,"platoon_probe.csv"))
    traces = pd.read_csv(os.path.join(HERE,"sim_speed_traces.csv"))
    energy = pd.read_csv(os.path.join(HERE,"sim_per_vehicle_energy.csv"))
    vehicles = [5,15,25,35,45,55]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 6.5))
    (ax1,ax2,ax3),(ax4,ax5,ax6) = axes

    # a: scatter
    for cfg,lab,c in [("human","human",HC),("av","AV",AVC)]:
        s = df[df.config==cfg].sort_values("position")
        ax1.scatter(s.position, s.gain, color=c, s=25, zorder=3)
        z = np.polyfit(s.position, s.gain, 2)
        xs = np.linspace(s.position.min(), s.position.max(), 50)
        ax1.plot(xs, np.polyval(z, xs), color=c, lw=1.5, ls="--")
    ax1.axhline(1,color="k",lw=0.8,ls="--")
    ax1.set_xlabel("position"); ax1.set_ylabel("chain gain")
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
    ax2.set_title("0% AV", fontsize=6.5, fontweight="bold")
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
    ax3.set_title("50% AV", fontsize=6.5, fontweight="bold")
    fig.colorbar(im3, ax=ax3, fraction=0.04).ax.tick_params(labelsize=5.5)
    SQ(ax3); L(ax3,"c")

    # d: dumbbell
    for pen,lab,c in [(0,"0%",HC),(50,"50%",AVC)]:
        stds = [traces[(traces.pen==pen)&(traces.veh==vi)].v.std() for vi in vehicles]
        ax4.plot(range(len(stds)), stds, "o-", color=c, ms=4, lw=1.2, label=lab)
    ax4.set_xticks(range(len(vehicles)))
    ax4.set_xticklabels([str(v) for v in vehicles], fontsize=6)
    ax4.set_xlabel("position"); ax4.set_ylabel("speed s.d.")
    ax4.legend()
    SQ(ax4); L(ax4,"d")

    # e: area (KDE)
    from scipy.stats import gaussian_kde
    e0 = energy[energy.pen==0].e_kwh.clip(0,50)
    e50 = energy[energy.pen==50].e_kwh.clip(0,50)
    x = np.linspace(0, 50, 100)
    ax5.fill_between(x, gaussian_kde(e0)(x), alpha=0.4, color=HC, label="0%")
    ax5.fill_between(x, gaussian_kde(e50)(x), alpha=0.4, color=AVC, label="50%")
    ax5.set_xlabel("kWh/100km"); ax5.set_ylabel("density")
    ax5.legend()
    SQ(ax5); L(ax5,"e")

    # f: bar
    for cfg,lab,c in [("human","human",HC),("av","AV",AVC)]:
        s = df[df.config==cfg].sort_values("position")
        cum = np.cumprod(s.gain.values)
        ax6.bar(range(1,len(cum)+1), cum, color=c, alpha=0.7, width=0.6, label=lab)
    ax6.axhline(1,color="k",lw=0.8,ls="--")
    ax6.set_xlabel("position"); ax6.set_ylabel("cumulative gain")
    ax6.legend()
    SQ(ax6); L(ax6,"f")

    fig.subplots_adjust(hspace=0.45, wspace=0.65)
    fig.savefig(os.path.join(OUT,"ed2_controller.pdf")); plt.close(fig)


def ed3():
    """a adaptive (wide heatmap), b/c/d/e square."""
    df = pd.read_csv(os.path.join(HERE,"p12_sensitivity.csv"))
    base = 41.09
    df["sav"] = 100*(1-df.e/base)
    sub50 = df[df.n_av==30]
    jcs = [1.0,1.5,2.0]; Ts = [1.1,1.4,1.7]

    fig = plt.figure(figsize=(7.1, 6.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.65)

    # a: wide heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    hm = np.full((3,3),np.nan)
    for ji,jc in enumerate(jcs):
        for ti,T in enumerate(Ts):
            s = sub50[(sub50.jerk_cap==jc)&(sub50.T_av==T)&(sub50.t_close==0.3)].sav
            if len(s)>0: hm[ji,ti] = s.mean()
    im = ax1.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(3)); ax1.set_xticklabels([str(t) for t in Ts], fontsize=6.5)
    ax1.set_yticks(range(3)); ax1.set_yticklabels([str(j) for j in jcs], fontsize=6.5)
    ax1.set_xlabel("headway T (s)"); ax1.set_ylabel("jerk cap")
    for ji in range(3):
        for ti in range(3):
            if not np.isnan(hm[ji,ti]):
                ax1.text(ti,ji,f"{hm[ji,ti]:.1f}%",ha="center",va="center",fontsize=7)
    cb = fig.colorbar(im, ax=ax1, fraction=0.03)
    cb.set_label("saving (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
    L(ax1,"a")  # adaptive

    # b: violin (square)
    ax2 = fig.add_subplot(gs[0, 2])
    data = []; labels = []
    for n_av,lab in [(12,"20%"),(30,"50%"),(60,"100%")]:
        sub = df[df.n_av==n_av].groupby(["jerk_cap","t_close","T_av"]).sav.mean()
        data.append(sub.values); labels.append(lab)
    parts = ax2.violinplot(data, positions=[1,2,3], showmedians=True, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor("#56B4E9"); pc.set_alpha(0.5)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax2.set_xticks([1,2,3]); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("saving (%)")
    SQ(ax2); L(ax2,"b")

    # c: dumbbell (square)
    ax3 = fig.add_subplot(gs[1, 0])
    for i,jc in enumerate(jcs):
        lo = sub50[(sub50.jerk_cap==jc)&(sub50.t_close==0)].sav.median()
        hi = sub50[(sub50.jerk_cap==jc)&(sub50.t_close==0.55)].sav.median()
        ax3.plot([lo,hi],[i,i],color="#999",lw=2)
        ax3.scatter(lo,i,color=HC,s=35,zorder=3)
        ax3.scatter(hi,i,color=AVC,s=35,zorder=3)
    ax3.set_yticks(range(3)); ax3.set_yticklabels([str(j) for j in jcs], fontsize=6.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("saving at 50% (%)"); ax3.set_ylabel("jerk cap")
    SQ(ax3); L(ax3,"c")

    # d: box (square)
    ax4 = fig.add_subplot(gs[1, 1])
    data_d = [sub50[sub50.t_close==tc].sav.values for tc in [0.0,0.3,0.55]]
    bp = ax4.boxplot(data_d, positions=[1,2,3], widths=0.5, patch_artist=True,
                     medianprops=dict(color="k", lw=1))
    for patch in bp["boxes"]:
        patch.set_facecolor("#E69F00"); patch.set_alpha(0.5)
    ax4.set_xticks([1,2,3]); ax4.set_xticklabels(["0","0.3","0.55"], fontsize=6.5)
    ax4.set_xlabel("anticipation"); ax4.set_ylabel("saving at 50% (%)")
    SQ(ax4); L(ax4,"d")

    # e: hexbin (square)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hexbin(100*df.n_av/60, df.sav, gridsize=15, cmap="YlOrRd", mincnt=5)
    ax5.set_xlabel("penetration (%)"); ax5.set_ylabel("saving (%)")
    SQ(ax5); L(ax5,"e")

    fig.savefig(os.path.join(OUT,"ed3_sensitivity.pdf")); plt.close(fig)


def ed4():
    """a adaptive (wide heatmap), b/c/d/e square."""
    tk = pd.read_csv("E:/av_style_data/tracks_full.csv",
                     usecols=["is_av","mean_speed","p95_abs_jerk","frac_aggr"])
    ng = pd.read_csv("E:/av_style_data/ngsim_tracks.csv",
                     usecols=["mean_speed","p95_abs_jerk","frac_aggr"])
    w_h = tk[(tk.is_av==0)&(tk.mean_speed>10)]
    w_a = tk[(tk.is_av==1)&(tk.mean_speed>10)]
    ng_h = ng[ng.mean_speed>10]
    colors = [(w_h,HC,"Human (perc.)"),(ng_h,NC,"NGSIM"),(w_a,AVC,"AV")]

    fig = plt.figure(figsize=(7.1, 6.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.65)

    # a: wide heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    bins = [(10,15),(15,20),(20,25),(25,35)]
    hm = np.zeros((3,4))
    for si,(s,_,_) in enumerate(colors):
        for bi,(lo,hi) in enumerate(bins):
            sub = s[(s.mean_speed>=lo)&(s.mean_speed<hi)]
            hm[si,bi] = sub.p95_abs_jerk.median() if len(sub)>10 else 0
    im = ax1.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"{lo}-{hi}" for lo,hi in bins], fontsize=6)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([c[2] for c in colors], fontsize=6)
    ax1.set_xlabel("speed (m s$^{-1}$)"); ax1.set_ylabel("median jerk")
    for si in range(3):
        for bi in range(4):
            ax1.text(bi,si,f"{hm[si,bi]:.1f}",ha="center",va="center",
                     fontsize=7, color="white" if hm[si,bi]>hm.max()*0.5 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.03).ax.tick_params(labelsize=6)
    L(ax1,"a")  # adaptive

    # b: ridgeline (square)
    ax2 = fig.add_subplot(gs[0, 2])
    from scipy.stats import gaussian_kde
    np.random.seed(0)
    for ri,(lo,hi) in enumerate(bins):
        sub_h = w_h[(w_h.mean_speed>=lo)&(w_h.mean_speed<hi)].p95_abs_jerk.sample(
            min(1500, len(w_h[(w_h.mean_speed>=lo)&(w_h.mean_speed<hi)])), random_state=ri).clip(0,40)
        sub_a = w_a[(w_a.mean_speed>=lo)&(w_a.mean_speed<hi)].p95_abs_jerk.sample(
            min(1500, len(w_a[(w_a.mean_speed>=lo)&(w_a.mean_speed<hi)])), random_state=ri).clip(0,40)
        x = np.linspace(0, 40, 80)
        kh = gaussian_kde(sub_h)(x); ka = gaussian_kde(sub_a)(x)
        sc = 0.8/max(kh.max(), ka.max(), 1e-9)
        off = len(bins)-1-ri
        ax2.fill_between(x, off, off+kh*sc, alpha=0.35, color=HC, lw=0)
        ax2.fill_between(x, off, off+ka*sc, alpha=0.35, color=AVC, lw=0)
        ax2.text(-2, off, f"{lo}-{hi}", ha="right", va="center", fontsize=5)
    ax2.set_ylim(-0.3, len(bins)+0.2)
    ax2.set_yticks([]); ax2.set_xlabel("95th p. jerk")
    SQ(ax2); L(ax2,"b")

    # c: scatter (square)
    ax3 = fig.add_subplot(gs[1, 0])
    for s,c,lab in colors:
        sub = s.sample(min(2000,len(s)),random_state=0)
        ax3.scatter(sub.mean_speed, sub.p95_abs_jerk.clip(0,40),
                   s=1.5, alpha=0.2, color=c, rasterized=True)
    ax3.set_xlabel("speed (m s$^{-1}$)"); ax3.set_ylabel("95th p. jerk")
    ax3.set_ylim(0,40)
    SQ(ax3); L(ax3,"c")

    # d: dumbbell (square)
    ax4 = fig.add_subplot(gs[1, 1])
    for mi,(m,lab) in enumerate([("p95_abs_jerk","jerk"),("frac_aggr","aggr.")]):
        vals = [s[m].median() for s,_,_ in colors]
        norm = vals[0] if vals[0]>0 else 1
        for si,v in enumerate(vals):
            ax4.barh(mi*3+si, v/norm, color=[HC,NC,AVC][si], height=0.7)
            ax4.text(v/norm+0.02, mi*3+si, f"{v:.2f}", va="center", fontsize=5.5)
    ax4.set_yticks([0,1,2,3,4])
    ax4.set_yticklabels(["jerk\nP","jerk\nN","jerk\nAV","aggr\nP","aggr\nN"], fontsize=5)
    ax4.axvline(1,color=AVC,lw=0.8,ls="--")
    ax4.set_xlabel("normalized to P")
    SQ(ax4); L(ax4,"d")

    # e: area ECDF (square)
    ax5 = fig.add_subplot(gs[1, 2])
    for s,c,lab in colors:
        v = np.sort(s.frac_aggr.dropna())
        ax5.fill_between(v, 0, np.arange(1,v.size+1)/v.size, alpha=0.3, color=c)
        ax5.plot(v, np.arange(1,v.size+1)/v.size, color=c, lw=1.1)
    ax5.set_xlabel("aggressive share"); ax5.set_ylabel("ECDF")
    handles = [Line2D([0],[0],color=c,lw=1.2,label=l) for _,c,l in colors]
    ax5.legend(handles=handles, fontsize=5.5)
    SQ(ax5); L(ax5,"e")

    fig.savefig(os.path.join(OUT,"ed4_validation.pdf")); plt.close(fig)


if __name__ == "__main__":
    ed1(); print("ED1: a adaptive, b/c/d/e 1:1")
    ed2(); print("ED2: all 1:1")
    ed3(); print("ED3: a adaptive, b/c/d/e 1:1")
    ed4(); print("ED4: a adaptive, b/c/d/e 1:1")
