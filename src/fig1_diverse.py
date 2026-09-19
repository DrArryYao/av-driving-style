from make_figures_header import *

def fig1():
    tk = pd.read_csv("E:/av_style_data/tracks_full2.csv",
                     usecols=["is_av","mean_speed","p95_abs_jerk","frac_aggr",
                              "frac_hard_brake","mean_vsp","frac_high_vsp",
                              "mean_abs_accel"])
    tk["regime"] = pd.cut(tk.mean_speed, [0,5,10,15,100],
                          labels=["0-5","5-10","10-15",">15"])
    p = pd.read_csv("E:/av_style_data/results_full/paired_scenarios.csv")
    p["regime"] = pd.cut(p.av_speed, [0,5,10,15,100],
                         labels=["0-5","5-10","10-15",">15"])
    regs = ["0-5","5-10","10-15",">15"]

    fig = plt.figure(figsize=(7.1, 6.2))
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.45,
                           height_ratios=[1,1,1])
    np.random.seed(0)
    sub_h = tk[tk.is_av==0].sample(min(5000,len(tk[tk.is_av==0])),random_state=0)
    sub_a = tk[tk.is_av==1].sample(min(5000,len(tk[tk.is_av==1])),random_state=0)

    # a: violin
    ax1 = fig.add_subplot(gs[0,0])
    parts = ax1.violinplot([sub_h.p95_abs_jerk.clip(0,50),
                            sub_a.p95_abs_jerk.clip(0,50)],
                           positions=[1,2], showmedians=True, widths=0.7)
    for pc,c in zip(parts["bodies"],[HC,AVC]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    for k in ["cmins","cmaxes","cbars","cmedians"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    ax1.set_xticks([1,2]); ax1.set_xticklabels(["Human","AV"],fontsize=7)
    ax1.set_ylabel("95th p.|jerk| (m s$^{-3}$)")
    ax1.set_ylim(0,35)
    letter(ax1,"a")

    # b: dumbbell
    ax2 = fig.add_subplot(gs[0,1:])
    y_pos = np.arange(len(regs))
    for i,r in enumerate(regs):
        sub = p[p.regime==r]
        if len(sub)<30: continue
        dh = sub.hdv_p95_abs_jerk.median()
        da = sub.av_p95_abs_jerk.median()
        ax2.plot([dh,da],[i,i],color="#999",lw=1.5,zorder=1)
        ax2.scatter(dh,i,color=HC,s=40,zorder=3,
                    label="Human" if i==0 else "")
        ax2.scatter(da,i,color=AVC,s=40,zorder=3,
                    label="AV" if i==0 else "")
        ax2.annotate(f"$-${100*(1-da/dh):.0f}%",
                     ((dh+da)/2,i+0.25),fontsize=6.5,ha="center",color="#333")
    ax2.set_yticks(y_pos); ax2.set_yticklabels(regs,fontsize=7)
    ax2.set_xlabel("median 95th-p. jerk (m s$^{-3}$)")
    ax2.invert_yaxis()
    ax2.legend(loc="lower left")
    letter(ax2,"b")

    # c: heatmap
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
    ax3.set_xticks(range(len(regs)));ax3.set_xticklabels(regs,fontsize=7)
    ax3.set_yticks(range(len(metrics)));ax3.set_yticklabels(mlabels,fontsize=7)
    for mi in range(len(metrics)):
        for ri in range(len(regs)):
            if not np.isnan(hm[mi,ri]):
                ax3.text(ri,mi,f"{hm[mi,ri]:.0f}",ha="center",va="center",
                         fontsize=6.5,color="k")
    # horizontal colorbar below heatmap (avoids overlap with panel d)
    cb = fig.colorbar(im,ax=ax3,fraction=0.035,pad=0.08,orientation="horizontal",
                      anchor=(0.5,1.0),shrink=0.6)
    cb.set_label("AV-human (%)",fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax3.set_xlabel("speed regime (m s$^{-1}$)")
    letter(ax3,"c")

    # d: bar
    ax4 = fig.add_subplot(gs[1,2])
    x = np.arange(len(regs))
    for key,lab,c,off in [(0,"Human",HC,-0.18),(1,"AV",AVC,0.18)]:
        vals = [tk[(tk.regime==r)&(tk.is_av==key)].frac_aggr.median()
                for r in regs]
        ax4.bar(x+off,[100*v for v in vals],width=0.34,color=c,label=lab)
    ax4.set_xticks(x);ax4.set_xticklabels(regs,fontsize=6.5,rotation=30)
    ax4.set_ylabel("aggressive share (%)")
    ax4.legend(fontsize=6)
    letter(ax4,"d")

    # e: ECDF jerk
    ax5 = fig.add_subplot(gs[2,0])
    ecdf(ax5,sub_h.p95_abs_jerk,HC)
    ecdf(ax5,sub_a.p95_abs_jerk,AVC)
    ax5.set_xscale("log")
    ax5.set_xlabel("95th p.|jerk| (m s$^{-3}$)")
    ax5.set_ylabel("ECDF")
    letter(ax5,"e")

    # f: ECDF VSP
    ax6 = fig.add_subplot(gs[2,1])
    ecdf(ax6,sub_h.mean_vsp,HC)
    ecdf(ax6,sub_a.mean_vsp,AVC)
    ax6.set_xscale("log")
    ax6.set_xlabel("mean VSP (kW t$^{-1}$)")
    ax6.set_ylabel("ECDF")
    letter(ax6,"f")

    # g: forest
    ax7 = fig.add_subplot(gs[2,2])
    mf = ["p95_abs_jerk","frac_aggr","frac_hard_brake","frac_high_vsp"]
    fl = ["jerk","aggr.","brake","VSP"]
    for i,m in enumerate(mf):
        d = (p[f"av_{m}"]-p[f"hdv_{m}"])/p[f"hdv_{m}"].replace(0,np.nan)
        d = d.dropna()
        med = d.median()
        ci = np.percentile([np.median(np.random.default_rng(s).choice(
            d.values,d.size)) for s in range(500)],[2.5,97.5])
        ax7.errorbar([med],[i],xerr=[[med-ci[0]],[ci[1]-med]],
                     fmt="o",color=AVC,capsize=3,ms=5)
    ax7.axvline(0,color="k",lw=0.8,ls="--")
    ax7.set_yticks(range(len(mf)));ax7.set_yticklabels(fl,fontsize=7)
    ax7.set_xlabel("rel. diff (%)")
    ax7.set_xlim(-100,10)
    ax7.invert_yaxis()
    letter(ax7,"g")

    handles = [Line2D([0],[0],color=HC,lw=1.3,label="Human vehicles"),
               Line2D([0],[0],color=AVC,lw=1.3,label="Waymo AV")]
    ax5.legend(handles=handles,fontsize=6)

    fig.savefig(os.path.join(OUT,"fig1_ecdf.pdf"))
    plt.close(fig)
