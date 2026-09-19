"""P1-3: cluster bootstrap by collection shard for headline paired contrasts.

Scenarios share one operator/cities/periods, so we resample SHARDS (the
930 tfrecord files) rather than scenarios to obtain defensible CIs, and
report the design effect relative to scenario-level resampling.
"""
import numpy as np
import pandas as pd

tk = pd.read_csv("${DATA_DIR}/tracks_full.csv",
                 usecols=["scenario_id", "file", "is_av", "p95_abs_jerk",
                          "frac_aggr", "frac_hard_brake", "frac_high_vsp", "mean_speed"])
p = tk.groupby(["scenario_id", "file"]).apply(
    lambda g: pd.Series({
        "is_av": int(g.is_av.max()),
        **{m: g.loc[g.is_av == 0, m].mean() for m in
           ["p95_abs_jerk", "frac_aggr", "frac_hard_brake", "frac_high_vsp", "mean_speed"]},
        **{"av_" + m: g.loc[g.is_av == 1, m].mean() for m in
           ["p95_abs_jerk", "frac_aggr", "frac_hard_brake", "frac_high_vsp", "mean_speed"]},
        "n_hdv": int((g.is_av == 0).sum()),
    }), include_groups=False).reset_index()

p = p[(p.is_av == 1) & (p.n_hdv > 0)]
shards = p.file.unique()
print(f"paired scenarios: {len(p)}  | shards: {len(shards)}")

rng = np.random.default_rng(0)
B = 2000
for m in ["p95_abs_jerk", "frac_aggr", "frac_hard_brake", "frac_high_vsp"]:
    d = (p["av_" + m] - p[m]).values
    # scenario-level bootstrap
    sc = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(500)])
    # shard-cluster bootstrap
    shard_groups = [np.flatnonzero((p.file == s).to_numpy()) for s in shards]
    cb = []
    for _ in range(B):
        pick = rng.integers(0, len(shards), len(shards))
        idx = np.concatenate([shard_groups[i] for i in pick])
        cb.append(d[idx].mean())
    cb = np.array(cb)
    de = (np.var(cb) / np.var(sc)) if np.var(sc) > 0 else np.nan
    print(f"{m:18s} diff={d.mean():+.4f}  cluster 95% CI "
          f"[{np.percentile(cb,2.5):+.4f}, {np.percentile(cb,97.5):+.4f}]  "
          f"design effect={de:.2f}  eff.n≈{len(p)/de:,.0f}")
