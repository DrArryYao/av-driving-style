"""Process NGSIM US-101 car trajectories (independent, camera-based data).

Outputs (same metric definitions as the WOMD pipeline, SG smoothing etc.):
1. ngsim_tracks.csv - per-vehicle style metrics for perception cross-check
2. ngsim_pairs.csv  - string-stability gains for human leader-follower pairs
                      (leader given explicitly by `preceding`), 30 s windows

Raw units are feet; converted to SI. Periods split at >60 s global-time gaps.
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FT = 0.3048
MIN_SEG_STEPS = 101
SG_WINDOW = 7
SG_ORDER = 2
AGGR_A = 2.0
HIGH_VSP = 20.0
PERIOD_GAP_MS = 60_000
MIN_SPEED = 3.0
MIN_PAIR_STEPS = 101
MA_SEC = 5.0
PAIR_WINDOW = 300  # 30 s sub-windows for WOMD comparability


def smooth(v):
    return savgol_filter(v, SG_WINDOW, SG_ORDER) if v.size > SG_WINDOW else v


def detrend(v, dt):
    v = smooth(v)
    half = max(int(MA_SEC / dt / 2), 2)
    ker = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(np.pad(v, half, mode="edge"), ker, mode="valid")[: v.size]
    return v - trend


def track_metrics(t, v):
    v_s = smooth(v)
    a = np.gradient(v_s, t)
    jerk = np.gradient(a, t)
    vsp = v_s * (1.1 * a + 0.132) + 0.000302 * v_s ** 3
    return {
        "duration_s": t[-1] - t[0] + 0.1,
        "mean_speed": float(np.mean(v_s)), "std_speed": float(np.std(v_s)),
        "mean_abs_accel": float(np.mean(np.abs(a))),
        "p95_abs_jerk": float(np.percentile(np.abs(jerk), 95)),
        "frac_aggr": float(np.mean(np.abs(a) > AGGR_A)),
        "frac_hard_brake": float(np.mean(a < -AGGR_A)),
        "mean_vsp": float(np.mean(vsp)),
        "frac_high_vsp": float(np.mean(vsp > HIGH_VSP)),
        "n_steps": int(v.size),
    }


def main():
    src, out_tracks, out_pairs = sys.argv[1:4]

    df = pd.read_csv(
        src, usecols=["vehicle_id", "frame_id", "global_time", "v_vel", "preceding"],
        dtype={"vehicle_id": "int32", "frame_id": "int32",
               "global_time": "int64", "v_vel": "float32", "preceding": "float32"},
    )
    df["v_ms"] = df.v_vel.values * FT

    # US-101 periods run back-to-back (7:50, 8:05, 8:20; 15 min each) so
    # time-gap detection fails. Bin global time into 15-min periods instead,
    # then guard against the rare boundary frame with a dedup pass.
    t0 = df.global_time.min()
    df["period"] = (df.global_time - t0) // 900_000
    df = df.drop_duplicates(["period", "frame_id", "vehicle_id"], keep="first")
    print("rows:", len(df), "| periods:", df.period.nunique(),
          "| vehicle episodes:", df.groupby(["period", "vehicle_id"]).ngroups)

    # per-period speed matrix: index frame_id, columns vehicle_id
    S = {p: g.pivot(index="frame_id", columns="vehicle_id", values="v_ms").sort_index()
         for p, g in df.groupby("period")}

    track_rows, pair_rows = [], []
    for (p, vid), g in df.groupby(["period", "vehicle_id"], sort=False):
        g = g.sort_values("frame_id")
        fr = g.frame_id.to_numpy()
        # split where frames are non-consecutive (vehicle re-entry or dropouts)
        eps = np.split(np.arange(fr.size), np.flatnonzero(np.diff(fr) > 2) + 1)
        mat = S[p]
        for ep in eps:
            fr_e = fr[ep]
            v = g.v_ms.to_numpy()[ep]
            prec_e = g.preceding.to_numpy()[ep]
            if v.size < MIN_SEG_STEPS:
                continue
            t = (fr_e - fr_e[0]) * 0.1
            dt = 0.1
            if np.mean(v) < MIN_SPEED and t[-1] * np.mean(v) < 15.0:
                continue  # stationary
            track_rows.append({"vehicle_id": vid, "period": p, **track_metrics(t, v)})

            # leader speed series via pivot matrix, split on leader changes
            lead_speed = np.full(v.size, np.nan)
            for pv in pd.unique(prec_e[~np.isnan(prec_e)]):
                if pv not in mat.columns:
                    continue
                sel = prec_e == pv
                lead_speed[sel] = mat.reindex(fr_e[sel])[pv].to_numpy()

            ok = ~np.isnan(lead_speed)
            splits = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
            for run in np.split(np.arange(v.size), splits):
                if run.size < MIN_PAIR_STEPS or not ok[run].all():
                    continue
                vv, ll = v[run], lead_speed[run]
                if vv.mean() < MIN_SPEED or ll.mean() < MIN_SPEED:
                    continue
                for s0 in range(0, run.size, PAIR_WINDOW):
                    seg = np.arange(s0, min(s0 + PAIR_WINDOW, run.size))
                    if seg.size < MIN_PAIR_STEPS:
                        break
                    f2 = detrend(vv[seg], dt)
                    l2 = detrend(ll[seg], dt)
                    if l2.std() < 0.05:
                        continue
                    pair_rows.append({
                        "vehicle_id": vid, "period": p,
                        "n_steps": int(seg.size),
                        "mean_speed": float(np.mean(vv[seg])),
                        "gain": float(f2.std() / l2.std()),
                        "leader_fluct_std": float(l2.std()),
                        "follower_fluct_std": float(f2.std()),
                    })

    print("tracks:", len(track_rows), "| pair windows:", len(pair_rows))
    pd.DataFrame(track_rows).to_csv(out_tracks, index=False)
    pd.DataFrame(pair_rows).to_csv(out_pairs, index=False)


if __name__ == "__main__":
    main()
