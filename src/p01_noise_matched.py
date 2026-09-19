"""P0-1: noise-matched re-estimation of follower-leader gains (Fig. 5b defense).

The referee's confound: in human-behind-AV pairs the leader series is the
AV's own (clean) speed; in human-behind-human pairs the leader is perceived
(Kalman-smoothed, correlated noise). Gains are ratios of fluctuation std,
so unequal leader noise makes pair types incomparable.

Here we inject AR(1) noise into AV-leader series, calibrated to the pooled
residual statistics of perceived HDV speeds (sigma and lag-1 autocorrelation
of v - savitzky-golay(v)), then re-estimate gains for all pair types with
identical processing.

Usage: python p01_noise_matched.py <n_shards> <out.csv>
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from womd_parsing import iter_scenarios
from string_stability import (build_tracks, find_pairs, contiguous_runs,
                              MIN_PAIR_STEPS)

SG_W, SG_O = 7, 2


def sg(v):
    return savgol_filter(v, SG_W, SG_O) if v.size > SG_W else v


def detrend(v):
    half = 25
    ker = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(np.pad(v, half, mode="edge"), ker, mode="valid")[: v.size]
    return v - trend


def speed_series(tr, n):
    v = np.hypot(tr["vx"], tr["vy"])[:n]
    ok = tr["valid"][:n]
    return v, ok


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = sys.argv[2] if len(sys.argv) > 2 else "p01_pairs.csv"
    files = sorted(glob.glob("${DATA_DIR}/tfrecords/*"))[:n_shards]

    # pass 1: residual statistics of perceived HDV speeds and AV ego speeds
    hdv_res, av_res = [], []
    for path in files[: max(20, n_shards // 6)]:
        for sc in iter_scenarios(path):
            t, tracks = build_tracks(sc)
            if not tracks:
                continue
            sdc = sc.tracks[sc.sdc_track_index].id
            for tid, tr in tracks.items():
                v, ok = speed_series(tr, len(t))
                if ok.sum() < 100 or v.mean() < 3:
                    continue
                r = v - sg(v)
                (av_res if tid == sdc else hdv_res).append(r.std())
    sig_hdv = float(np.median(hdv_res))
    sig_av = float(np.median(av_res))
    sig_inj = float(np.sqrt(max(sig_hdv ** 2 - sig_av ** 2, 0.0)))
    print(f"residual sigma: HDV {sig_hdv:.3f}  AV {sig_av:.3f}  inject {sig_inj:.3f} m/s")

    # lag-1 autocorrelation of HDV residuals (AR(1) rho)
    rhos = []
    for path in files[:10]:
        for sc in iter_scenarios(path):
            t, tracks = build_tracks(sc)
            sdc = sc.tracks[sc.sdc_track_index].id if sc.tracks else None
            for tid, tr in tracks.items():
                if tid == sdc:
                    continue
                v, ok = speed_series(tr, len(t))
                if ok.sum() < 100:
                    continue
                r = (v - sg(v))[5:-5]
                if r.size > 50 and r.std() > 1e-6:
                    rhos.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    rho = float(np.median(rhos))
    print(f"HDV residual lag-1 autocorr (AR rho): {rho:.3f}")

    rng = np.random.default_rng(7)
    rows = []
    for path in files:
        for sc in iter_scenarios(path):
            t, tracks = build_tracks(sc)
            if len(tracks) < 2:
                continue
            sdc = sc.tracks[sc.sdc_track_index].id
            spd = {i: np.hypot(tracks[i]["vx"], tracks[i]["vy"]) for i in tracks}
            for (fid, lid), steps in find_pairs(t, tracks).items():
                for run in contiguous_runs(steps, MIN_PAIR_STEPS):
                    fv = spd[fid][run]
                    lv = spd[lid][run]
                    if fv.mean() < 3 or lv.mean() < 3:
                        continue
                    lf = detrend(lv)
                    if lf.std() < 0.05:
                        continue
                    g_raw = detrend(fv).std() / lf.std()
                    # noise-matched leader: inject AR(1) if leader is AV
                    if lid == sdc and sig_inj > 0:
                        e = rng.normal(0, sig_inj * np.sqrt(1 - rho ** 2), lv.size)
                        ar = np.empty_like(e)
                        ar[0] = e[0]
                        for k in range(1, ar.size):
                            ar[k] = rho * ar[k - 1] + e[k]
                        lv_nm = lv + ar
                        g_nm = detrend(fv).std() / detrend(lv_nm).std()
                    else:
                        g_nm = g_raw
                    rows.append(dict(
                        f_av=int(fid == sdc), l_av=int(lid == sdc),
                        mean_speed=float(fv.mean()),
                        l_fluct=float(lf.std()),
                        g_raw=float(g_raw), g_nm=float(g_nm)))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    s = df[(df.mean_speed > 10) & (df.l_fluct >= 0.3)]
    print(f"\npairs (freeway, disturbance>=0.3): {len(s)}")
    for fa, la, lab in [(0, 0, "human|human"), (0, 1, "human|AV"), (1, 0, "AV|human")]:
        sub = s[(s.f_av == fa) & (s.l_av == la)]
        if len(sub) < 30:
            continue
        print(f"{lab:12s} n={len(sub):6d}  raw={sub.g_raw.median():.3f}  "
              f"noise-matched={sub.g_nm.median():.3f}")


if __name__ == "__main__":
    main()
