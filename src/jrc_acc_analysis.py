"""JRC Open ACC Database: parse trajectories and compute string-stability gains.

Data format: CSV with metadata header rows, then Time,Speed1..N,E1..N,N1..N columns.
Vehicle 1 is the leader (manual driving), followers 2-5 use ACC.
We compute the same fluctuation-gain metric as WOMD/NGSIM for direct comparison.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DATA_DIR = "E:/av_style_data/jrc_acc"
DT = 0.1
SG_W, SG_O = 7, 2
MIN_PAIR_STEPS = 81  # 8.1 s
MA_SEC = 5.0
MIN_SPEED = 3.0


def detrend(v):
    v = savgol_filter(v, SG_W, SG_O) if v.size > SG_W else v
    half = 25
    ker = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(np.pad(v, half, mode="edge"), ker, mode="valid")[: v.size]
    return v - trend


def parse_jrc_csv(path):
    """Parse a JRC ACC CSV file into per-vehicle speed arrays."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # find the header row (starts with "Time,Speed")
    hdr_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Time,Speed"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None, 0

    # parse header to get column names
    hdr = lines[hdr_idx].strip().split(",")
    speed_cols = [j for j, c in enumerate(hdr) if c.startswith("Speed")]
    n_veh = len(speed_cols)
    acc_line = ""
    for line in lines[:hdr_idx]:
        if line.startswith("ACC,"):
            acc_line = line.strip().split(",")[1]
            break
    is_acc = acc_line == "1"

    # parse data
    data = []
    for line in lines[hdr_idx + 1:]:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            speeds = [float(parts[j]) for j in speed_cols if j < len(parts)]
            data.append([t] + speeds)
        except (ValueError, IndexError):
            continue

    if len(data) < MIN_PAIR_STEPS:
        return None, n_veh

    arr = np.array(data)
    return arr, n_veh


def compute_gains(path):
    """Compute leader-follower gains for all consecutive pairs."""
    arr, n_veh = parse_jrc_csv(path)
    if arr is None or n_veh < 2:
        return []

    results = []
    t = arr[:, 0]
    # normalize time to start at 0
    t = t - t[0]

    # compute gains for each consecutive pair (leader=i, follower=i+1)
    for i in range(n_veh - 1):
        v_lead = arr[:, 1 + i]
        v_foll = arr[:, 1 + i + 1]

        # filter valid segments
        valid = (v_lead > MIN_SPEED) & (v_foll > MIN_SPEED)
        if valid.sum() < MIN_PAIR_STEPS:
            continue

        # split into contiguous valid runs
        splits = np.flatnonzero(np.diff(valid.astype(int)) != 0) + 1
        for run in np.split(np.arange(len(valid)), splits):
            if run.size < MIN_PAIR_STEPS or not valid[run].all():
                continue

            tt = t[run]
            lf = detrend(v_lead[run])
            ff = detrend(v_foll[run])
            if lf.std() < 0.05:
                continue

            gain = ff.std() / lf.std()

            # 30s windows for comparability
            for s0 in range(0, max(run.size - 300, 1), 300):
                seg = run[s0:s0 + 300]
                if seg.size < MIN_PAIR_STEPS:
                    break
                f2 = detrend(v_foll[seg])
                l2 = detrend(v_lead[seg])
                if l2.std() < 0.05:
                    continue
                results.append(dict(
                    file=os.path.basename(path),
                    pair=f"{i+1}->{i+2}",
                    n_steps=int(seg.size),
                    mean_speed=float(np.mean(v_foll[seg])),
                    gain=float(f2.std() / l2.std()),
                    leader_fluct_std=float(l2.std()),
                    follower_fluct_std=float(f2.std()),
                ))
    return results


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "JRC-VC_*.csv")))
    files = [f for f in files if os.path.getsize(f) > 10000]
    print(f"parsing {len(files)} files...")

    all_rows = []
    for path in files:
        rows = compute_gains(path)
        all_rows.extend(rows)
        print(f"  {os.path.basename(path)}: {len(rows)} pairs")

    df = pd.DataFrame(all_rows)
    out = os.path.join(DATA_DIR, "jrc_acc_gains.csv")
    df.to_csv(out, index=False)
    print(f"\ntotal pairs: {len(df)} -> {out}")

    # summary
    if len(df) > 0:
        print("\n=== overall ===")
        print(f"gain median: {df.gain.median():.3f}")
        print(f"frac unstable (>1): {(df.gain > 1).mean():.3f}")
        print(f"mean speed: {df.mean_speed.mean():.1f} m/s")

        # by disturbance stratum
        for lo, hi in [(0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 3.0)]:
            sub = df[(df.leader_fluct_std >= lo) & (df.leader_fluct_std < hi)]
            if len(sub) >= 10:
                print(f"  disturbance {lo}-{hi}: n={len(sub)}, "
                      f"gain={sub.gain.median():.3f}, "
                      f"unstable={(sub.gain > 1).mean():.3f}")


if __name__ == "__main__":
    main()
