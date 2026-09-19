"""String-stability (H2) analysis on WOMD scenarios.

For every sustained leader-follower pair we estimate the empirical
amplification gain of speed fluctuations from leader to follower:

    G = std(follower fluctuation) / std(leader fluctuation)

with fluctuation = speed - 5 s moving average (0.2-0.5 Hz band proxy).
G > 1 means the follower amplifies disturbances (string-unstable driving).

Pairs are identified geometrically: B follows A if A sits within a corridor
along B's heading (gap 0-60 m, lateral offset < 2.5 m) for >= 8 s with both
speeds > 3 m/s. The follower is labelled AV or HDV via sdc_track_index.

Usage: python string_stability.py <out_pairs.csv> <tfrecord files...>
"""
import sys
import os
import csv

import numpy as np
from scipy.signal import savgol_filter

from womd_parsing import iter_scenarios
import waymo_open_dataset.protos.scenario_pb2 as scenario_pb2

MIN_SPEED = 3.0          # m/s, exclude stop-start creep noise in pilot v1
MIN_GAP = 0.0
MAX_GAP = 60.0           # m along follower heading
MAX_LAT = 2.5            # m lateral offset
MIN_PAIR_STEPS = 81      # 8.1 s of sustained following at 10 Hz
SG_WINDOW = 7
SG_ORDER = 2
MA_SEC = 5.0             # fluctuation band removal window

COLS = [
    "scenario_id", "file", "follower_is_av", "leader_is_av",
    "n_steps", "mean_speed",
    "gain", "gain_fft", "gain_raw", "lag_s",
    "leader_fluct_std", "follower_fluct_std",
]


def build_tracks(sc):
    """Return (t, tracks) truncated to the shortest of timestamps/states.

    v1.3 occasionally has len(timestamps) != len(states); we cut everything
    to the common length so downstream indexing stays safe.
    """
    veh = [tr for tr in sc.tracks if tr.object_type == scenario_pb2.Track.TYPE_VEHICLE]
    if not veh:
        return np.array([]), {}
    n = min([len(sc.timestamps_seconds)] + [len(tr.states) for tr in veh])
    t = np.array(sc.timestamps_seconds[:n], dtype=float)
    tracks = {}
    for tr in veh:
        st = tr.states[:n]
        v = np.array([s.valid for s in st], dtype=bool)
        if not v.any():
            continue
        tracks[tr.id] = {
            "x": np.array([s.center_x for s in st], dtype=float),
            "y": np.array([s.center_y for s in st], dtype=float),
            "vx": np.array([s.velocity_x for s in st], dtype=float),
            "vy": np.array([s.velocity_y for s in st], dtype=float),
            "h": np.array([s.heading for s in st], dtype=float),
            "valid": v,
        }
    return t, tracks


def find_pairs(t, tracks):
    """Geometric leader-follower detection, vectorised per timestep."""
    n = len(t)
    ids = list(tracks.keys())
    pairs = {}  # (follower, leader) -> list of timestep indices
    X = np.stack([tracks[i]["x"] for i in ids])  # (n_tracks, n_steps)
    Y = np.stack([tracks[i]["y"] for i in ids])
    H = np.stack([tracks[i]["h"] for i in ids])
    VAL = np.stack([tracks[i]["valid"] for i in ids])
    VX = np.stack([tracks[i]["vx"] for i in ids])
    VY = np.stack([tracks[i]["vy"] for i in ids])
    k = len(ids)
    for step in range(n):
        dx = X[:, step][None, :] - X[:, step][:, None]   # leader - follower
        dy = Y[:, step][None, :] - Y[:, step][:, None]
        hx = np.cos(H[:, step])
        hy = np.sin(H[:, step])
        lon = dx * hx[:, None] + dy * hy[:, None]        # along follower heading
        lat = -dx * hy[:, None] + dy * hx[:, None]       # perpendicular
        spd = np.hypot(VX[:, step], VY[:, step])
        ok = (
            (np.eye(k, dtype=bool) == False)
            & VAL[:, step][:, None] & VAL[:, step][None, :]
            & (spd[:, None] > MIN_SPEED) & (spd[None, :] > MIN_SPEED)
            & (lon > MIN_GAP) & (lon < MAX_GAP) & (np.abs(lat) < MAX_LAT)
        )
        f, l = np.nonzero(ok)
        for fi, li in zip(f, l):
            key = (ids[fi], ids[li])
            pairs.setdefault(key, []).append(step)
    return pairs


def contiguous_runs(steps, min_len):
    steps = np.asarray(steps)
    splits = np.flatnonzero(np.diff(steps) > 1) + 1
    for run in np.split(steps, splits):
        if run.size >= min_len:
            yield run


def pair_metrics(run, t, fa, la, sdc_id):
    tt = t[run]
    dt = np.median(np.diff(tt))
    ma_half = max(int(MA_SEC / dt / 2), 2)
    fs, ls = fa[run], la[run]

    def fluct(v):
        if v.size > SG_WINDOW:
            v = savgol_filter(v, SG_WINDOW, SG_ORDER)
        # 5 s moving-average detrend (edge-safe via convolution)
        ker = np.ones(2 * ma_half + 1) / (2 * ma_half + 1)
        pad = np.pad(v, ma_half, mode="edge")
        trend = np.convolve(pad, ker, mode="valid")[: v.size]
        return v - trend, v

    ff, fs_s = fluct(fs)
    lf, ls_s = fluct(ls)
    denom = lf.std()
    if denom < 0.05:  # leader essentially steady; gain not identifiable
        return None
    gain = ff.std() / denom
    # robust companion: std ratio on smoothed speeds (no detrend)
    gain_raw = fs_s.std() / max(ls_s.std(), 1e-9)

    # frequency-domain gain in the 0.05-0.5 Hz band (stop-and-go band)
    nfft = int(2 ** np.ceil(np.log2(ff.size)))
    Ff = np.abs(np.fft.rfft(ff - ff.mean(), nfft)) ** 2
    Lf = np.abs(np.fft.rfft(lf - lf.mean(), nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=dt)
    band = (freq >= 0.05) & (freq <= 0.5)
    gain_fft = (Ff[band].sum() / max(Lf[band].sum(), 1e-9)) ** 0.5

    # reaction lag: argmax cross-correlation of follower vs leader fluctuations
    a = (ff - ff.mean()) / ff.std()
    b = (lf - lf.mean()) / lf.std()
    max_lag = min(int(2.0 / dt), ff.size // 4)
    lags = range(-max_lag, max_lag + 1)
    cc = [np.mean(a[max(0, -l): ff.size - max(0, l)] *
                  b[max(0, l): ff.size - max(0, -l)]) for l in lags]
    lag_s = list(lags)[int(np.argmax(cc))] * dt

    return {
        "n_steps": int(run.size),
        "mean_speed": float(fs_s.mean()),
        "gain": float(gain),
        "gain_fft": float(gain_fft),
        "gain_raw": float(gain_raw),
        "lag_s": float(lag_s),
        "leader_fluct_std": float(lf.std()),
        "follower_fluct_std": float(ff.std()),
    }


def main():
    out_csv, files = sys.argv[1], sys.argv[2:]
    rows = []
    n_sc = 0
    for path in files:
        for sc in iter_scenarios(path):
            n_sc += 1
            ts = list(sc.timestamps_seconds)
            if len(ts) < 100:
                continue
            t, tracks = build_tracks(sc)
            if len(tracks) < 2:
                continue
            sdc_id = sc.tracks[sc.sdc_track_index].id
            # speed series cache
            spd = {i: np.hypot(tracks[i]["vx"], tracks[i]["vy"]) for i in tracks}
            for (fid, lid), steps in find_pairs(t, tracks).items():
                for run in contiguous_runs(steps, MIN_PAIR_STEPS):
                    m = pair_metrics(run, t, spd[fid], spd[lid], sdc_id)
                    if m is None:
                        continue
                    rows.append({
                        "scenario_id": sc.scenario_id,
                        "file": os.path.basename(path),
                        "follower_is_av": int(fid == sdc_id),
                        "leader_is_av": int(lid == sdc_id),
                        **m,
                    })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"scenarios={n_sc} following_pairs={len(rows)} -> {out_csv}")


if __name__ == "__main__":
    main()
