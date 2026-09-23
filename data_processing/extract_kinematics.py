"""Extract per-track kinematic style metrics from WOMD Scenario tfrecords.

For every vehicle track in every scenario we compute smoothness/aggression
metrics on contiguous valid segments. The AV (sdc_track_index) and the
surrounding human-driven vehicles (HDV) are flagged, giving a paired sample
per scenario. Output: one CSV row per track.

Usage: python extract_kinematics.py <out.csv> <tfrecord files...>
"""
import sys
import os
import csv

import numpy as np
from scipy.signal import savgol_filter

from womd_parsing import iter_scenarios
import waymo_open_dataset.protos.scenario_pb2 as scenario_pb2

# --- analysis constants ---
MIN_SEG_STEPS = 51          # >= 5.1 s @10 Hz
SEG_GAP_STEPS = 1           # any invalid step splits a segment
SG_WINDOW = 7               # Savitzky-Golay window (0.7 s), same for AV & HDV
SG_ORDER = 2
AGGR_A = 2.0                # |a| threshold for aggressive acc/dec (m/s^2)
HIGH_VSP = 20.0             # high-power OpMode proxy threshold (kW/tonne)
MIN_MEAN_SPEED = 1.0        # drop parked/creeping vehicles (m/s)
MIN_DISPLACEMENT = 15.0     # meters of net movement required

COLS = [
    "scenario_id", "file", "track_id", "is_av", "n_valid_steps", "duration_s",
    "mean_speed", "std_speed", "max_speed",
    "mean_abs_accel", "p95_abs_jerk", "p99_abs_accel", "p01_accel",
    "frac_aggr", "frac_hard_brake", "mean_vsp", "frac_high_vsp",
    # power-model integrals (P_wheel = m*va + m*g*Crr*v + 0.5*rho*CdA*v^3)
    "sum_v", "sum_v3", "sum_va",
    # duration-matched metrics (first ~11.3 s = 113 steps of the segment)
    "p95_abs_jerk_dm", "frac_aggr_dm", "mean_vsp_dm",
]


def segment_metrics(t, v, a, jerk, vsp):
    d = v.size
    # power-model integrals (dt-weighted sums)
    dt = np.gradient(t)
    return {
        "n_valid_steps": d,
        "duration_s": t[-1] - t[0] + 0.1,
        "mean_speed": float(np.mean(v)),
        "std_speed": float(np.std(v)),
        "max_speed": float(np.max(v)),
        "mean_abs_accel": float(np.mean(np.abs(a))),
        "p95_abs_jerk": float(np.percentile(np.abs(jerk), 95)),
        "p99_abs_accel": float(np.percentile(a, 99)),
        "p01_accel": float(np.percentile(a, 1)),
        "frac_aggr": float(np.mean(np.abs(a) > AGGR_A)),
        "frac_hard_brake": float(np.mean(a < -AGGR_A)),
        "mean_vsp": float(np.mean(vsp)),
        "frac_high_vsp": float(np.mean(vsp > HIGH_VSP)),
        "sum_v": float(np.sum(v * dt)),
        "sum_v3": float(np.sum(v ** 3 * dt)),
        "sum_va": float(np.sum(v * a * dt)),
        "p95_abs_jerk_dm": float(np.percentile(np.abs(jerk[:113]), 95)) if d >= 113 else None,
        "frac_aggr_dm": float(np.mean(np.abs(a[:113]) > AGGR_A)) if d >= 113 else None,
        "mean_vsp_dm": float(np.mean(vsp[:113])) if d >= 113 else None,
    }


def track_metrics(track, timestamps):
    """Return list of per-segment metric dicts for one vehicle track."""
    n = len(timestamps)
    valid = np.array([s.valid for s in track.states], dtype=bool)
    if valid.size != n or not valid.any():
        return []
    vx = np.array([s.velocity_x for s in track.states], dtype=float)
    vy = np.array([s.velocity_y for s in track.states], dtype=float)
    x = np.array([s.center_x for s in track.states], dtype=float)
    y = np.array([s.center_y for s in track.states], dtype=float)
    t = np.array(timestamps, dtype=float)

    # split into contiguous valid segments
    splits = np.flatnonzero(np.diff(valid.astype(int)) != 0) + 1
    segs = np.split(np.arange(n), splits)
    out = []
    for idx in segs:
        if not valid[idx].all() or idx.size < MIN_SEG_STEPS:
            continue
        seg = idx[valid[idx]] if not valid[idx].all() else idx
        tt, vxx, vyy = t[seg], vx[seg], vy[seg]
        v = np.hypot(vxx, vyy)
        disp = np.hypot(x[seg[-1]] - x[seg[0]], y[seg[-1]] - y[seg[0]])
        if v.mean() < MIN_MEAN_SPEED and disp < MIN_DISPLACEMENT:
            continue  # parked / stationary
        # equal smoothing for AV and perceived HDVs
        if v.size > SG_WINDOW:
            v_s = savgol_filter(v, SG_WINDOW, SG_ORDER)
        else:
            v_s = v
        a = np.gradient(v_s, tt)
        jerk = np.gradient(a, tt)
        # light-duty VSP (kW/tonne), zero road grade
        vsp = v_s * (1.1 * a + 0.132) + 0.000302 * v_s ** 3
        out.append(segment_metrics(tt, v_s, a, jerk, vsp))
    return out


def main():
    out_csv = sys.argv[1]
    files = sys.argv[2:]
    rows = []
    n_sc = 0
    for path in files:
        for sc in iter_scenarios(path):
            n_sc += 1
            ts = list(sc.timestamps_seconds)
            if len(ts) < 50:
                continue
            for tr in sc.tracks:
                if tr.object_type != scenario_pb2.Track.TYPE_VEHICLE:
                    continue
                is_av = int(tr.id == sc.tracks[sc.sdc_track_index].id)
                for m in track_metrics(tr, ts):
                    rows.append({
                        "scenario_id": sc.scenario_id,
                        "file": os.path.basename(path),
                        "track_id": tr.id,
                        "is_av": is_av,
                        **m,
                    })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"scenarios={n_sc} vehicle_tracks_with_rows={len(rows)} -> {out_csv}")


if __name__ == "__main__":
    main()
