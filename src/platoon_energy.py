"""H3 platoon energy simulation (pure Python, vectorized over vehicles).

Translates the measured style/stability differences into energy effects.
A platoon of N vehicles follows a lead profile that injects stop-and-go
waves; followers are either human-calibrated (IDM + 0.7 s sample-hold
delay + accel noise) or AV-calibrated (0.1 s update, jerk cap, closing-
speed anticipation). Controllers were validated against measured chain
gains (human ~1.05 cumulative, AV ~0.40 per platoon_probe.py).

Outputs: energy per vehicle-km (compact EV model), stop fraction and
speed variability vs AV penetration; wave survival depth.
"""
import json
import os

import numpy as np
import pandas as pd

N = 60
DT = 0.1
T_TOTAL = 900.0          # s
WAVE_EVERY = 45.0        # s between wave injections
WAVE = dict(t_dec=3.0, v_hold=0.0, t_hold=6.0, t_acc=6.0, v_base=12.0)

HUMAN = dict(a_max=1.8, b_comf=2.2, s0=2.0, T=1.1, v0=20.0,
             every=7, noise=0.35, jerk_cap=None)
AV = dict(a_max=1.4, b_comf=1.8, s0=2.0, T=1.1, v0=20.0,
          every=1, noise=0.0, jerk_cap=1.5, t_close=0.3, t_cap=1.8)

M, CRR, CDA, RHO, ETA, REGEN, REGEN_P_MAX = 1800.0, 0.010, 0.70, 1.20, 0.85, 0.60, 60e3
VEH_LEN = 5.0


def leader_speed(t):
    """Base cruise with periodic full stop-and-go waves."""
    v = np.full_like(t, WAVE["v_base"])
    phase = np.mod(t, WAVE_EVERY)
    d, h, a = WAVE["t_dec"], WAVE["t_hold"], WAVE["t_acc"]
    dec = phase < d
    v[dec] = WAVE["v_base"] * (1 - phase[dec] / d)
    hold = (phase >= d) & (phase < d + h)
    v[hold] = WAVE["v_hold"]
    acc = (phase >= d + h) & (phase < d + h + a)
    v[acc] = WAVE["v_base"] * (phase[acc] - d - h) / a
    return v


def battery_power(v, a):
    p = v * (M * a + M * 9.81 * CRR + 0.5 * RHO * CDA * v * v)
    out = np.where(p >= 0, p / ETA, -np.minimum(-p, REGEN_P_MAX / REGEN) * REGEN)
    return out


def idm_accel(v, dv, gap, a_max, b_comf, s0, T, v0):
    s_star = s0 + np.maximum(0.0, v * T - v * dv / (2 * np.sqrt(a_max * b_comf)))
    return a_max * (1 - (v / v0) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)


def run_platoon(n_av, seed, n=N, t_total=T_TOTAL, record_from=100.0):
    rng = np.random.default_rng(seed)
    n_steps = int(t_total / DT)
    t = np.arange(n_steps) * DT
    lead_v = leader_speed(t)

    # AV placement: uniformly spaced
    is_av = np.zeros(n, dtype=bool)
    if n_av > 0:
        idx = np.linspace(0, n - 1, n_av).round().astype(int)
        is_av[idx] = True
    # positions: index 0 nearest the leader
    x = np.cumsum(np.full(n, -(30.0)))[::-1]  # 30 m initial spacing
    v = np.full(n, WAVE["v_base"])

    a_cmd = np.zeros(n)
    last_cmd = v.copy()
    e_pos = np.zeros(n)

    # per-vehicle parameter vectors
    pv = dict(
        a_max=np.where(is_av, AV["a_max"], HUMAN["a_max"]),
        b_comf=np.where(is_av, AV["b_comf"], HUMAN["b_comf"]),
        s0=np.where(is_av, AV["s0"], HUMAN["s0"]),
        T=np.where(is_av, AV["T"], HUMAN["T"]),
        v0=np.where(is_av, AV["v0"], HUMAN["v0"]),
        noise=np.where(is_av, AV["noise"], HUMAN["noise"]),
        every=np.where(is_av, AV["every"], HUMAN["every"]),
        jcap=np.where(is_av, AV["jerk_cap"], 10.0),
    )
    stop_time = np.zeros(n)
    dist = np.zeros(n)
    v_mean_acc = np.zeros(n)
    v_sq_acc = np.zeros(n)
    v_cnt = np.zeros(n)

    rec_start = int(record_from / DT)
    for k in range(n_steps):
        # leader (virtual) position/speed drives vehicle 0
        if k == 0:
            x_lead, v_lead = 0.0, lead_v[0]
        else:
            x_lead += (lead_v[k - 1] + lead_v[k]) / 2 * DT
            v_lead = lead_v[k]

        # gaps: bumper-to-bumper to vehicle ahead (vehicle 0 -> virtual leader)
        gaps = np.empty(n)
        gaps[0] = (x_lead - x[0]) - VEH_LEN
        gaps[1:] = (x[:-1] - x[1:]) - VEH_LEN
        lv = np.empty(n)
        lv[0] = v_lead
        lv[1:] = v[:-1]

        # controller updates: humans every 7 steps, AVs every step
        upd = is_av | (k % HUMAN["every"] == 0)
        idx_upd = np.nonzero(upd)[0]
        if idx_upd.size:
            i = idx_upd
            closing = v[i] - lv[i]
            T_eff = pv["T"][i].copy()
            antic = is_av[i] & (closing > 0)
            T_eff = np.where(antic, np.clip(pv["T"][i] + AV["t_close"] * closing,
                                            pv["T"][i], AV["t_cap"]), T_eff)
            acc = idm_accel(v[i], closing, gaps[i], pv["a_max"][i], pv["b_comf"][i],
                            pv["s0"][i], T_eff, pv["v0"][i])
            acc = acc + rng.normal(0.0, 1.0, i.size) * pv["noise"][i]
            acc = np.clip(acc, -pv["b_comf"][i], pv["a_max"][i])
            # slew-rate limit relative to previous command (jerk cap, AVs)
            dt_hold = np.where(is_av[i], DT, HUMAN["every"] * DT)
            acc = np.clip(acc, a_cmd[i] - pv["jcap"][i] * dt_hold,
                          a_cmd[i] + pv["jcap"][i] * dt_hold)
            a_cmd[i] = acc

        # safety floor: avoid negative gaps under noise
        v_safe = np.sqrt(np.maximum(0.0, lv ** 2 + 2 * 3.0 * (gaps - 1.5)))
        v_next = np.minimum(v + a_cmd * DT, v_safe + 0.3)
        v_next = np.maximum(v_next, v - 8.0 * DT)
        v_next = np.maximum(v_next, 0.0)

        if k >= rec_start:
            p_bat = battery_power(v, a_cmd)
            e_pos += np.maximum(p_bat, 0.0) * DT
            dist += v * DT
            stop_time += (v < 0.5).astype(float) * DT
            v_mean_acc += v
            v_sq_acc += v ** 2
            v_cnt += 1

        x = x + (v + v_next) / 2 * DT
        # keep leader frame: re-anchor to leader position drift
        x = x - (x_lead - 0.0) * 0.0
        v = v_next

    vm = v_mean_acc / np.maximum(v_cnt, 1)
    vs = np.sqrt(np.maximum(v_sq_acc / np.maximum(v_cnt, 1) - vm ** 2, 0))
    out = dict(
        n_av=n_av, seed=seed,
        e_pos_kWh_100km=100 * e_pos.sum() / 3.6e6 / max(dist.sum() / 1000, 1e-6),
        e_av_kWh_100km=100 * e_pos[is_av].sum() / 3.6e6 / max(dist[is_av].sum() / 1000, 1e-6) if is_av.any() else np.nan,
        e_hum_kWh_100km=100 * e_pos[~is_av].sum() / 3.6e6 / max(dist[~is_av].sum() / 1000, 1e-6) if (~is_av).any() else np.nan,
        stop_frac=stop_time.mean() / (t_total - record_from),
        fleet_v_std=float(vs.mean()),
        throughput_vmean=float(vm.mean()),
        frac_touched_zero=float((stop_time > 0.5).mean()),
    )
    return out


def main():
    rows = []
    for p in [0, 5, 10, 15, 20, 30, 50, 100]:
        n_av = max(0, int(round(N * p / 100)))
        for seed in range(6):
            r = run_platoon(n_av, seed)
            rows.append(r)
            print(f"p={p:3d}% n_av={n_av:2d} seed={seed}: "
                  f"e={r['e_pos_kWh_100km']:.2f} kWh/100km "
                  f"stops={r['stop_frac']:.3f} touched0={r['frac_touched_zero']:.2f} "
                  f"vmean={r['throughput_vmean']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "platoon_results.csv"), index=False)
    print("\n=== summary (mean over seeds) ===")
    print(df.groupby("n_av").agg(
        e=("e_pos_kWh_100km", "mean"), e_sd=("e_pos_kWh_100km", "std"),
        stops=("stop_frac", "mean"), zero=("frac_touched_zero", "mean"),
        vmean=("throughput_vmean", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
