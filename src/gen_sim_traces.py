"""Generate simulation time-series data for Fig 3 and ED Fig 2 panels."""
import os
import numpy as np
import pandas as pd

import platoon_energy as PE


def run_with_series(n_av, seed, t_total=400, record_from=100):
    """Like run_platoon but returns speed time series for sampled vehicles."""
    rng = np.random.default_rng(seed)
    n = PE.N
    n_steps = int(t_total / PE.DT)
    t = np.arange(n_steps) * PE.DT
    lead_v = PE.leader_speed(t)

    is_av = np.zeros(n, dtype=bool)
    if n_av > 0:
        idx = np.linspace(0, n - 1, n_av).round().astype(int)
        is_av[idx] = True

    x = np.cumsum(np.full(n, -30.0))[::-1]
    v = np.full(n, PE.WAVE["v_base"])
    a_cmd = np.zeros(n)

    pv = dict(
        a_max=np.where(is_av, PE.AV["a_max"], PE.HUMAN["a_max"]),
        b_comf=np.where(is_av, PE.AV["b_comf"], PE.HUMAN["b_comf"]),
        s0=np.where(is_av, PE.AV["s0"], PE.HUMAN["s0"]),
        T=np.where(is_av, PE.AV["T"], PE.HUMAN["T"]),
        v0=np.where(is_av, PE.AV["v0"], PE.HUMAN["v0"]),
        noise=np.where(is_av, PE.AV["noise"], PE.HUMAN["noise"]),
        every=np.where(is_av, PE.AV["every"], PE.HUMAN["every"]),
        jcap=np.where(is_av, PE.AV["jerk_cap"], 10.0),
    )

    rec_start = int(record_from / PE.DT)
    sampled = [5, 15, 25, 35, 45, 55]
    series = {i: [] for i in sampled}
    e_pos = np.zeros(n)
    dist = np.zeros(n)

    x_lead = 0.0
    for k in range(n_steps):
        if k == 0:
            pass
        else:
            x_lead += (lead_v[k-1] + lead_v[k]) / 2 * PE.DT
        v_lead = lead_v[k]

        gaps = np.empty(n)
        gaps[0] = (x_lead - x[0]) - PE.VEH_LEN
        gaps[1:] = (x[:-1] - x[1:]) - PE.VEH_LEN
        lv = np.empty(n)
        lv[0] = v_lead
        lv[1:] = v[:-1]

        upd = is_av | (k % PE.HUMAN["every"] == 0)
        idx_upd = np.nonzero(upd)[0]
        if idx_upd.size:
            i = idx_upd
            closing = v[i] - lv[i]
            T_eff = pv["T"][i].copy()
            antic = is_av[i] & (closing > 0)
            T_eff = np.where(antic, np.clip(pv["T"][i] + PE.AV["t_close"] * closing,
                                            pv["T"][i], PE.AV["t_cap"]), T_eff)
            acc = PE.idm_accel(v[i], closing, gaps[i], pv["a_max"][i],
                               pv["b_comf"][i], pv["s0"][i], T_eff, pv["v0"][i])
            acc = acc + rng.normal(0.0, 1.0, i.size) * pv["noise"][i]
            acc = np.clip(acc, -pv["b_comf"][i], pv["a_max"][i])
            dt_hold = np.where(is_av[i], PE.DT, PE.HUMAN["every"] * PE.DT)
            acc = np.clip(acc, a_cmd[i] - pv["jcap"][i] * dt_hold,
                          a_cmd[i] + pv["jcap"][i] * dt_hold)
            a_cmd[i] = acc

        v_safe = np.sqrt(np.maximum(0.0, lv**2 + 2 * 3.0 * (gaps - 1.5)))
        v_next = np.minimum(v + a_cmd * PE.DT, v_safe + 0.3)
        v_next = np.maximum(v_next, v - 8.0 * PE.DT)
        v_next = np.maximum(v_next, 0.0)

        if k >= rec_start:
            p_bat = PE.battery_power(v, a_cmd)
            e_pos += np.maximum(p_bat, 0.0) * PE.DT
            dist += v * PE.DT
            for i in sampled:
                series[i].append(v[i])

        x = x + (v + v_next) / 2 * PE.DT
        v = v_next

    return t[rec_start:], series, e_pos, dist, is_av, sampled


def main():
    # generate time series at 0% and 50% penetration
    results = {}
    for pen, n_av in [(0, 0), (50, 30)]:
        t, series, e_pos, dist, is_av, sampled = run_with_series(n_av, 0)
        results[pen] = dict(t=t, series=series, e_pos=e_pos, dist=dist,
                            is_av=is_av, sampled=sampled)
        print(f"pen={pen}%: {len(t)} time steps, {len(sampled)} vehicles sampled")

    # save speed traces
    rows = []
    for pen in [0, 50]:
        r = results[pen]
        for i in r["sampled"]:
            for ti, vi in zip(r["t"], r["series"][i]):
                rows.append(dict(pen=pen, veh=i, t=ti, v=vi))
    pd.DataFrame(rows).to_csv("sim_speed_traces.csv", index=False)
    print("saved sim_speed_traces.csv")

    # save per-vehicle energies
    erows = []
    for pen in [0, 50]:
        r = results[pen]
        for i in range(len(r["e_pos"])):
            erows.append(dict(pen=pen, veh=i, is_av=bool(r["is_av"][i]),
                              e_kwh=100 * r["e_pos"][i] / 3.6e6 /
                              max(r["dist"][i] / 1000, 1e-6)))
    pd.DataFrame(erows).to_csv("sim_per_vehicle_energy.csv", index=False)
    print("saved sim_per_vehicle_energy.csv")


if __name__ == "__main__":
    main()
