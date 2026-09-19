"""H3 ring-road experiment: stop-and-go wave damping and energy effects of
Waymo-style AV driving.

Setup follows the ring paradigm of Stern et al. (2018, TR-C): 22 vehicles on
a 230 m single-lane ring. Every vehicle is controlled through traci with
explicit longitudinal controllers calibrated to our measured distributions:

  human : IDM + 0.7 s reaction delay (sample-and-hold) + accel noise,
          no jerk limit  (p95 |jerk| ~ 10-16 m/s^3, disturbance amplifying)
  AV    : IDM with 0.1 s update, longer headway T, acceleration slew-rate
          limiter (jerk cap ~1.5 m/s^3), lower accel authority
          (measured p95 |jerk| ~ 2.5 m/s^3, disturbance attenuating)

Protocol: settle -> hard braking perturbation on one human -> measure 450 s.
"""
import math
import os
import subprocess

import numpy as np
import pandas as pd
import traci

SUMO_HOME = os.environ.get("SUMO_HOME", r"${SUMO_HOME}")
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(SIM_DIR, "ring.net.xml")

RING_LEN = 230.0
N_VEH = 22

HUMAN = dict(a_max=1.8, b_comf=2.2, s0=2.0, T=1.1, v0=15.0,
             delay=0.7, noise=0.35, jerk_cap=None)
# AV is capacity-neutral at equilibrium and stabilizes by anticipation:
# fast updates, no noise, jerk cap, and a closing-speed-dependent headway
# that brakes early when approaching a disturbance (wave absorption) - the
# mechanism implied by the measured low-speed anticipatory style and the
# strong disturbance-conditional attenuation (gain 0.4-0.78).
AV = dict(a_max=1.4, b_comf=1.8, s0=2.0, T=1.1, v0=15.0,
          delay=0.1, noise=0.0, jerk_cap=1.5, t_close=0.55, t_cap=2.4)

# compact EV energy model (relative savings robust to model error)
M, CRR, CDA, RHO, ETA, REGEN, REGEN_P_MAX = 1800.0, 0.010, 0.70, 1.20, 0.85, 0.60, 60e3


def battery_power(v, a):
    p = v * (M * a + M * 9.81 * CRR + 0.5 * RHO * CDA * v * v)
    if p >= 0:
        return p / ETA
    p_wheel_max = REGEN_P_MAX / REGEN
    recov = min(-p, p_wheel_max) * REGEN
    return -recov  # negative = battery charging


def vsp(v, a):
    return v * (1.1 * a + 0.132) + 0.000302 * v ** 3


def arc_shape(rr, a0, a1):
    return " ".join("%.2f,%.2f" % (rr * math.cos(t), rr * math.sin(t))
                    for t in np.linspace(a0, a1, 25))


def build_ring_net():
    r = RING_LEN / (2 * math.pi)
    nod = os.path.join(SIM_DIR, "ring.nod.xml")
    edg = os.path.join(SIM_DIR, "ring.edg.xml")
    with open(nod, "w") as f:
        f.write('<nodes><node id="A" x="%.2f" y="0"/><node id="B" x="%.2f" y="0"/></nodes>'
                % (-r, r))
    with open(edg, "w") as f:
        f.write('<edges>')
        f.write('<edge id="top" from="A" to="B" numLanes="1" speed="30" shape="%s"/>'
                % arc_shape(r, math.pi, 0))
        f.write('<edge id="bot" from="B" to="A" numLanes="1" speed="30" shape="%s"/>'
                % arc_shape(r, 0, -math.pi))
        f.write('</edges>')
    subprocess.run([os.path.join(SUMO_HOME, "bin", "netconvert.exe"),
                    "-n", nod, "-e", edg, "--no-internal-links", "true",
                    "--no-turnarounds", "true", "-o", NET], check=True)


def idm_accel(v, dv, gap, p):
    s_star = p["s0"] + max(0.0, v * p["T"] - v * dv / (2 * math.sqrt(p["a_max"] * p["b_comf"])))
    return p["a_max"] * (1 - (v / p["v0"]) ** 4 - (s_star / max(gap, 0.1)) ** 2)


def run_ring(n_av, seed, t_end=600.0, t_perturb=150.0, perturb_dur=1.5):
    if not os.path.exists(NET):
        build_ring_net()
    rng = np.random.default_rng(seed)
    traci.start([os.path.join(SUMO_HOME, "bin", "sumo.exe"), "-n", NET,
                 "--no-step-log", "true", "--time-to-teleport", "-1",
                 "--collision.action", "warn",
                 "--step-length", "0.1"])
    try:
        traci.route.add("loopA", ["top", "bot"] * 60)
        traci.route.add("loopB", ["bot", "top"] * 60)
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "car")
        traci.vehicletype.setLength("car", 5.0)
        traci.vehicletype.setMinGap("car", 0.0)
        flags = [1] * n_av + [0] * (N_VEH - n_av)
        rng.shuffle(flags)
        ids = []
        for i in range(N_VEH):
            vid = "v%02d" % i
            pos = i * RING_LEN / N_VEH
            route, dp = ("loopA", pos) if pos < RING_LEN / 2 else ("loopB", pos - RING_LEN / 2)
            traci.vehicle.add(vid, route, typeID="car", departPos="%.2f" % dp,
                              departSpeed="0", departLane="0")
            ids.append(vid)

        traci.simulation.step()  # simultaneous insertion (minGap=0)
        for vid in ids:
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setLaneChangeMode(vid, 0)
        assert len(traci.vehicle.getIDList()) == N_VEH, "insertion failed"

        v_cmd = {i: 3.0 for i in ids}
        last_sample = {i: -9.0 for i in ids}
        accel_hist = {i: 0.0 for i in ids}
        rec_t, rec_v, rec_a = [], {i: [] for i in ids}, {i: [] for i in ids}
        st = {"t": [], "x": {i: [] for i in ids}}
        hum_idx = [i for i, f in enumerate(flags) if f == 0]
        pert_idx = hum_idx[0] if hum_idx else 0

        while traci.simulation.getTime() < t_end:
            t = traci.simulation.getTime()
            # manual ring-frame leader computation (robust across junctions)
            ring_pos, speeds = {}, {}
            for vid in ids:
                road = traci.vehicle.getRoadID(vid)
                pos = traci.vehicle.getLanePosition(vid)
                ring_pos[vid] = pos % (RING_LEN / 2) + (0 if road == "top" else RING_LEN / 2)
                speeds[vid] = traci.vehicle.getSpeed(vid)
            info = {}
            for vid in ids:
                best_gap, best_lv = None, None
                for oid in ids:
                    if oid == vid:
                        continue
                    g = (ring_pos[oid] - ring_pos[vid]) % RING_LEN
                    if g < 1.0:  # overlap guard during transients
                        g += RING_LEN
                    if best_gap is None or g < best_gap:
                        best_gap, best_lv = g, speeds[oid]
                gap = max(best_gap - 5.0, 0.5)  # bumper-to-bumper
                info[vid] = (ring_pos[vid], speeds[vid], gap, best_lv)

            for vi, vid in enumerate(ids):
                p = AV if flags[vi] else HUMAN
                pos, spd, gap, lv = info[vid]
                if t - last_sample[vid] >= p["delay"]:
                    dv = spd - lv
                    if "t_close" in p and dv > 0:
                        # anticipatory wave absorption: brake early when closing
                        p_eff = dict(p, T=float(np.clip(p["T"] + p["t_close"] * dv,
                                                        p["T"], p["t_cap"])))
                    else:
                        p_eff = p
                    acc = idm_accel(spd, dv, gap, p_eff)
                    if p["noise"]:
                        acc += rng.normal(0, p["noise"])
                    acc = float(np.clip(acc, -p["b_comf"], p["a_max"]))
                    if p["jerk_cap"]:
                        acc = float(np.clip(acc,
                                            accel_hist[vid] - p["jerk_cap"] * p["delay"],
                                            accel_hist[vid] + p["jerk_cap"] * p["delay"]))
                    # safety floor against collisions under noise
                    v_safe = math.sqrt(max(0.0, lv * lv + 2 * 3.0 * (gap - 1.5)))
                    v_target = min(spd + acc * 0.1, v_safe + 0.3)
                    v_cmd[vid] = max(spd - 8.0 * 0.1, max(0.0, v_target))
                    accel_hist[vid] = acc
                    last_sample[vid] = t
                if vi == pert_idx and t_perturb <= t < t_perturb + perturb_dur:
                    v_cmd[vid] = max(0.0, spd - 2.4 * 0.1)
                traci.vehicle.setSpeed(vid, v_cmd[vid])

            if t >= t_perturb - 0.05:
                rec_t.append(t)
                for vid in ids:
                    rec_v[vid].append(info[vid][1])
                    rec_a[vid].append(traci.vehicle.getAcceleration(vid))
                if t >= t_perturb and t < t_perturb + 240:
                    st["t"].append(t)
                    for vid in ids:
                        st["x"][vid].append(traci.vehicle.getLanePosition(vid)
                                            + (0 if traci.vehicle.getRoadID(vid) == "top"
                                               else RING_LEN / 2))
            traci.simulation.step()

        # metrics
        rows = []
        for vi, vid in enumerate(ids):
            v = np.array(rec_v[vid]); a = np.array(rec_a[vid]); dt = 0.1
            p_bat = np.array([battery_power(vi_, ai) for vi_, ai in zip(v, a)])
            rows.append(dict(
                vid=vid, is_av=flags[vi],
                dist_km=v.sum() * dt / 1000.0,
                e_pos_kWh=np.maximum(p_bat, 0).sum() * dt / 3.6e6,
                e_net_kWh=(np.maximum(p_bat, 0).sum() - np.maximum(-p_bat, 0).sum()) * dt / 3.6e6,
                vsp_kJ=np.maximum(vsp(v, a), 0).sum() * dt / 1000.0,
                v_mean=v.mean(), v_std=v.std(), v_min=v.min()))
        df = pd.DataFrame(rows)
        fleet = dict(
            n_av=n_av, seed=seed,
            e_pos_kWh_100km=100 * df.e_pos_kWh.sum() / df.dist_km.sum(),
            e_net_kWh_100km=100 * df.e_net_kWh.sum() / df.dist_km.sum(),
            vsp_per_km=df.vsp_kJ.sum() / df.dist_km.sum(),
            fleet_v_std=float(np.mean(df.v_std)),
            fleet_v_min=float(df.v_min.min()),
            stop_fraction=float(np.mean([np.mean(np.array(rec_v[i]) < 0.5) for i in ids])),
            n_veh=len(df))
        spacetime = None
        if len(st["t"]) > 0:
            spacetime = pd.DataFrame(st["x"]); spacetime.insert(0, "t", st["t"])
        return fleet, spacetime
    finally:
        traci.close()


def main():
    out, st0, st4 = [], None, None
    for n_av in range(0, N_VEH + 1, 2):
        for seed in range(5):
            fleet, st = run_ring(n_av, seed)
            out.append(fleet)
            if n_av == 0 and seed == 0:
                st0 = st
            if n_av == 4 and seed == 0:
                st4 = st
            print(f"n_av={n_av:2d} seed={seed}: e_pos={fleet['e_pos_kWh_100km']:.2f} "
                  f"kWh/100km v_std={fleet['fleet_v_std']:.2f} "
                  f"v_min={fleet['fleet_v_min']:.2f} stops={fleet['stop_fraction']:.3f}",
                  flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(SIM_DIR, "ring_results.csv"), index=False)
    if st0 is not None:
        st0.to_csv(os.path.join(SIM_DIR, "spacetime_av0.csv"), index=False)
    if st4 is not None:
        st4.to_csv(os.path.join(SIM_DIR, "spacetime_av4.csv"), index=False)
    print("\n=== summary by n_av (mean +/- sd over 5 seeds) ===")
    print(df.groupby("n_av").agg(
        e_pos=("e_pos_kWh_100km", "mean"), e_pos_sd=("e_pos_kWh_100km", "std"),
        e_net=("e_net_kWh_100km", "mean"),
        v_std=("fleet_v_std", "mean"), v_min=("fleet_v_min", "mean"),
        stops=("stop_fraction", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
