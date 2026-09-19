"""Controller validation probe: open-road platoon chain gains.

Leader drives a sinusoidal speed profile; we measure how the disturbance
amplitude propagates down a platoon (gain_k = std(v_k)/std(v_{k-1})).
Targets from measured data: human ~0.95 (neutral/mild), AV <=0.8 under
meaningful disturbances. If the controllers reproduce these chain gains
here, ring-road damping should follow at the right density.
"""
import math
import os

import numpy as np
import pandas as pd
import traci

import ring_sim as R


def run_platoon(av_all=False, n=15, t_settle=120.0, t_probe=80.0, seed=0):
    net = os.path.join(R.SIM_DIR, "platoon.net.xml")
    if not os.path.exists(net):
        nod = os.path.join(R.SIM_DIR, "platoon.nod.xml")
        edg = os.path.join(R.SIM_DIR, "platoon.edg.xml")
        with open(nod, "w") as f:
            f.write('<nodes><node id="A" x="0" y="0"/><node id="B" x="3000" y="0"/></nodes>')
        with open(edg, "w") as f:
            f.write('<edges><edge id="e" from="A" to="B" numLanes="1" speed="30"/></edges>')
        subprocess_netconvert(net, nod, edg)

    rng = np.random.default_rng(seed)
    traci.start([os.path.join(R.SUMO_HOME, "bin", "sumo.exe"), "-n", net,
                 "--no-step-log", "true", "--time-to-teleport", "-1",
                 "--collision.action", "warn", "--step-length", "0.1"])
    try:
        traci.route.add("r", ["e"])
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "car")
        traci.vehicletype.setLength("car", 5.0)
        traci.vehicletype.setMinGap("car", 0.0)
        ids = []
        spacing = 22.0
        for i in range(n):
            vid = "p%02d" % i
            traci.vehicle.add(vid, "r", typeID="car",
                              departPos="%.2f" % (i * spacing), departSpeed="8", departLane="0")
            ids.append(vid)
        traci.simulation.step()
        for vid in ids:
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setLaneChangeMode(vid, 0)
        assert len(traci.vehicle.getIDList()) == n

        v_cmd = {i: 8.0 for i in ids}
        last = {i: -9.0 for i in ids}
        ah = {i: 0.0 for i in ids}
        rec = {i: [] for i in ids}
        t_end = t_settle + t_probe
        while traci.simulation.getTime() < t_end:
            t = traci.simulation.getTime()
            # leader profile: sinusoid during probe window
            if t < t_settle:
                lead_speed = 8.0
            else:
                lead_speed = 7.0 + 3.0 * math.sin(2 * math.pi * 0.12 * (t - t_settle))
            for vi, vid in enumerate(ids):
                if vi == 0:
                    v_cmd[vid] = lead_speed
                    traci.vehicle.setSpeed(vid, v_cmd[vid])
                    if t >= t_settle:
                        rec[vid].append(lead_speed)
                    continue
                p = R.AV if av_all else R.HUMAN
                spd = traci.vehicle.getSpeed(vid)
                lead = traci.vehicle.getLeader(vid, 1e4)
                if lead is None:
                    traci.vehicle.setSpeed(vid, v_cmd[vid])
                    if t >= t_settle:
                        rec[vid].append(traci.vehicle.getSpeed(vid))
                    continue
                lid, gap = lead
                lv = traci.vehicle.getSpeed(lid)
                if t - last[vid] >= p["delay"]:
                    dv = spd - lv
                    if "t_close" in p and dv > 0:
                        p_eff = dict(p, T=float(np.clip(p["T"] + p["t_close"] * dv,
                                                        p["T"], p["t_cap"])))
                    else:
                        p_eff = p
                    acc = R.idm_accel(spd, dv, gap, p_eff)
                    if p["noise"]:
                        acc += rng.normal(0, p["noise"])
                    acc = float(np.clip(acc, -p["b_comf"], p["a_max"]))
                    if p["jerk_cap"]:
                        acc = float(np.clip(acc,
                                            ah[vid] - p["jerk_cap"] * p["delay"],
                                            ah[vid] + p["jerk_cap"] * p["delay"]))
                    v_safe = math.sqrt(max(0.0, lv * lv + 2 * 3.0 * (gap - 1.5)))
                    v_cmd[vid] = max(spd - 0.8, max(0.0, min(spd + acc * 0.1, v_safe + 0.3)))
                    ah[vid] = acc
                    last[vid] = t
                traci.vehicle.setSpeed(vid, v_cmd[vid])
                if t >= t_settle:
                    rec[vid].append(traci.vehicle.getSpeed(vid))
            traci.simulation.step()

        gains = {}
        arrs = [np.array(rec[i]) for i in ids]
        base_std = arrs[0].std()
        print(f"leader fluct std: {base_std:.3f} m/s")
        for k in range(1, len(arrs)):
            gains[k] = arrs[k].std() / max(arrs[k - 1].std(), 1e-6)
        return gains
    finally:
        traci.close()


def subprocess_netconvert(net, nod, edg):
    import subprocess
    subprocess.run([os.path.join(R.SUMO_HOME, "bin", "netconvert.exe"),
                    "-n", nod, "-e", edg, "-o", net], check=True)


if __name__ == "__main__":
    for name, av in [("ALL HUMAN", False), ("ALL AV", True)]:
        g = run_platoon(av_all=av, seed=0)
        gs = " ".join(f"{v:.2f}" for v in list(g.values())[:8])
        print(f"{name}: chain gains {gs}")
        print(f"  cumulative at position 8: {list(g.values())[7]:.3f}")
