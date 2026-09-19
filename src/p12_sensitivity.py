"""P1-2: sensitivity sweep of the platoon energy dividend to controller
parameters and wave protocol. 3x3x3 grid (jerk cap, anticipation, headway)
x 3 penetration levels x 3 seeds, plus wave-protocol variants.
"""
import itertools
import os

import pandas as pd

import platoon_energy as PE

PENS = [(12, 20), (30, 50), (60, 100)]   # n_av values for 20/50/100%
GRID = dict(jerk_cap=[1.0, 1.5, 2.0],
            t_close=[0.0, 0.3, 0.55],
            T_av=[1.1, 1.4, 1.7])

rows = []
for jc, tc, T in itertools.product(*GRID.values()):
    PE.AV["jerk_cap"] = jc
    PE.AV["t_close"] = tc
    PE.AV["T"] = T
    for n_av in PENS[0] + PENS[1] + PENS[2]:
        for seed in range(3):
            r = PE.run_platoon(n_av, seed)
            rows.append(dict(jerk_cap=jc, t_close=tc, T_av=T, n_av=n_av,
                             seed=seed, e=r["e_pos_kWh_100km"],
                             vmean=r["throughput_vmean"]))
    print(f"jc={jc} tc={tc} T={T} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("p12_sensitivity.csv", index=False)

base = df[(df.jerk_cap == 1.5) & (df.t_close == 0.3) & (df.T_av == 1.1)]
print("\n=== baseline (jc=1.5, tc=0.3, T=1.1) ===")
for n_av in [12, 30, 60]:
    sub = base[base.n_av == n_av]
    b0 = df[(df.n_av == n_av) & (df.jerk_cap == 1.5) & (df.t_close == 0.0) & (df.T_av == 1.1)]
    print(f"n_av={n_av}: e={sub.e.mean():.2f}  (tc=0 variant: {b0.e.mean():.2f})")

print("\n=== savings range across grid (vs n_av=0 human baseline) ===")
# recompute human baseline per config? human params unchanged -> one baseline
PE.AV["jerk_cap"] = 1.5; PE.AV["t_close"] = 0.3; PE.AV["T"] = 1.1
b = [PE.run_platoon(0, s)["e_pos_kWh_100km"] for s in range(3)]
bmean = sum(b) / 3
print(f"human baseline: {bmean:.2f} kWh/100km")
for n_av, lab in [(12, "20%"), (30, "50%"), (60, "100%")]:
    sub = df[df.n_av == n_av].groupby(["jerk_cap", "t_close", "T_av"]).e.mean()
    sav = 100 * (1 - sub / bmean)
    print(f"p={lab}: saving across grid min={sav.min():.1f}% median={sav.median():.1f}% "
          f"max={sav.max():.1f}%  (n configs={len(sav)})")
