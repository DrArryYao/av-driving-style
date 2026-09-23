"""Extract a per-scenario signal-presence flag (fast pass, no track math).

Writes scenario_id,has_signal CSV for joining with paired analyses.
has_signal = any timestep has at least one traffic-signal lane state.
"""
import glob
import sys

import pandas as pd

from womd_parsing import iter_scenarios

out = sys.argv[1] if len(sys.argv) > 1 else "E:/av_style_data/signal_flags.csv"
files = sorted(glob.glob("E:/av_style_data/tfrecords/*"))

rows = []
for path in files:
    for sc in iter_scenarios(path):
        has = any(len(d.lane_states) > 0 for d in sc.dynamic_map_states)
        rows.append((sc.scenario_id, int(has)))
df = pd.DataFrame(rows, columns=["scenario_id", "has_signal"])
df.to_csv(out, index=False)
print(f"scenarios={len(df)} | with signals={int(df.has_signal.sum())} -> {out}")
