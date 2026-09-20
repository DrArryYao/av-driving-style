# av-driving-style

Paired analysis of automated-vehicle driving style versus human driving,
from open trajectory data.

This repository contains the full analysis pipeline behind a study of how
a production automated vehicle (AV) drives relative to the human-driven
vehicles around it, using the paired structure embedded in the Waymo Open
Motion Dataset (in every scene, the AV shares identical road, weather and
traffic conditions with the tracked human vehicles), cross-validated with
the independently instrumented NGSIM US-101 dataset, compared against
commercial adaptive cruise control (ACC) from the JRC Open ACC Database,
plus calibrated mixed-traffic simulations.

## What the pipeline does

1. **Parsing** (`src/womd_parsing.py`) — pure-Python TFRecord reader and
   Waymo `Scenario` protobuf parsing (no TensorFlow required).
2. **Style extraction** (`src/extract_kinematics.py`) — per-vehicle
   kinematics (speed, acceleration, jerk), style metrics, VSP operating
   modes, and physical wheel-energy integrals for ~965,000 vehicle
   episodes.
3. **Paired statistics** (`src/analyze_pilot.py`) — within-scene paired
   contrasts of the AV against same-scene human vehicles, bootstrap CIs,
   speed-regime stratification.
4. **String stability** (`src/string_stability.py`) — geometric
   leader–follower identification, fluctuation-gain estimation,
   disturbance-conditional analysis.
5. **Independent human baseline** (`src/ngsim_analysis.py`) — NGSIM
   US-101 processing with identical metric definitions.
6. **Commercial ACC baseline** (`src/jrc_acc_analysis.py`) — JRC Open
   ACC Database processing: parses five-vehicle platoon trajectories,
   computes fluctuation gains for ACC followers using the same pipeline
   as WOMD/NGSIM, enabling cross-automation-level comparison
   (human vs commercial ACC vs L4).
7. **Robustness suite** (`src/p0*.py`, `src/p1*.py`) — estimator-bias
   null simulations, leader-noise equalization, cluster bootstrap,
   controller-parameter sensitivity sweeps, duration matching.
8. **Simulations** (`src/platoon_energy.py`, `src/platoon_probe.py`,
   `src/ring_sim.py`) — controller validation and penetration-sweep
   energy counterfactuals (SUMO optional; platoon model is pure Python).
9. **Figures** (`figures/make_figures.py`, `figures/fig2_9panel.py`,
   `figures/fig1_diverse.py`, etc.) — all manuscript figures.

## Data requirements (not included)

- **Waymo Open Motion Dataset v1.3** (20-second training scenarios).
  Obtain from the official source under the Waymo Open Dataset License
  (registration required). Place the 930 uncompressed `.tfrecord` shards
  under `$AVS_DATA_DIR/tfrecords/`.
- **NGSIM US-101 vehicle trajectories** (public). Download the
  passenger-car CSV from the US DOT open data portal
  (`data.transportation.gov`, dataset `8ect-6jqj`) and save as
  `$AVS_DATA_DIR/ngsim_us101_cars.csv`.
- **JRC Open ACC Database** (public). Download car-following trajectory
  CSVs from the European Commission JRC open data portal
  (`data.jrc.ec.europa.eu`, DOI: `10.2905/JRC.KMH3D00`) and save under
  `$AVS_DATA_DIR/jrc_acc/`. Contains five-vehicle platoon experiments
  with production ACC systems on Italian roads.

## Setup

```bash
pip install -r requirements.txt
export AVS_DATA_DIR=/path/to/data      # default ./data
export OUT_DIR=/path/to/outputs        # default ./out
```

Protocol-buffer modules are bundled in `gen/`; to regenerate them from
`protos/` run:

```bash
python -m grpc_tools.protoc -Iprotos --python_out=gen protos/waymo_open_dataset/protos/scenario.proto
```

## Reproducing the analysis

```bash
cd src

# Main analysis (WOMD)
python run_full.py

# Independent baselines
python ngsim_analysis.py $AVS_DATA_DIR/ngsim_us101_cars.csv \
  $AVS_DATA_DIR/ngsim_tracks.csv $AVS_DATA_DIR/ngsim_pairs.csv

python jrc_acc_analysis.py  # outputs jrc_acc_gains.csv

# Robustness suite
python p02_gain_bias.py                 # estimator-bias calibration
python p13_cluster_bootstrap.py         # shard-cluster robustness
python p01_noise_matched.py 300 $AVS_DATA_DIR/p01_pairs.csv
python p12_sensitivity.py               # controller grid

# Figures
cd ../figures && python make_figures.py
```

## Repository layout

```
src/        analysis pipeline (see list above)
figures/    figure-generation scripts
protos/     Waymo Scenario .proto definitions
gen/        compiled protobuf modules
docs/       data-acquisition notes
```

## License

MIT (code). The underlying datasets are governed by their own licenses:
Waymo Open Dataset License; NGSIM is public US Department of
Transportation data; JRC Open ACC Database is European Commission reuse
with attribution.
