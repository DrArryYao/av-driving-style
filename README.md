# av-driving-style

Paired analysis of automated-vehicle driving style versus human driving,
from open trajectory data.

This repository contains the **data-processing and figure-generation
code** for a study of how a production automated vehicle (AV) drives
relative to the human-driven vehicles around it, using the paired
structure embedded in the Waymo Open Motion Dataset (in every scene, the
AV shares identical road, weather and traffic conditions with the tracked
human vehicles), cross-validated with the independently instrumented
NGSIM US-101 dataset and compared against commercial adaptive cruise
control (ACC) from the JRC Open ACC Database.

## Repository layout

```
data_processing/    Core pipeline: WOMD parsing -> kinematics/style
                    extraction -> paired statistics -> string-stability
                    analysis -> NGSIM baseline -> JRC ACC baseline
figure_generation/  Manuscript figure scripts (matplotlib), the
                    source-data Excel export, and fix_figures.py
                    (vector-level axis-label/legend/legend revisions
                    applied to the final figure PDFs; its paths refer to
                    the authors' local paper work package)
protos/             Waymo Open Dataset Scenario .proto definitions
gen/                Pre-compiled protobuf modules (import path used by
                    the parsing scripts)
```

## What the pipeline does

1. **Parsing** (`data_processing/womd_parsing.py`) — pure-Python
   TFRecord reader and Waymo `Scenario` protobuf parsing (no TensorFlow
   required).
2. **Style extraction** (`data_processing/extract_kinematics.py`) —
   per-vehicle kinematics (speed, acceleration, jerk), style metrics,
   VSP operating modes, and physical wheel-energy integrals for
   ~965,000 vehicle episodes.
3. **Paired statistics** (`data_processing/analyze_pilot.py`) —
   within-scene paired contrasts of the AV against same-scene human
   vehicles, bootstrap CIs, speed-regime stratification.
4. **String stability** (`data_processing/string_stability.py`) —
   geometric leader–follower identification, fluctuation-gain
   estimation, disturbance-conditional analysis.
5. **Independent human baseline** (`data_processing/ngsim_analysis.py`)
   — NGSIM US-101 processing with identical metric definitions.
6. **Commercial ACC baseline** (`data_processing/jrc_acc_analysis.py`)
   — JRC Open ACC Database processing: parses five-vehicle platoon
   trajectories and computes fluctuation gains for ACC followers using
   the same pipeline as WOMD/NGSIM, enabling cross-automation-level
   comparison (human vs commercial ACC vs L4).
7. **Figures and source data** (`figure_generation/`) — all manuscript
   figures and the figure source-data workbook
   (`export_excel.py`, sheets named by manuscript figure number).

## Data requirements (not included)

- **Waymo Open Motion Dataset v1.3** (20-second training scenarios).
  Obtain from the official source under the Waymo Open Dataset License
  (registration required). Place the 930 uncompressed `.tfrecord`
  shards under `$AVS_DATA_DIR/tfrecords/`.
- **NGSIM US-101** trajectory data (public domain, US FHWA).
- **JRC Open ACC Database** (European Commission Joint Research Centre,
  DOI 10.2905/JRC.KMH3D00).

## Reproduce

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/gen:$PYTHONPATH"   # compiled Scenario protos
python data_processing/run_full.py           # full pipeline -> results
python figure_generation/unified_figs.py     # manuscript figures
python figure_generation/export_excel.py     # figure source-data workbook
```

Paths to the datasets and output directories are configured at the top
of the scripts.

## License

MIT (see LICENSE). The underlying datasets are governed by their own
licenses.
