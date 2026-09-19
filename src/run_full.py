"""Driver for the full-data run: globs tfrecords internally to avoid the
Windows command-line length limit, then runs extraction, string stability
and the paired analysis sequentially."""
import glob
import sys

sys.path.insert(0, r"${CODE_DIR}")
import os
os.chdir(r"${CODE_DIR}")

files = sorted(glob.glob(r"${DATA_DIR}/tfrecords/*"))
print(f"found {len(files)} tfrecord files", flush=True)

import extract_kinematics
sys.argv = ["extract", r"${DATA_DIR}/tracks_full.csv"] + files
extract_kinematics.main()

import string_stability
sys.argv = ["string", r"${DATA_DIR}/pairs_full.csv"] + files
string_stability.main()

import analyze_pilot
sys.argv = ["analyze", r"${DATA_DIR}/tracks_full.csv", r"${DATA_DIR}/results_full"]
analyze_pilot.main()

print("FULL_RUN_DONE", flush=True)
