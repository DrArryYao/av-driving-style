"""Driver for the full-data run: globs tfrecords internally to avoid the
Windows command-line length limit, then runs extraction, string stability
and the paired analysis sequentially."""
import glob
import sys

sys.path.insert(0, r"C:\Users\Arry\Desktop\NC\av_style_pilot\code")
import os
os.chdir(r"C:\Users\Arry\Desktop\NC\av_style_pilot\code")

files = sorted(glob.glob(r"E:/av_style_data/tfrecords/*"))
print(f"found {len(files)} tfrecord files", flush=True)

import extract_kinematics
sys.argv = ["extract", r"E:/av_style_data/tracks_full.csv"] + files
extract_kinematics.main()

import string_stability
sys.argv = ["string", r"E:/av_style_data/pairs_full.csv"] + files
string_stability.main()

import analyze_pilot
sys.argv = ["analyze", r"E:/av_style_data/tracks_full.csv", r"E:/av_style_data/results_full"]
analyze_pilot.main()

print("FULL_RUN_DONE", flush=True)
