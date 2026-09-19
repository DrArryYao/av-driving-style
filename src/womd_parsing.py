"""Minimal pure-python reader for TFRecord files + Waymo Scenario protos.

TFRecord framing (no CRC verification):
  [8B little-endian length][4B masked crc32c of length][payload][4B masked crc32c of payload]
"""
import struct

import sys
import os

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen")
if GEN_DIR not in sys.path:
    sys.path.insert(0, GEN_DIR)

import waymo_open_dataset.protos.scenario_pb2 as scenario_pb2


def iter_tfrecord(path):
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            (length,) = struct.unpack("<Q", hdr)
            f.read(4)  # length crc, skipped
            payload = f.read(length)
            f.read(4)  # payload crc, skipped
            if len(payload) < length:
                return  # truncated file tail
            yield payload


def iter_scenarios(tfrecord_path, limit=None):
    n = 0
    for payload in iter_tfrecord(tfrecord_path):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(payload)
        yield sc
        n += 1
        if limit is not None and n >= limit:
            return


if __name__ == "__main__":
    # smoke test: parse first scenario of each given file and print a summary
    import glob

    files = []
    for pat in sys.argv[1:]:
        files.extend(glob.glob(pat))
    for path in files[:3]:
        for sc in iter_scenarios(path, limit=1):
            n_veh = sum(
                1 for t in sc.tracks if t.object_type == scenario_pb2.Track.TYPE_VEHICLE
            )
            valid_states = sum(
                1 for s in sc.tracks[sc.sdc_track_index].states if s.valid
            )
            print(
                f"{os.path.basename(path)}: id={sc.scenario_id} steps={len(sc.timestamps_seconds)} "
                f"tracks={len(sc.tracks)} vehicles={n_veh} sdc_index={sc.sdc_track_index} "
                f"sdc_valid_states={valid_states}"
            )
