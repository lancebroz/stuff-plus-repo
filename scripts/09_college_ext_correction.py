#!/usr/bin/env python3
"""
09_college_ext_correction.py — apply the measured college TrackMan extension
correction to a college pitch-shape table. The spec lives in
data/ext_correction_v2.json.

The correction (fit July 2026):
    add_ft = slope * raw_ext + intercept[bucket],  clamped to [0, 1.2]
    buckets: NB = Fastball/Sinker/Changeup/Splitter
             BB = Curveball/Sweeper
             SL = Slider,  FC = Cutter

Provenance: college TrackMan systematically undercuts extension (Barrand,
May 2026: +0.5 ft non-breaking / +1.0 ft breaking, 32 arms, college TM vs pro
Hawkeye). We re-measured on our own arms: 18 pitchers from the 2025 college
file who reached affiliated ball in the 2026 MiLB sample -> 72 matched
pitcher x pitch-type pairs. Findings: same direction as Barrand but ~2/3 the
magnitude (NB +0.34, BB +0.71, SL +0.55, FC +0.42), AND the gap scales with
extension (slope -0.25 ft/ft, r=-0.49: short-extension arms are undercut
more). The per-type-intercept + shared-slope form fit best under residual
RMSE (0.263 vs 0.309 flat). This replaced the earlier flat Barrand offsets.

The corrected extension is baked into the dashboard's college data (cpd Ext
column) and therefore flows to the shape table, arm-angle computation, and
Stuff+ features. RelHt is left RAW (Barrand's analysis covers extension only).

Usage: correct_ext(pitch_type, raw_ext) -> corrected_ext
Run directly for a demonstration table.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec_path = os.path.join(HERE, "..", "data", "ext_correction_v2.json")
if not os.path.exists(_spec_path): _spec_path = "/home/claude/ext_correction_v2.json"
SPEC = json.load(open(_spec_path))

def bucket(pt):
    if pt in {"Fastball","Sinker","Changeup","Splitter"}: return "NB"
    if pt in {"Curveball","Sweeper"}: return "BB"
    if pt == "Slider": return "SL"
    if pt == "Cutter": return "FC"
    return "NB"

def correct_ext(pitch_type, raw_ext):
    """Return the corrected (MLB-scale) extension for a raw college value."""
    add = SPEC["slope"] * raw_ext + SPEC["intercepts"][bucket(pitch_type)]
    add = max(0.0, min(1.2, add))
    return raw_ext + add

if __name__ == "__main__":
    print(f"spec: add = {SPEC['slope']:+.4f}*raw + intercept  "
          f"(NB {SPEC['intercepts']['NB']:.3f}, BB {SPEC['intercepts']['BB']:.3f}, "
          f"SL {SPEC['intercepts']['SL']:.3f}, FC {SPEC['intercepts']['FC']:.3f})  "
          f"fit n={SPEC['n']}, RMSE {SPEC['rmse']:.3f}")
    print(f"{'pitch':10s} {'raw':>5s} {'corrected':>10s}")
    for pt in ["Fastball","Curveball","Slider","Cutter","Changeup"]:
        for e in (5.5, 6.0, 6.5, 7.0):
            print(f"{pt:10s} {e:>5.1f} {correct_ext(pt,e):>10.2f}")
        print()
