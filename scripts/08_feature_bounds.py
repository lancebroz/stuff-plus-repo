#!/usr/bin/env python3
"""
08_feature_bounds.py — derive the MLB-supported range (p2-p98) of the fastball
differentials per pitch type, used by the college scorer to clip out-of-domain
shapes before prediction. Produces data/feature_bounds.json.

Source: a representative slice of raw Statcast parquets from the
mlb-pitcher-data repo (three 2024 monthly files by default - the bounds are
distributional and insensitive to the exact months chosen).

Why this exists: LightGBM trees cannot extrapolate; scoring a college pitch
whose differentials sit outside the training distribution rides whatever the
outermost leaf says. Clipping to p2-p98 keeps predictions in supported
territory. (Note: the clip alone does NOT fix within-range tree cliffs - the
Monte Carlo smoothing in 06c handles those. Both guards were added July 2026
after the Kuhns curveball case: a 12-point grade cliff across 0.5 mph of velo
separation, well inside the nominal range.)
"""
import glob, json, os, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE_DIR = "/home/claude/mlbslice"
FILES = ["04_april.parquet", "06_june.parquet", "08_august.parquet"]
BASE = "https://raw.githubusercontent.com/lancebroz/mlb-pitcher-data/main/data/raw/2024/monthly/"
FB = ["FF", "SI", "FC"]

def main():
    os.makedirs(SLICE_DIR, exist_ok=True)
    for f in FILES:
        dest = os.path.join(SLICE_DIR, f)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(BASE + f, dest); print("downloaded", f)
    cols = ["pitcher_id","pitcher_hand","pitch_type","start_speed","pfx_x","pfx_z"]
    df = pd.concat([pd.read_parquet(f, columns=cols)
                    for f in glob.glob(SLICE_DIR + "/*.parquet")], ignore_index=True).dropna()
    df["ivb"] = df.pfx_z * 12; df["hb"] = df.pfx_x * 12
    df.loc[df.pitcher_hand == "L", "hb"] *= -1  # arm-side-positive frame

    cnt = df.groupby(["pitcher_id","pitch_type"]).size().rename("n").reset_index()
    def prim(g):
        fb = g[g.pitch_type.isin(FB)]
        return (fb if len(fb) else g).sort_values("n", ascending=False).iloc[0].pitch_type
    pr = cnt.groupby("pitcher_id").apply(prim).rename("prim").reset_index()
    df = df.merge(pr, on="pitcher_id")
    fbs = df[df.pitch_type == df.prim].groupby("pitcher_id").agg(
        fv=("start_speed","mean"), fi=("ivb","mean"), fh=("hb","mean")).reset_index()
    df = df.merge(fbs, on="pitcher_id")
    df["velo_diff"] = df.start_speed - df.fv
    df["ivb_diff"]  = df.ivb - df.fi
    df["hb_diff"]   = df.hb - df.fh

    bounds = {}
    for pt, g in df.groupby("pitch_type"):
        if len(g) < 3000: continue
        bounds[pt] = {c: [float(g[c].quantile(.02)), float(g[c].quantile(.98))]
                      for c in ["velo_diff","ivb_diff","hb_diff"]}
    dest = os.path.join(HERE, "..", "data", "feature_bounds.json")
    if not os.path.isdir(os.path.dirname(dest)): dest = "/home/claude/feature_bounds.json"
    json.dump(bounds, open(dest, "w"), indent=1)
    print(f"wrote {dest}  ({len(bounds)} pitch types)")

if __name__ == "__main__":
    main()
