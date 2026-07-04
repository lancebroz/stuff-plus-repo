#!/usr/bin/env python3
"""
02_features.py  —  Stage 02 of the Stuff+ rebuild  (STUFF_PLUS.md Stage 02).

  - Modeled pitch types: FF, SI, SL, CH, ST, FC, CU, FS, KC, SV.
  - Movement -> inches: ivb = pfx_z*12, hb = pfx_x*12.
  - Arm-side-positive frame: lefties mirrored to RHP-equivalent (hb, rel_side
    sign-flipped) so handedness collapses to a same_side flag.
  - Fastball differentials: velo_diff, ivb_diff, hb_diff vs the pitcher's
    primary fastball (most-thrown FF/SI/FC else most-thrown pitch).
  - Arm angle: imputed from a height/release regression (no optical angle in the
    raw feed). Same estimator the dashboard/college path uses, for consistency.

Output: features.parquet
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = "/home/claude/stuff/artifacts"
MODELED = ["FF","SI","SL","CH","ST","FC","CU","FS","KC","SV"]
FB_TYPES = ["FF","SI","FC"]

# Arm-angle imputation regression (dashboard's model, height in FEET).
# arm_angle = 27.27 + 20.1049*rel_z - 23.7934*height - 2.7459*|rel_side| + 7.5368*ext
def impute_arm_angle(rel_z, height_ft, rel_side, ext):
    a = 27.2700 + 20.1049*rel_z - 23.7934*height_ft - 2.7459*np.abs(rel_side) + 7.5368*ext
    return np.clip(a, 0, 90)

def main():
    df = pd.read_parquet(f"{OUT_DIR}/pitches_with_rv.parquet")
    df = df[df["pitch_type"].isin(MODELED)].copy()
    df = df.dropna(subset=["start_speed","pfx_x","pfx_z","release_x","release_z",
                            "extension","spin_rate","pitcher_hand","batter_hand"])

    # movement to inches
    df["ivb"] = df["pfx_z"] * 12.0
    df["hb"]  = df["pfx_x"] * 12.0
    df["rel_side"] = df["release_x"]
    df["release_z"] = df["release_z"]
    df["extension"] = df["extension"]

    # arm-side-positive frame: mirror LHP into RHP-equivalent geometry
    is_lhp = df["pitcher_hand"].eq("L")
    df.loc[is_lhp, "hb"]       *= -1.0
    df.loc[is_lhp, "rel_side"] *= -1.0
    df["same_side"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    # primary fastball per pitcher (most-thrown FB type; else most-thrown pitch)
    counts = df.groupby(["pitcher_id","pitch_type"]).size().rename("n").reset_index()
    def primary(grp):
        fb = grp[grp["pitch_type"].isin(FB_TYPES)]
        pick = (fb if len(fb) else grp).sort_values("n", ascending=False).iloc[0]
        return pick["pitch_type"]
    prim = counts.groupby("pitcher_id").apply(primary).rename("prim_fb").reset_index()
    df = df.merge(prim, on="pitcher_id", how="left")

    # per-pitcher primary-fastball shape (mean velo/ivb/hb of the primary type)
    fb_rows = df[df["pitch_type"] == df["prim_fb"]]
    fb_shape = fb_rows.groupby("pitcher_id").agg(
        fb_velo=("start_speed","mean"), fb_ivb=("ivb","mean"), fb_hb=("hb","mean")
    ).reset_index()
    df = df.merge(fb_shape, on="pitcher_id", how="left")

    df["velo_diff"] = df["start_speed"] - df["fb_velo"]
    df["ivb_diff"]  = df["ivb"]        - df["fb_ivb"]
    df["hb_diff"]   = df["hb"]         - df["fb_hb"]

    # arm angle: impute from release geometry. Height not in feed -> derive an
    # effective height proxy from release_z (shoulder ~ rel_z); dashboard uses a
    # height code, but for MLB training we impute angle from release directly via
    # the same regression using a height proxy = release_z + ~1.6 ft (release pt
    # sits ~ that far below the listed height for a typical delivery).
    height_proxy = df["release_z"] + 1.6
    df["arm_angle"] = impute_arm_angle(df["release_z"], height_proxy,
                                       df["rel_side"], df["extension"])

    # pitch type code (categorical int)
    code_map = {pt:i for i,pt in enumerate(MODELED)}
    df["pitch_type_code"] = df["pitch_type"].map(code_map)

    feats = ["start_speed","ivb","hb","rel_side","release_z","extension","spin_rate",
             "velo_diff","ivb_diff","hb_diff","arm_angle","pitch_type_code"]
    keep = feats + ["pitch_rv","pitcher_id","pitcher_hand","batter_hand","same_side",
                    "pitch_type","season"]
    out = df[keep].dropna(subset=feats).copy()
    out.to_parquet(f"{OUT_DIR}/features.parquet", index=False)
    import json
    json.dump(code_map, open(f"{OUT_DIR}/pitch_type_map.json","w"))
    print(f"[02] wrote features.parquet  ({len(out):,} rows, {len(feats)} features)")
    print("     cells:", out.groupby(['pitcher_hand','batter_hand']).size().to_dict())

if __name__ == "__main__":
    main()
