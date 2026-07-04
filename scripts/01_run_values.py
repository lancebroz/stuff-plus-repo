#!/usr/bin/env python3
"""
01_run_values.py  —  Stage 01 of the Stuff+ rebuild.

Prices every pitch by delta run expectancy with contact-quality outcomes,
per STUFF_PLUS.md Stage 01:

  - Count RE matrix rebuilt by walking each PA (handles 2-strike fouls).
  - Terminal pitch:     terminal_event_value - RE(count_before)
  - Non-terminal pitch: RE(next_count)      - RE(count_before)
  - Sign: pitcher side, NEGATIVE = runs prevented = good.
  - Balls in play valued by EXPECTED run value from launch speed+angle
    (a LightGBM EV/LA -> event_rv model), not the actual single/out.
  - K/BB/HBP keep fixed (count-anchored delta) values via the telescoping.

Output: pitches_with_rv.parquet   (one row per pitch, adds pitch_rv)
"""
import glob, os, re, json
import numpy as np
import pandas as pd
import lightgbm as lgb

RAW_DIR   = "/home/claude/mlbdata/raw"
OUT_DIR   = "/home/claude/stuff/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_YEARS = {"2023", "2024", "2025"}   # MLB training pool (2026 is partial, held out of training)

def _year(f):
    m = re.search(r"raw__(\d{4})__", f); return m.group(1) if m else "?"

def load_raw():
    files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
    files = [f for f in files if _year(f) in TRAIN_YEARS]
    cols = ["game_pk","at_bat_number","pitch_number","balls","strikes","count","outs",
            "pitch_type","pitcher_id","pitcher_name","pitcher_hand","batter_hand","stand",
            "events","call_description","is_in_play","launch_speed","launch_angle",
            "start_speed","pfx_x","pfx_z","release_x","release_z","extension","spin_rate",
            "game_date"]
    dfs = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d["season"] = _year(f)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df

# ---------------------------------------------------------------------------
# Linear weights for terminal events (runs, league-ish constants on RV scale).
# These value the *event*; the count-RE telescoping turns them into pitch RV.
# ---------------------------------------------------------------------------
EVENT_RUNS = {
    "strikeout": -0.27, "strikeout_double_play": -0.27,
    "walk": 0.32, "intent_walk": 0.32, "hit_by_pitch": 0.34,
    "single": 0.47, "double": 0.77, "triple": 1.05, "home_run": 1.37,
    "field_out": -0.27, "force_out": -0.27, "grounded_into_double_play": -0.45,
    "double_play": -0.45, "fielders_choice_out": -0.27, "fielders_choice": -0.27,
    "sac_fly": -0.20, "sac_bunt": -0.21, "field_error": 0.30,
    "catcher_interf": 0.32, "strikeout_triple_play": -0.27,
}
BIP_EVENTS = {"single","double","triple","home_run","field_out","force_out",
              "grounded_into_double_play","double_play","fielders_choice",
              "fielders_choice_out","sac_fly","sac_bunt","field_error"}

def build_ev_la_model(df):
    """LightGBM EV/LA -> event run value, trained on balls in play with launch data.
    Gives each BIP its expected (contact-quality) run value."""
    bip = df[(df["is_in_play"] == 1) & df["events"].isin(BIP_EVENTS)
             & df["launch_speed"].notna() & df["launch_angle"].notna()].copy()
    bip["ev_rv"] = bip["events"].map(EVENT_RUNS).astype(float)
    X = bip[["launch_speed","launch_angle"]].values
    y = bip["ev_rv"].values
    dtrain = lgb.Dataset(X, label=y)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=31,
                  min_child_samples=200, feature_fraction=1.0, verbose=-1)
    model = lgb.train(params, dtrain, num_boost_round=300)
    model.save_model(f"{OUT_DIR}/ev_la_rv.txt")
    return model

def count_re_matrix(df):
    """Empirical run value by count: mean pitch-level contribution proxy.
    We build RE(count) as the mean end-of-PA event runs over PAs passing through
    that count, which gives a monotone count ladder used for telescoping deltas."""
    d = df.copy()
    d["ev_runs"] = d["events"].map(EVENT_RUNS)
    # terminal row per PA carries the event; propagate to all pitches in the PA
    term = d.dropna(subset=["events"]).groupby(["game_pk","at_bat_number"])["ev_runs"].last()
    d = d.join(term.rename("pa_runs"), on=["game_pk","at_bat_number"])
    re_by_count = d.groupby("count")["pa_runs"].mean()
    return re_by_count.to_dict()

def main():
    print("[01] loading raw …")
    df = load_raw()
    print(f"     {len(df):,} pitches  ({df['season'].value_counts().to_dict()})")

    print("[01] EV/LA contact-quality model …")
    ev_model = build_ev_la_model(df)

    print("[01] count RE matrix …")
    re_count = count_re_matrix(df)
    json.dump(re_count, open(f"{OUT_DIR}/re_by_count.json","w"))

    # Expected run value for each pitch's terminal event (contact quality for BIP)
    df = df.sort_values(["game_pk","at_bat_number","pitch_number"]).reset_index(drop=True)
    df["re_before"] = df["count"].map(re_count)

    # next count within the same PA (for non-terminal telescoping)
    df["next_count"] = df.groupby(["game_pk","at_bat_number"])["count"].shift(-1)
    df["re_next"] = df["next_count"].map(re_count)

    is_terminal = df["events"].notna()

    # terminal event value: contact-quality for BIP, else linear weight
    ev_rv = df["events"].map(EVENT_RUNS).astype(float)
    bip_mask = df["is_in_play"].eq(1) & df["events"].isin(BIP_EVENTS) \
               & df["launch_speed"].notna() & df["launch_angle"].notna()
    if bip_mask.any():
        Xb = df.loc[bip_mask, ["launch_speed","launch_angle"]].values
        ev_rv.loc[bip_mask] = ev_model.predict(Xb)

    # delta-RE, pitcher side (negative = good)
    pitch_rv = np.where(is_terminal,
                        ev_rv - df["re_before"],
                        df["re_next"] - df["re_before"])
    # non-terminal with no next count (data edge) -> 0 contribution
    pitch_rv = np.where(~is_terminal & df["re_next"].isna(), 0.0, pitch_rv)
    df["pitch_rv"] = pitch_rv.astype(float)

    keep = ["season","game_pk","at_bat_number","pitch_number","count","outs",
            "pitch_type","pitcher_id","pitcher_name","pitcher_hand","batter_hand","stand",
            "start_speed","pfx_x","pfx_z","release_x","release_z","extension","spin_rate",
            "pitch_rv"]
    out = df[keep].copy()
    out.to_parquet(f"{OUT_DIR}/pitches_with_rv.parquet", index=False)
    print(f"[01] wrote pitches_with_rv.parquet  ({len(out):,} rows)")
    print(f"     pitch_rv mean={out.pitch_rv.mean():.4f}  K≈{EVENT_RUNS['strikeout']}  HR≈{EVENT_RUNS['home_run']}")

if __name__ == "__main__":
    main()
