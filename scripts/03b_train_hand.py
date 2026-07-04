#!/usr/bin/env python3
"""
03b_train_hand.py — retrain as TWO pitcher-hand models with batter stance as a
feature (PitchSim-style platoon handling; adopted to shrink spurious R/L splits).

  - R model: all RHP pitches (former R_R + R_L cells), L model likewise.
  - 13 features = the original 12 + same_side (batter same-handed as pitcher).
    Within a hand model, same_side maps 1:1 to batter stance, so the platoon
    effect is learned inside ONE function instead of two independent models.
  - Same hyperparameters, 700 rounds, fixed seed, 5-fold GroupKFold by pitcher
    -> leakage-free OOF.
  - Anchors: still per (pitcher-hand x batter-hand x pitch type) from the OOF,
    so the Stuff+ scale definition is unchanged.

Outputs: stuff_hand_R.txt, stuff_hand_L.txt, oof_hand_{R,L}.parquet, anchors_v2.json
"""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

ART = "/home/claude/stuff/artifacts"
FEATURES = ["start_speed","ivb","hb","rel_side","release_z","extension","spin_rate",
            "velo_diff","ivb_diff","hb_diff","arm_angle","pitch_type_code","same_side"]
CAT = ["pitch_type_code","same_side"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31,
              min_child_samples=300, feature_fraction=0.85, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1,
              deterministic=True, force_row_wise=True)
ROUNDS = 700

def train_hand(df, hand):
    d = df[df.pitcher_hand == hand].copy().reset_index(drop=True)
    print(f"[{hand}] {len(d):,} pitches, {d.pitcher_id.nunique()} pitchers", flush=True)
    X = d[FEATURES]; y = d.pitch_rv.values; g = d.pitcher_id.values
    oof = np.zeros(len(d))
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, g)):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=CAT),
                      num_boost_round=ROUNDS)
        oof[va] = m.predict(X.iloc[va]); print(f"  [{hand}] fold {k+1} done", flush=True)
    final = lgb.train(PARAMS, lgb.Dataset(X, label=y, categorical_feature=CAT),
                      num_boost_round=ROUNDS)
    final.save_model(f"{ART}/stuff_hand_{hand}.txt")
    d = d.assign(pred_rv=oof)
    d[["pitcher_id","pitch_type","pitch_type_code","batter_hand","pitch_rv","pred_rv"]] \
        .to_parquet(f"{ART}/oof_hand_{hand}.parquet", index=False)
    print(f"[{hand}] saved", flush=True)
    return d

def main():
    df = pd.read_parquet(f"{ART}/features.parquet")
    frames = []
    for hand in ["L","R"]:  # small first
        frames.append(train_hand(df, hand).assign(pitcher_hand=hand))
    alld = pd.concat(frames, ignore_index=True)
    # anchors per cell x pitch type from OOF (stuff_raw = -pred_rv)
    alld["stuff_raw"] = -alld["pred_rv"]
    inv = {v:k for k,v in json.load(open(f"{ART}/pitch_type_map.json")).items()}
    anchors = {}
    for (ph, bh), grp in alld.groupby(["pitcher_hand","batter_hand"]):
        cell = f"{ph}_{bh}"; anchors[cell] = {}
        for code, g2 in grp.groupby("pitch_type_code"):
            pt = inv[int(code)]
            anchors[cell][pt] = {"mu": float(g2.stuff_raw.mean()),
                                 "sigma": float(g2.stuff_raw.std(ddof=0)),
                                 "n": int(len(g2))}
    json.dump(anchors, open(f"{ART}/anchors_v2.json","w"), indent=1)
    print("[done] anchors_v2.json written", flush=True)

if __name__ == "__main__":
    main()
