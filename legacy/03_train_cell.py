#!/usr/bin/env python3
"""
03_train_cell.py CELL  —  train ONE cell (resumable), save model + per-cell OOF.
CELL in {R_R, R_L, L_R, L_L}. Writes stuff_model_CELL.txt and oof_CELL.parquet.
Skips if outputs already exist.
"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

OUT="/home/claude/stuff/artifacts"
FEATURES=["start_speed","ivb","hb","rel_side","release_z","extension","spin_rate",
          "velo_diff","ivb_diff","hb_diff","arm_angle","pitch_type_code"]
CAT=["pitch_type_code"]
PARAMS=dict(objective="regression",learning_rate=0.03,num_leaves=31,min_child_samples=300,
            feature_fraction=0.85,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
            seed=42,verbose=-1,deterministic=True,force_row_wise=True)
ROUNDS=700

def main():
    cell=sys.argv[1]; ph,bh=cell.split("_")
    mpath=f"{OUT}/stuff_model_{cell}.txt"; opath=f"{OUT}/oof_{cell}.parquet"
    if os.path.exists(mpath) and os.path.exists(opath):
        print(f"[{cell}] already done, skip"); return
    df=pd.read_parquet(f"{OUT}/features.parquet")
    dfc=df[(df.pitcher_hand==ph)&(df.batter_hand==bh)].copy().reset_index(drop=True)
    print(f"[{cell}] {len(dfc):,} pitches")
    X=dfc[FEATURES]; y=dfc.pitch_rv.values; g=dfc.pitcher_id.values
    oof=np.zeros(len(dfc))
    for k,(tr,va) in enumerate(GroupKFold(5).split(X,y,g)):
        m=lgb.train(PARAMS,lgb.Dataset(X.iloc[tr],label=y[tr],categorical_feature=CAT),num_boost_round=ROUNDS)
        oof[va]=m.predict(X.iloc[va]); print(f"  fold {k+1} done")
    final=lgb.train(PARAMS,lgb.Dataset(X,label=y,categorical_feature=CAT),num_boost_round=ROUNDS)
    final.save_model(mpath)
    dfc=dfc.assign(pred_rv=oof)
    dfc[["pitcher_id","pitch_type","pitch_type_code","pitch_rv","pred_rv"]].to_parquet(opath,index=False)
    print(f"[{cell}] saved model + oof")

if __name__=="__main__": main()
