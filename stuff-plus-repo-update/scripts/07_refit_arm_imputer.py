#!/usr/bin/env python3
"""
07_refit_arm_imputer.py — refit the arm-angle imputation regression against
Savant optical arm angles. Produces models/imputer_v2.json.

Inputs (repo-relative, with /home/claude fallbacks):
  data/20{24,25,26} arm angle.csv   Savant per-pitcher x pitch-type exports
                                    (must include arm_angle, release_pos_z,
                                    release_pos_x, release_extension, pitches)
  data/mlbam_height.json            MLBAM id -> height in inches
                                    (Chadwick register + Baseball Databank join)

Method: pitcher-season aggregates (pitch-weighted) of optical arm angle and
release geometry, joined to height; OLS of
    aa ~ release_z + height_ft + |release_x| + extension
The July 2026 fit: n=1,447 pitcher-seasons, R^2 = 0.82, RMSE 5.8 deg.
(Original deploy fit on 1,025 rows gave the coefficients in imputer_v2.json;
re-running on updated CSVs will shift coefficients slightly - that is expected.)

The output coefficients must be kept in sync in THREE places:
  - models/imputer_v2.json (this file's output)
  - scripts/06b_score_milb.py / 06c_score_college_v2.py (impute_arm_angle)
  - the dashboard's computeArmAngle() in index.html
"""
import glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def find(rel, fallback):
    p = os.path.join(HERE, "..", rel)
    return p if os.path.exists(p) else fallback

def main():
    csvs = glob.glob(find("data", "/home/claude/armangles") + "/*arm angle.csv")
    assert csvs, "no arm angle CSVs found"
    frames = []
    for f in csvs:
        d = pd.read_csv(f); d["season"] = os.path.basename(f)[:4]
        frames.append(d)
    aa = pd.concat(frames, ignore_index=True)
    aa["arm_angle"] = pd.to_numeric(aa["arm_angle"], errors="coerce")
    aa = aa.dropna(subset=["arm_angle"])

    hmap = json.load(open(find("data/mlbam_height.json", "/home/claude/mlbam_height.json")))
    aa["height_in"] = aa.player_id.map(lambda i: hmap.get(str(int(i))))

    def wavg(g, c): return np.average(g[c], weights=g["pitches"])
    ps = (aa.dropna(subset=["height_in","release_pos_z","release_pos_x","release_extension"])
            .groupby(["player_id","season"])
            .apply(lambda g: pd.Series({
                "aa": wavg(g,"arm_angle"), "relz": wavg(g,"release_pos_z"),
                "relx": abs(wavg(g,"release_pos_x")), "ext": wavg(g,"release_extension"),
                "hft": g.height_in.iloc[0]/12.0}))
            .reset_index())
    print(f"fit sample: {len(ps)} pitcher-seasons")

    X = np.column_stack([ps.relz, ps.hft, ps.relx, ps.ext, np.ones(len(ps))])
    coef, _, _, _ = np.linalg.lstsq(X, ps.aa.values, rcond=None)
    pred = X @ coef
    resid = ps.aa.values - pred
    r2 = 1 - (resid**2).sum() / ((ps.aa - ps.aa.mean())**2).sum()
    rmse = float(np.sqrt((resid**2).mean()))
    out = {"intercept": float(coef[4]), "rel_z": float(coef[0]),
           "height_ft": float(coef[1]), "abs_rel_side": float(coef[2]),
           "ext": float(coef[3]), "r2": float(r2), "rmse": rmse, "n": len(ps)}
    dest = find("models/imputer_v2.json", "/home/claude/imputer_v2.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"aa = {out['intercept']:.4f} + {out['rel_z']:.4f}*relZ "
          f"{out['height_ft']:+.4f}*heightFt {out['abs_rel_side']:+.4f}*|relSide| "
          f"{out['ext']:+.4f}*ext   R2={r2:.3f} RMSE={rmse:.2f}")
    print(f"wrote {dest}")
    print("NOTE (validated July 2026): height-segmented 'mini models', interaction "
          "terms, and GBMs all tested WORSE under pitcher-grouped CV - the linear "
          "form is at the information ceiling of these inputs. Known-off arms are "
          "handled by scout overrides (AA_OVERRIDES), not model changes.")

if __name__ == "__main__":
    main()
