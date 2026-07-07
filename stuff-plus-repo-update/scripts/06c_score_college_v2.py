#!/usr/bin/env python3
"""
06_score_college.py  —  score the college arms through the rebuilt Stuff+ models.

Per STUFF_PLUS.md Stage 06, adapted for college:
  - Map college pitch types to MLB codes; infer hand from sign of RelSd.
  - Shape adjust: BALL adjustment only (altitude OFF for college). College lacks
    spin efficiency, so an assumed per-pitch-type SpinEff is used; dampen=0.25
    (gentler than MiLB's 0.40, since the coeffs are MiLB-ball-derived and applied
    out-of-domain to the NCAA ball). The primary-FB reference is adjusted too.
  - Build the 12 features in the arm-side-positive frame; arm angle imputed from
    the REAL per-pitcher height (height code) + release geometry.
  - Route to the (pitcher-hand x batter-hand) model -> pred_rv.
    College data has no batter-stance split, so we score vs both stances using
    the pitcher-hand's two cells and report overall + vsR + vsL.
  - Stuff+ = 100 + 10*(stuff_raw - mu)/sigma against the MLB anchors.
  - Aggregate: per-pitch (min 10), usage-weighted overall (min 30).

Output: college_stuff.json  -> {pitcher: {ovr,n,by:{pt:{s,n,R,L}},ovrR,ovrL}}
"""
import os, json, sys
import numpy as np, pandas as pd, lightgbm as lgb
_HERE=os.path.dirname(os.path.abspath(__file__))
_REPO=os.path.dirname(_HERE)
sys.path.insert(0, _HERE)          # milb_shape_adjustments.py lives in scripts/
sys.path.insert(0, "/home/claude") # session-environment fallback
def _resolve(*cands):
    for c in cands:
        if c and os.path.exists(c): return c
    return cands[-1]
from milb_shape_adjustments import ball_adjust, TYPE_MAP

ART = _resolve(os.path.join(_REPO,"models"), "/home/claude/stuff/artifacts")
MODELED = ["FF","SI","SL","CH","ST","FC","CU","FS","KC","SV"]
FB_TYPES = ["FF","SI","FC"]
FEATURES = ["start_speed","ivb","hb","rel_side","release_z","extension","spin_rate",
            "velo_diff","ivb_diff","hb_diff","arm_angle","pitch_type_code","same_side"]

# college label -> MLB code
PT_COLLEGE = {"Fastball":"FF","Sinker":"SI","Slider":"SL","Sweeper":"ST","Cutter":"FC",
              "Curveball":"CU","Changeup":"CH","Splitter":"FS"}
# assumed spin efficiency (whole %) per type — ball-adj is near-insensitive to this
ASSUMED_SE = {"Fastball":92.0,"Sinker":88.0,"Slider":35.0,"Sweeper":40.0,
              "Cutter":55.0,"Curveball":70.0,"Changeup":88.0,"Splitter":50.0}
DAMPEN = 0.25
AA_OVERRIDES={"Mendes, Wes":40.0,"Renfrow, Brett":50.0}
# Out-of-domain guards: clip FB differentials to the MLB-supported range (p2-p98),
# then Monte-Carlo smooth each grade over input measurement uncertainty so tree
# cliffs cannot swing a grade on half a mph.
_fb_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","feature_bounds.json")
if not os.path.exists(_fb_path): _fb_path="/home/claude/feature_bounds.json"
FEATURE_BOUNDS=json.load(open(_fb_path))
MC_SIG={"start_speed":0.4,"ivb":0.8,"hb":0.8,"rel_side":0.06,"release_z":0.06,
        "extension":0.10,"spin_rate":30.0,"velo_diff":0.5,"ivb_diff":1.0,
        "hb_diff":1.0,"arm_angle":1.5}
MC_K=64
_RNG=np.random.default_rng(42)

def decode_height(code):
    if code is None: return None
    c = int(round(code)); return c//100 + (c%100)/12.0

def impute_arm_angle(rel_z, height_ft, rel_side, ext):
    a = 10.8506 + 20.7891*rel_z + (-22.3478)*height_ft + (-3.8044)*abs(rel_side) + 8.5466*ext
    return float(np.clip(a, -90, 90))

def main():
    col = json.load(open(_resolve(os.path.join(_REPO,"data","college_data.json"),
                                   "college_data.json","/home/claude/college_data.json")))
    H = col["cph"]; ix = {h:i for i,h in enumerate(H)}
    code_map = json.load(open(f"{ART}/pitch_type_map.json"))
    anchors = json.load(open(f"{ART}/anchors_v2.json"))
    models = {hd: lgb.Booster(model_file=f"{ART}/stuff_hand_{hd}.txt") for hd in ["R","L"]}

    rows = col["cpd"]
    # group by pitcher
    bypit = {}
    for r in rows:
        bypit.setdefault(r[ix["Pitcher"]], []).append(r)

    def g(r, k):
        v = r[ix[k]]; return v if isinstance(v,(int,float)) else None

    out = {}
    for name, prs in bypit.items():
        # infer hand from mean RelSd sign (neg = RHP in this data's convention)
        relsd = np.mean([g(r,"RelSd") for r in prs if g(r,"RelSd") is not None] or [0])
        hand = "R" if relsd < 0 else "L"
        height_ft = decode_height(g(prs[0],"height"))

        # primary fastball (most-thrown FB type, else most-thrown)
        usage = {}
        for r in prs:
            pt = r[ix["PITCH_TYPE"]]; usage[pt] = usage.get(pt,0) + (g(r,"Pitches") or 0)
        fb_cands = {pt:n for pt,n in usage.items() if PT_COLLEGE.get(pt) in FB_TYPES}
        prim_label = max(fb_cands or usage, key=(fb_cands or usage).get)

        # adjusted shape per pitch type (ball adj, altitude off)
        def adj_shape(r):
            pt = r[ix["PITCH_TYPE"]]; vm = g(r,"VM"); hm = g(r,"HM")
            if vm is None or hm is None: return None,None
            se = ASSUMED_SE.get(pt, 60.0)
            return ball_adjust(pt, vm, hm, se, below_aaa=True, dampen=DAMPEN)

        prim_rows = [r for r in prs if r[ix["PITCH_TYPE"]]==prim_label]
        if prim_rows:
            pv = np.mean([g(r,"Velo") for r in prim_rows if g(r,"Velo") is not None])
            avm_hm = [adj_shape(r) for r in prim_rows]
            pivb = np.mean([a for a,_ in avm_hm if a is not None])
            phb  = np.mean([b for _,b in avm_hm if b is not None])
        else:
            pv = pivb = phb = None

        by = {}
        for pt, n in usage.items():
            code = PT_COLLEGE.get(pt)
            if code not in MODELED or n < 10: continue
            prows = [r for r in prs if r[ix["PITCH_TYPE"]]==pt]
            velo = np.mean([g(r,"Velo") for r in prows if g(r,"Velo") is not None])
            adj = [adj_shape(r) for r in prows]
            ivb = np.mean([a for a,_ in adj if a is not None])
            hb  = np.mean([b for _,b in adj if b is not None])
            rel_side = np.mean([g(r,"RelSd") for r in prows if g(r,"RelSd") is not None])
            rel_z    = np.mean([g(r,"RelHt") for r in prows if g(r,"RelHt") is not None])
            ext      = np.mean([g(r,"Ext")   for r in prows if g(r,"Ext")   is not None])
            spin     = np.mean([g(r,"Spin")  for r in prows if g(r,"Spin")  is not None])
            if any(v is None or (isinstance(v,float) and np.isnan(v))
                   for v in [velo,ivb,hb,rel_side,rel_z,ext,spin]): continue
            # arm-side-positive frame: mirror LHP
            hb_f = -hb if hand=="L" else hb
            rels_f = -rel_side if hand=="L" else rel_side
            arm = AA_OVERRIDES.get(name, impute_arm_angle(rel_z, height_ft if height_ft else rel_z+1.6, rels_f, ext))
            vdiff = velo - pv if pv else 0.0
            ivbdiff = ivb - pivb if pivb is not None else 0.0
            hbdiff = (hb_f - (-phb if hand=="L" else phb)) if phb is not None else 0.0
            # clip differentials into MLB-supported territory
            fb_bounds=FEATURE_BOUNDS.get(code,{})
            def _clip(v,key):
                b=fb_bounds.get(key)
                return min(max(v,b[0]),b[1]) if b else v
            vdiff=_clip(vdiff,"velo_diff"); ivbdiff=_clip(ivbdiff,"ivb_diff"); hbdiff=_clip(hbdiff,"hb_diff")
            grades = {}
            for bh in ["R","L"]:
                same = 1 if bh==hand else 0
                base=[velo,ivb,hb_f,rels_f,rel_z,ext,spin,vdiff,ivbdiff,hbdiff,arm,code_map[code],same]
                feat = pd.DataFrame([base]*MC_K, columns=FEATURES)
                for col_,s_ in MC_SIG.items():
                    feat[col_]=feat[col_]+_RNG.normal(0,s_,MC_K)
                pred = float(np.mean(models[hand].predict(feat)))
                a = anchors.get(f"{hand}_{bh}",{}).get(code)
                if not a or a["sigma"]==0: grades[bh]=None; continue
                grades[bh] = 100 + 10*((-pred) - a["mu"])/a["sigma"]
            # overall for the pitch = avg of vsR/vsL weighted by MLB platoon exposure (~.55/.45 for this hand)
            wR, wL = (0.55,0.45) if hand=="R" else (0.45,0.55)
            gs = None
            if grades["R"] is not None and grades["L"] is not None:
                gs = wR*grades["R"] + wL*grades["L"]
            elif grades["R"] is not None: gs = grades["R"]
            elif grades["L"] is not None: gs = grades["L"]
            by[pt] = {"s": round(gs) if gs is not None else None,
                      "n": int(n),
                      "R": round(grades["R"]) if grades["R"] is not None else None,
                      "L": round(grades["L"]) if grades["L"] is not None else None}
        if not by: continue
        tot = sum(v["n"] for v in by.values())
        def wavg(key):
            num=den=0
            for v in by.values():
                if v[key] is not None: num+=v[key]*v["n"]; den+=v["n"]
            return round(num/den) if den else None
        ovr = wavg("s") if tot>=30 else None
        out[name] = {"ovr": ovr, "n": tot, "by": by,
                     "ovrR": wavg("R"), "ovrL": wavg("L"), "hand": hand}
    _outdir=os.path.join(_REPO,"grades")
    _out=os.path.join(_outdir,"college_stuff_v2.json") if os.path.isdir(_outdir) else "/home/claude/college_stuff_v2.json"
    json.dump(out, open(_out,"w"))
    print(f"[06] wrote {_out}")
    graded = [v["ovr"] for v in out.values() if v["ovr"] is not None]
    print(f"[06] scored {len(out)} college arms; {len(graded)} with overall grade")
    print(f"     overall Stuff+ range {min(graded)}-{max(graded)}, mean {np.mean(graded):.1f}")
    # top 10
    top = sorted([(v['ovr'],k) for k,v in out.items() if v['ovr'] is not None], reverse=True)[:10]
    for s,k in top: print(f"     {s}  {k}  ({out[k]['hand']}HP)")

if __name__ == "__main__":
    main()
