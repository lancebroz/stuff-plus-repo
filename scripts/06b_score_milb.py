#!/usr/bin/env python3
"""
06b_score_milb.py — score the MiLB board through the two-hand-model pipeline.

Per STUFF_PLUS.md stage 06:
  - shapes: DATA.p26d (ALL rows) + DATA.bd (per-stance rows), real SpinEff.
  - hand from sign of pitch-weighted RelSd (neg = RHP, matches dashboard isRHP).
  - ball adjustment at dampen=0.40 with real SpinEff.
    LEVEL FALLBACK: traditional_stats.json unavailable in this environment ->
    every pitcher treated below-AAA (doc's default for unmatched). Altitude off.
    If /home/claude/traditional_stats.json exists it is used for the AAA gate
    and IP-weighted altitude factors automatically.
  - features in arm-side-positive frame; arm angle imputed from DATA.heights.
  - vsR / vsL grades from the per-stance shapes routed through the pitcher-hand
    model with same_side set per stance; per-pitch s = stance-n-weighted blend.
  - Stuff+ vs anchors_v2 (per cell x pitch type). by entry min 10 total pitches;
    stance grade min 10 stance pitches; overall min 30.

Output: milb_stuff_v2.json  {pitcher: {ovr,n,by:{pt:{s,n,R,L}},ovrR,ovrL}}
"""
import json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/claude")
from milb_shape_adjustments import ball_adjust, pitcher_altitude_factor

ART = "/home/claude/stuff/artifacts"
MODELED = ["FF","SI","SL","CH","ST","FC","CU","FS","KC","SV"]
FB_TYPES = ["FF","SI","FC"]
FEATURES = ["start_speed","ivb","hb","rel_side","release_z","extension","spin_rate",
            "velo_diff","ivb_diff","hb_diff","arm_angle","pitch_type_code","same_side"]
PT_MAP = {"Fastball":"FF","Sinker":"SI","Slider":"SL","Sweeper":"ST","Cutter":"FC",
          "Curveball":"CU","Changeup":"CH","Splitter":"FS","Knuckle Curve":"KC","Slurve":"SV"}
DAMPEN = 0.40

def decode_height(code):
    if code is None: return None
    try: c = int(round(float(code)))
    except Exception: return None
    return c//100 + (c%100)/12.0

def impute_arm_angle(rel_z, height_ft, rel_side, ext):
    a = 10.8506 + 20.7891*rel_z + (-22.3478)*height_ft + (-3.8044)*abs(rel_side) + 8.5466*ext
    return float(np.clip(a, -90, 90))

def num(v): return v if isinstance(v,(int,float)) and v==v else None

def main():
    # DATA from the shipped dashboard
    h = open("/mnt/user-data/outputs/index.html", errors="ignore").read()
    s = h.find("{", h.find("DATA=")); d = 0
    for j in range(s, len(h)):
        if h[j]=="{": d+=1
        elif h[j]=="}":
            d-=1
            if d==0: break
    DATA = json.loads(h[s:j+1])
    P = {k:i for i,k in enumerate(DATA["p26h"])}
    B = {k:i for i,k in enumerate(DATA["bh"])}
    heights = DATA.get("heights", {})

    # optional levels file
    levels = {}
    if os.path.exists("/home/claude/traditional_stats.json"):
        ts = json.load(open("/home/claude/traditional_stats.json"))
        for name, lv in ts.items():
            levels[name] = [(e[0], e[1]) for e in (lv or [])]
        print(f"[06b] levels loaded for {len(levels)} pitchers")
    else:
        print("[06b] traditional_stats.json NOT available -> all below-AAA, altitude off")

    NO_BALL = {"Triple-A","AAA"}
    def pitcher_env(name, org):
        ip = levels.get(name)
        if not ip: return True, 1.0            # below-AAA fallback, no altitude
        main_lvl = max(ip, key=lambda t: t[1])[0]
        below = main_lvl not in NO_BALL
        alt = pitcher_altitude_factor(org, ip)
        return below, alt

    models = {hd: lgb.Booster(model_file=f"{ART}/stuff_hand_{hd}.txt") for hd in ["R","L"]}
    anchors = json.load(open(f"{ART}/anchors_v2.json"))
    code_map = json.load(open(f"{ART}/pitch_type_map.json"))

    # group rows
    all_by, st_by = {}, {}
    for r in DATA["p26d"]: all_by.setdefault(r[P["Pitcher"]], []).append(r)
    for r in DATA["bd"]:   st_by.setdefault(r[B["Pitcher"]], []).append(r)

    def grade(feat_row, hand, bh, code):
        cell = f"{hand}_{bh}"
        a = anchors.get(cell, {}).get(code)
        if not a or a["sigma"] == 0: return None
        pred = float(models[hand].predict(feat_row)[0])
        return 100 + 10*((-pred) - a["mu"])/a["sigma"]

    out = {}
    for name, prs in all_by.items():
        # hand
        ws=wt=0.0
        for r in prs:
            v=num(r[P["RelSd"]]); p=num(r[P["Pitches"]]) or 0
            if v is not None: ws+=v*p; wt+=p
        hand = "R" if (wt>0 and ws/wt<0) else ("L" if wt>0 else "R")
        org = prs[0][P["Org"]]
        below, alt = pitcher_env(name, org)
        hft = decode_height(heights.get(name))

        def adj(pt, vm, hm, se):
            if vm is None or hm is None: return None, None
            vm, hm = vm*alt, hm*alt
            return ball_adjust(pt, vm, hm, se if se is not None else float("nan"),
                               below, dampen=DAMPEN)

        # primary fastball off ALL rows
        usage = {}
        for r in prs:
            usage[r[P["PITCH_TYPE"]]] = usage.get(r[P["PITCH_TYPE"]],0) + (num(r[P["Pitches"]]) or 0)
        fb_c = {pt:n for pt,n in usage.items() if PT_MAP.get(pt) in FB_TYPES}
        prim = max(fb_c or usage, key=(fb_c or usage).get) if usage else None
        pv=pivb=phb=None
        for r in prs:
            if r[P["PITCH_TYPE"]] != prim: continue
            pv = num(r[P["Velo"]])
            av, ah = adj(prim, num(r[P["VM"]]), num(r[P["HM"]]), num(r[P["SpinEff"]]))
            pivb, phb = av, ah
        # mirror primary hb for LHP frame
        phb_f = -phb if (phb is not None and hand=="L") else phb

        stance_rows = st_by.get(name, [])
        by = {}
        for r in prs:
            pt = r[P["PITCH_TYPE"]]; code = PT_MAP.get(pt)
            n_all = num(r[P["Pitches"]]) or 0
            if code not in MODELED or n_all < 10: continue

            def feats_from(row, idx, bh):
                velo=num(row[idx["Velo"]]); vm=num(row[idx["VM"]]); hm=num(row[idx["HM"]])
                se=num(row[idx["SpinEff"]]); rs=num(row[idx["RelSd"]]); rz=num(row[idx["RelHt"]])
                ex=num(row[idx["Ext"]]); sp=num(row[idx["Spin"]])
                if None in (velo,vm,hm,rs,rz,ex,sp): return None
                avm, ahm = adj(pt, vm, hm, se)
                if avm is None: return None
                hb_f = -ahm if hand=="L" else ahm
                rs_f = -rs if hand=="L" else rs
                arm = impute_arm_angle(rz, hft if hft else rz+1.6, rs_f, ex)
                vd = velo-pv if pv is not None else 0.0
                ivd = avm-pivb if pivb is not None else 0.0
                hbd = hb_f-phb_f if phb_f is not None else 0.0
                same = 1 if bh==hand else 0
                return pd.DataFrame([[velo,avm,hb_f,rs_f,rz,ex,sp,vd,ivd,hbd,arm,
                                      code_map[code],same]], columns=FEATURES)

            gR=gL=None; nR=nL=0
            for sr in stance_rows:
                if sr[B["PITCH_TYPE"]] != pt: continue
                bh = sr[B["BATTER_STANCE"]]
                n_st = num(sr[B["Pitches"]]) or 0
                if n_st < 10: continue
                fr = feats_from(sr, B, bh)
                if fr is None: continue
                g = grade(fr, hand, bh, code)
                if bh=="R": gR, nR = g, n_st
                else:       gL, nL = g, n_st
            # per-pitch s: stance-n-weighted blend; fallback to ALL-row scored both stances
            if gR is not None or gL is not None:
                num_=den=0.0
                for g,nn in ((gR,nR),(gL,nL)):
                    if g is not None: num_+=g*nn; den+=nn
                s_val = num_/den if den else None
            else:
                fr = feats_from(r, P, "R"); s_val=None
                if fr is not None:
                    gr = grade(fr, hand, "R", code)
                    fr2 = feats_from(r, P, "L")
                    gl = grade(fr2, hand, "L", code) if fr2 is not None else None
                    parts=[g for g in (gr,gl) if g is not None]
                    s_val = float(np.mean(parts)) if parts else None
                    gR, gL = gr, gl
            if s_val is None: continue
            by[pt] = {"s": round(s_val), "n": int(n_all),
                      "R": round(gR) if gR is not None else None,
                      "L": round(gL) if gL is not None else None}
        if not by: continue
        tot = sum(v["n"] for v in by.values())
        def wavg(key):
            num_=den=0.0
            for v in by.values():
                if v[key] is not None: num_+=v[key]*v["n"]; den+=v["n"]
            return round(num_/den) if den else None
        out[name] = {"ovr": wavg("s") if tot>=30 else None, "n": int(tot), "by": by,
                     "ovrR": wavg("R"), "ovrL": wavg("L")}

    json.dump(out, open("/home/claude/milb_stuff_v2.json","w"))
    graded=[v["ovr"] for v in out.values() if v["ovr"] is not None]
    print(f"[06b] scored {len(out)} MiLB arms, {len(graded)} with ovr; mean {np.mean(graded):.1f}, sd {np.std(graded):.1f}")

if __name__ == "__main__":
    main()
