# Stuff+ Pipeline

Shape-only pitch-quality model. Grades every pitch from its physical shape
(velocity, movement, spin, release, extension, arm angle) on an MLB-anchored
scale: **100 = MLB average for that pitch type and platoon cell, 10 points = 1 SD.**
Scores both MiLB and college (draft) arms through the same fixed models.

Rebuilt July 2026 from `STUFF_PLUS.md` documentation against the public
`mlb-pitcher-data` repo (3.6M Statcast pitches, 2023–2026).

## Architecture (current = v3)

- **Two pitcher-hand LightGBM models** (`models/stuff_hand_{R,L}.txt`) with
  batter stance (`same_side`) as a feature — PitchSim-style platoon handling.
  One function per hand produces both the vs-RHH and vs-LHH grades, so platoon
  splits reflect learned interactions, not cross-model noise. (The original
  4-cell architecture is preserved in `legacy/` for comparison.)
- **13 features:** start_speed, ivb, hb, rel_side, release_z, extension,
  spin_rate, velo_diff, ivb_diff, hb_diff, arm_angle, pitch_type_code, same_side.
  Arm-side-positive frame (LHP geometry mirrored); differentials vs the
  pitcher's primary fastball.
- **Arm angle:** Savant optical arm angle for 98% of training rows
  (2024–2026 exports joined by MLBAM id × pitch type × season, pitcher-average
  fallback). Where unavailable — and for all MiLB/college scoring — imputed via
  `models/imputer_v2.json`, refit on 1,025 pitcher-seasons vs optical
  (R² = 0.80, RMSE 6.1°): `aa = 10.85 + 20.79·relHt − 22.35·heightFt −
  3.80·|relSide| + 8.55·ext`. Heights from Chadwick register + Baseball
  Databank (`data/mlbam_height.json`).
- **Target:** contact-quality delta run expectancy (EV/LA LightGBM values balls
  in play; count-RE telescoping; K/BB fixed weights). Negative = runs prevented.
- **Training:** 2.87M pitches (2023–2025), doc hyperparameters
  (lr .03, 31 leaves, min_child 300, ff .85, bag .8/1, λ2 1.0, 700 rounds,
  seed 42), 5-fold GroupKFold by pitcher → leakage-free OOF.
- **Anchors** (`models/anchors_v2.json`): μ/σ of −pred_rv per
  (pitcher-hand × batter-hand × pitch type) from the OOF.
  `Stuff+ = 100 + 10·(−pred_rv − μ)/σ`.

## Scoring

- **MiLB** (`scripts/06b_score_milb.py`): per-stance shapes + real SpinEff from
  the dashboard DATA blob; altitude neutralization (all levels) + MLB-ball
  adjustment below Triple-A (dampen 0.40); level gate from
  `traditional_stats.json` (lives in the Dash-Minor-Leagues repo, rebuilt daily).
- **College** (`scripts/06c_score_college_v2.py`): NCAA ball is out-of-domain for
  the MiLB-ball coefficients, so assumed per-pitch-type SpinEff (near-lossless;
  the spin term is tiny) and conservative dampen 0.25; altitude off. The
  measured extension correction (`scripts/09_college_ext_correction.py`,
  per-type intercept + shared slope fit on 72 college->MiLB matched pairs) is
  baked into the college data upstream; RelHt stays raw. Two out-of-domain
  guards protect every grade: FB differentials are clipped to the MLB p2-p98
  supported range (`data/feature_bounds.json`, derived by
  `scripts/08_feature_bounds.py`), and each pitch grade is Monte Carlo smoothed
  over input measurement noise (K=64, seeded) so LightGBM tree cliffs cannot
  swing a grade on half a mph. Scout arm-angle overrides (`AA_OVERRIDES` in the
  scorer and in the dashboard's computeArmAngle) supersede the imputer for
  eye-verified arms; current: Mendes 40, Renfrow 50.
- Aggregation: per-pitch grade (min 10), usage-weighted overall (min 30),
  vs-R / vs-L splits. Output: `{ovr, n, by:{pitch:{s,n,R,L}}, ovrR, ovrL}`.

## Rebuilding from scratch

1. Download raw parquets from `lancebroz/mlb-pitcher-data` → `mlbdata/raw/`.
2. `scripts/01_run_values.py` → pitches_with_rv.parquet
3. `scripts/02_features.py` → features.parquet, then join optical arm angles
   (Savant per-season exports) and refit the imputer.
4. `scripts/03b_train_hand.py` → stuff_hand_{R,L}.txt + anchors_v2.json
5. `scripts/06b_score_milb.py` / `06c_score_college_v2.py` → grade JSONs,
   embedded into the dashboard as `DATA.stuffPlus` / `DATA.collegeStuff`.

Training is deterministic (fixed seeds). Note: `anchors_v2.json` must come from
the same training run as the model files — anchors and models travel together.

## Files

| Path | What |
|---|---|
| `models/stuff_hand_{R,L}.txt` | The two production models |
| `models/anchors_v2.json` | Stuff+ scale anchors (μ/σ per cell × pitch type) |
| `models/imputer_v2.json` | Optical-calibrated arm-angle regression |
| `models/ev_la_rv.txt`, `models/re_by_count.json` | Run-value machinery |
| `models/pitch_type_map.json` | Pitch-type → categorical code |
| `grades/milb_stuff_v2.json` | Current MiLB board grades |
| `grades/college_stuff_v2.json` | Current college grades |
| `grades/stuffplus_original_backup.json` | Pre-rebuild board (revert path) |
| `legacy/` | v1 4-cell models + anchors (proxy arm angle) |
| `data/mlbam_height.json` | MLBAM id → height (inches), Chadwick-derived |
| `data/feature_bounds.json` | MLB p2-p98 differential ranges (scoring clip) |
| `data/ext_correction_v2.json` | Measured college extension correction spec |
| `scripts/07_refit_arm_imputer.py` | Rebuild the arm-angle imputer from optical CSVs |
| `scripts/08_feature_bounds.py` | Rebuild the scoring clip bounds from raw MLB data |
| `scripts/09_college_ext_correction.py` | Extension correction spec + apply function |
| `data/20{24,25,26} arm angle.csv` | Savant optical arm-angle exports (training inputs) |
| `VALIDATION.md` | Out-of-sample tests & version comparison |

## Caveats

- Grades are **not** command/sequencing aware — read alongside results.
- College grades ride an out-of-domain ball adjustment; treat as estimates.
- The rebuilt scale correlates ~0.55 (overall) with the pre-rebuild board;
  marquee/high-volume arms agree closely (mean move 3.9 pts), churn is in
  short-sample arms. See VALIDATION.md.
