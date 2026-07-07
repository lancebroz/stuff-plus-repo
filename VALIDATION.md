# Validation — which model predicts MLB performance?

**Test protocol** (mirrors `STUFF_PLUS.md` Test 2/4): grade every pitcher from
their **2025** pitch shapes using leakage-free out-of-fold predictions, then
correlate with their **actual 2026 run value** — a season the models never saw
(2026 is fully excluded from training). Filters: ≥500 pitches in 2025,
≥300 in 2026 → 408 arms. 2026 outcomes priced with the same contact-quality
delta-RE method (713K pitches through July 4, 2026).

## Results (same 408 arms; more negative = more predictive for stuff)

| Predictor | r vs 2026 actual RV |
|---|---|
| **v3 — 2-hand models, optical arm angle (production)** | **−0.174** |
| v1 — 4-cell models, proxy arm angle | −0.165 |
| Benchmark: 2025 actual results | +0.190 |
| Original model (documented, earlier 2026 snapshot) | −0.21 |

Quintile spread (2026 RV/100, worst→best stuff): v3 **+0.79 → −1.28**
(2.06 runs/100), v1 +0.84 → −1.22 (2.07). Both produce a clean monotone
gradient out of sample.

## Year-over-year stability (2024 → 2025 grades, n=359, ≥500 pitches both)

| Version | r |
|---|---|
| v1 (proxy angle) | 0.895 |
| v3 (optical angle) | 0.847 |
| Original (documented) | 0.82 |

v1's higher stability is partly artifactual: its arm angle is a fixed function
of release geometry, so the feature cannot move year to year; v3's measured
optical angle injects real variation. Given v3 predicts the future slightly
better *with* lower stability, its year-to-year movement carries more signal.

## Platoon-split quality (|vsR − vsL| per pitch, n ≥ 30)

| Board | mean split |
|---|---|
| v1 4-cell (college, identical shapes through 2 models) | 12.4 |
| Original board (MiLB) | 8.0 |
| **v3 (MiLB)** | **5.9 → 5.2** |
| **v3 (college)** | **5.1 → 3.8** |

Cross-model disagreement on *identical* shapes: 7.8 Stuff+ pts under the
4-cell architecture vs 4.0 under the 2-hand model — the remaining spread is
learned platoon interaction, not model noise.

## Verdict

**v3 (the shipped model) is the better instrument.** On pure out-of-sample MLB
prediction it is equal to or slightly ahead of v1 (−0.174 vs −0.165; the gap is
within noise at n=408), and every measurement-quality axis favors it: platoon
splits reflect real interactions instead of cross-model variance, MiLB and
college sit on one scale, and the arm-angle feature is grounded in optical
measurements rather than a geometry proxy. All versions sit in the same
predictive band as the original's documented −0.21; results-based metrics
remain the better pure one-year forecaster for established arms (+0.19 here),
which matches the original documentation — Stuff+'s edge is stability and the
small-sample regime where prospects live.


---

## Post-release updates (July 2026)

**College extension correction, measured on own data** (`data/ext_correction_v2.json`,
applied by `scripts/09_college_ext_correction.py`): 18 arms from the 2025 college
class matched to their 2026 MiLB tracking (72 pitcher x pitch-type pairs) replicate
Barrand's direction at ~2/3 magnitude, and the gap scales with extension
(-0.25 ft per ft). Deployed form: per-pitch-type intercept + shared slope
(residual RMSE 0.263 vs 0.309 for a flat offset), clamped to [0, 1.2] ft.

**Arm angle** (`scripts/07_refit_arm_imputer.py`): CV on 1,447 pitcher-seasons shows
height-segmented "mini models" test WORSE than the global linear fit (5.94 vs 5.88
RMSE) and LightGBM worse still (7.32) - the geometry->angle relationship is linear
and at its information ceiling (~5.9 deg). Residuals show no height bias; mild
compression at the extremes. Policy: the regression is the default; eye-verified
scout overrides (AA_OVERRIDES in the dashboard and college scorer) supersede it
for specific arms.

**Scoring robustness** (`scripts/08_feature_bounds.py`, college scorer): fastball
differentials are clipped to the MLB-supported p2-p98 range, and every grade is
Monte-Carlo smoothed over input measurement noise (K=64, fixed seed). Motivating
case: a curveball graded 113 on a 12-point tree-split cliff between -17.0 and
-17.5 mph of velocity separation; smoothed it grades 107. Mean effect across the
college board: 1.2 pts.