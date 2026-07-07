# stuff-plus-repo update — arm angles, extension correction, scoring robustness

Drop these files into your repo checkout (paths are repo-relative, they overwrite
in place), then:

    git add -A
    git commit -m "Extension correction v2, arm-angle overrides, clip + MC smoothing"
    git push

## Changed
- `README.md`, `VALIDATION.md` — document everything below
- `scripts/06c_score_college_v2.py` — arm-angle scout overrides (Mendes 40, Renfrow 50),
  p2-p98 differential clipping, Monte-Carlo grade smoothing (K=64, seed 42).
  Now fully repo-portable: reads models/, data/, writes grades/ when run from the repo.
- `grades/college_stuff_v2.json` — current college board (smoothed; Kuhns CB 113->107)

## New
- `data/ext_correction_v2.json` — measured extension correction spec (slope + per-type intercepts)
- `data/feature_bounds.json` — MLB p2-p98 supported ranges for FB differentials
- `scripts/07_refit_arm_imputer.py` — reproduces models/imputer_v2.json from the Savant CSVs + heights
- `scripts/08_feature_bounds.py` — reproduces data/feature_bounds.json from raw MLB parquets
- `scripts/09_college_ext_correction.py` — applies the extension correction to raw college shapes

Everything else in the repo is unchanged. Anchors + models still travel together.
