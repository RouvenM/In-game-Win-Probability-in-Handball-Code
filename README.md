# In-game win probabilities in handball

R code for the paper "A time-segmented generalized additive model for
forecasting in-game win probabilities in handball".

## Data

The play-by-play data of the German Handball-Bundesliga were provided under a
confidentiality agreement and are not included. The scripts expect them as
`raw_data/preprocessed_data.csv`, one row per possession, with the variables
described in Section 2 of the paper. The pre-match betting odds were collected
from www.betexplorer.com.

## Scripts

Run from the project root (the scripts locate files with the `here` package).
Tables, figures and the numbers quoted in the text are written to
`paper/generated/`.

| Script | Content |
|---|---|
| `01_model_comparison.R` | Leave-one-season-out comparison of the baselines, random forests, XGBoost and the ordered-categorical GAM (Section 4, Appendices A and B) |
| `02_tune_blend_parameters.R` | Grid over the time segmentations of the GAM (Section 4.1, Appendix D) |
| `04_use_cases.R` | Comebacks, value added and the goalkeeper decision (Section 5) |
| `05_leader_baseline.R` | Leading-team baseline, added to the comparison table of `01` |
| `06_rps_by_time.R` | RPS by game time (Figure 2), from the output of `01` |
| `07_rebuild_comparison_table.R` | Rebuilds the comparison table from a finished run of `01` without refitting |
| `08_blend_preview.R` | Short version of the segmentation grid, from the output of `02` |

Order: `01`, `05`, `06`, `02`, `08`, `04`. The cross-validation in `01` and
`02` refits every model on every season and takes several hours.

Helper functions are in `functions/`.

## R packages

dplyr, tidyr, purrr, stringr, magrittr, lubridate, data.table, arrow, here,
ggplot2, mgcv, MASS, nnet, VGAM, glmnet, caret, MLmetrics, ranger,
randomForest, xgboost, rpart, rpart.plot, party, locfit, skellam.
