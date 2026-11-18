# 01_model_comparison.R -------------------------------------------------------
# Paper workflow – Step 1: Model comparison
#
# The paper's thesis is that an interpretable GAM beats the flexible tree
# ensembles that dominate win-probability work. The candidates reflect that:
#
#   baseline        Betting-odds baseline (pre-match only, no in-game info)
#   GAM (ocat)      the focus model — smooth tensor products
#   Random Forest   flexible ML comparison (standard + ordinal cumulative-link)
#   XGBoost         flexible ML comparison (standard + ordinal cumulative-link)
#
# STATE CONVENTION. Every row is the state AFTER the possession it records: the
# clock is the terminating stamp, the score contains the goal if one was scored,
# the player counts already carry a suspension awarded during the attack. That
# is how the ticker writes the data down, and the state after possession i is
# the state from which possession i+1 is played, so nothing is lost by reading
# it that way---see pre_process_data for the full argument.
#
# The win-probability model deliberately EXCLUDES the goalkeeper decision. It
# conditions on the game situation, not on a team's tactical choice; letting the
# keeper pull into the model would absorb into the value function exactly the
# effect that 04 is set up to measure. The only numerical-strength term is
# susp_diff, the suspension-only man-up/man-down difference (raw court-player
# difference minus the keeper pull).
#
# NOTE on blending. This script compares the STANDARD single-fit models only,
# so the GAM here is a plain GAM fit once on the full training data. The
# time-varying K-segment blend is developed and tuned in a separate step
# (02_tune_blend_parameters.R) on top of the GAM selected here; keeping it out
# of this comparison makes the model-family comparison clean and independent of
# the blend design.
#
# Validation: full leave-one-season-out CV — every season is the held-out
# fold exactly once, none reserved. RF/XGB tune internally per fold, so a
# further held-out test season would mainly guard against model-type
# selection bias (small here); using every season instead gives a more
# precise comparison and out-of-fold diagnostics across the whole dataset.
#
# Selection criterion: lowest mean RPS (to 4 decimals); ECE breaks near-ties.
# -----------------------------------------------------------------------------

rm(list = ls())
library(here)
library(dplyr)
library(tidyr)
library(ggplot2)

source(here("functions", "general_functions.R"))
source(here("functions", "pre_processing.R"))
source(here("functions", "model_wrappers.R"))
source(here("functions", "evaluation_functions.R"))
source(here("functions", "momentum_features.R"))
source(here("functions", "output_helpers.R"))
source(here("functions", "comparison_table.R"))

# =============================================================================
# 1. Data preparation
# =============================================================================
daten <- read.csv(here("raw_data", "preprocessed_data.csv"))
daten <- pre_process_data(daten)

# Four-factors-style momentum features (recent shooting/turnover/penalty rates
# for both teams) — computed on the full possession sequence BEFORE the
# rare-player-count rows are filtered out below, so "last 5 attacks" windows
# aren't artificially punctured by dropped rows. See functions/momentum_features.R
momentum_window <- 5
daten <- add_momentum_features(daten, window = momentum_window)

# Which of the three scoreboard states a possession is in, from the point of
# view of the team in possession. Defined before the data step because the
# lead3 column below calls it, and used by the two scoreboard-reading baselines
# and by 05_leader_baseline.R, which must classify states the same way.
lead_cell <- function(x) {
  d <- suppressWarnings(as.numeric(as.character(x)))
  ifelse(is.na(d), "0", ifelse(d > 0, "1", ifelse(d < 0, "-1", "0")))
}

# Possession count BEFORE the rare-player-count filter below. The data section
# quotes both this and the estimation sample, and quoting either as a literal is
# how they drift apart from the pipeline.
n_poss_all <- nrow(daten)

daten <- daten |>
  ungroup() |>
  mutate(
    final_int = as.integer(factor(final, levels = c(-1, 0, 1))),
    final_ord = ordered(final, levels = c("-1", "0", "1")),
    matchid   = as.factor(matchid),
    final_rf  = factor(final, levels = c("-1", "0", "1"),
                       labels = c("loss", "draw", "win")),
    HZ        = ifelse(as.character(HZ) == "30", 1, 0),
    # Keeper-pulled indicator (1 = keeper substituted for a field player,
    # 0 = keeper in goal). NOT a covariate in the win-probability model:
    # pulling the keeper is a DECISION the possessing team takes, not part of
    # the situation it faces. It is used only to build the suspension-only
    # field-player difference below (and, in 04, to value the decision itself).
    # Read with the ten-second tolerance (gk_pos_10s / no_field_players_pos_10s,
    # see pre_process_data): a keeper restored in the closing seconds of an
    # attack played seven against six should not turn that attack into an even-
    # strength one. The opponent has no 10 s variant and needs none---it is the
    # attacking team's keeper that moves.
    gk_pulled = as.integer(as.character(gk_pos_10s) == "0"),
    # Raw court-player difference (own - opponent), which conflates suspensions
    # with the keeper pull.
    player_diff_pos_opp = as.integer(as.character(no_field_players_pos_10s)) -
                          as.integer(as.character(no_field_players_opp)),
    # SUSPENSION-ONLY numerical difference: subtracting gk_pulled removes the
    # extra attacker the possessing team gains by pulling its keeper, leaving
    # only the man-up/man-down imbalance caused by time penalties (0 = even,
    # -1 = one own player suspended, +1 = opponent a man down, ...). This is
    # part of the situation; the keeper decision is not.
    susp_diff = player_diff_pos_opp - gk_pulled,
    # The scoreboard reduced to its sign: leading, level, or trailing, from the
    # possessing team's view. This is the whole of the in-game information the
    # odds-and-scoreboard baseline below is allowed to see -- not the size of
    # the lead, not the clock.
    lead3 = factor(lead_cell(gls_diff_pos_opp), levels = c("-1", "0", "1"))
  ) |>
  # Remove rare player-count situations (too sparse to estimate reliably)
  filter(
    no_field_players_pos_10s %in% c("5", "6", "7"),
    no_field_players_opp     %in% c("5", "6", "7")
  )

#daten = daten |> filter(season %in% unique(daten$season)[1:2])

make_actual <- function(data) {
  ifelse(data$final == "1",  "win",
         ifelse(data$final == "-1", "loss", "draw"))
}

# =============================================================================
# 2. Season split — every season is a LOSO-CV fold, none held back
# =============================================================================
cv_seasons <- sort(unique(daten$season))

cat("LOSO-CV seasons: ", paste(cv_seasons, collapse = ", "), "\n\n")

# =============================================================================
# 3. (No blend here.) The time-varying K-segment blend is designed and tuned in
#    02_tune_blend_parameters.R on top of the GAM selected below. This script
#    compares the standard, single-fit model families only.
# =============================================================================

# =============================================================================
# 4. Variable sets and formulas per model type
# =============================================================================
responses  <- c("final", "final_int", "final_ord", "final_rf")

# Momentum swing = own minus opponent rolling stat (positive = the possessing
# team is currently in better recent form than the opponent). The 8 raw
# mom_*_pos/opp columns are also available in `daten` for anyone who wants to
# explore the one-sided effects separately; only the 4 diff terms are wired
# into the formulas below to keep the added covariate count moderate.
momentum_covariates <- c("mom_conversion_diff_pos_opp", "mom_shot_pct_diff_pos_opp",
                         "mom_turnover_diff_pos_opp",   "mom_penalty_diff_pos_opp")

covariates <- c("bet_win_prob", "gls_diff_pos_opp", "time", "home",
                "susp_diff", "HZ", momentum_covariates)

vars_baseline <- c("final", "final_int", "final_ord", "final_rf", "bet_win_prob")
# The in-game counterpart of the betting-odds baseline reads nothing but the
# scoreboard, so the score difference is the only covariate it needs.
vars_leader   <- c("final", "final_int", "final_ord", "final_rf", "gls_diff_pos_opp")
# The two references combined: what was known before throw-off, plus the one
# piece of in-game information anyone reads off a scoreboard.
vars_odds_leader <- c("final", "final_int", "final_ord", "final_rf",
                      "bet_win_prob", "lead3")
vars_models   <- c(responses, covariates, "matchid")   # GAM needs matchid
# RF ignores matchid (not referenced by its formula); XGBoost needs it to
# build match-grouped internal CV folds for hyperparameter tuning
vars_rfxgb    <- c(responses, covariates, "matchid")

# Baseline: betting odds only (no in-game information)
formula_baseline <- final_ord ~ bet_win_prob

# Baseline: the odds AND the sign of the score difference. Deliberately nothing
# else -- no clock, no margin, no momentum -- so that the step from this to the
# full model measures what the rest of the game state is worth once the two
# obvious things are already in hand.
formula_odds_leader <- final_ord ~ bet_win_prob + lead3

# GAM (ocat) — the win-probability model, evaluated at the state each row
# records, i.e. after that possession. The only numerical-strength term is
# susp_diff, the suspension-only man-up/man-down difference; the keeper pull
# does not enter (see above). Score difference and pre-match odds get bivariate
# tensor-product interactions with time; momentum swings enter linearly.
formula_gam <- final_int ~
  te(time, gls_diff_pos_opp, bs = c("tp", "tp"), k = c(10, 10)) +
  te(time, bet_win_prob,           bs = c("tp", "tp"), k = c(10, 10)) +
  s(matchid, bs = "re") +
  mom_conversion_diff_pos_opp +
  mom_shot_pct_diff_pos_opp +
  mom_turnover_diff_pos_opp +
  mom_penalty_diff_pos_opp +
  home + susp_diff + HZ

# Random Forest / Ordinal RF / Ordinal XGBoost — same variables,
# interactions auto-detected
formula_rf <- final_rf ~
  time + gls_diff_pos_opp + bet_win_prob +
  mom_conversion_diff_pos_opp + mom_shot_pct_diff_pos_opp +
  mom_turnover_diff_pos_opp   + mom_penalty_diff_pos_opp +
  home + susp_diff + HZ

# (Ordered-logit / polr was dropped from the comparison — the GAM is the
#  linear model's strictly more flexible superset and is the focus here.)

# Skellam — Poisson rates for both teams' remaining goals; prediction only
# uses the current game state (time, strength, score), the rem_goals_* columns
# are training responses computed from the final score
# formula_skellam_pos <- rem_goals_pos ~
#   te(time, bet_win_prob, k = c(10, 10)) + s(gls_diff_pos_opp, k = 10) + home
# formula_skellam_opp <- rem_goals_opp ~
#   te(time, bet_win_prob, k = c(10, 10)) + s(gls_diff_pos_opp, k = 10) + home
# 
# vars_skellam <- c(responses, "time", "bet_win_prob", "gls_diff_pos_opp",
#                   "home", "rem_goals_pos", "rem_goals_opp")

# =============================================================================
# 5. Helper: evaluate a prediction matrix against actuals
# =============================================================================
evaluate_predictions <- function(preds, actuals) {
  preds <- as.data.frame(preds)
  if (!all(c("win", "draw", "loss") %in% names(preds))) {
    stop("preds must have columns win, draw, loss")
  }
  complete_rows <- complete.cases(preds)
  if (mean(complete_rows) < 0.99) {
    warning(sprintf("%d rows (%.1f%%) with NA predictions excluded from metrics.",
                    sum(!complete_rows), 100 * mean(!complete_rows)))
  }
  p <- preds[complete_rows, ]
  a <- actuals[complete_rows]
  list(
    rps     = mean(compute_rps(p, a)),
    brier   = mean(compute_brier_score(p, a)),
    logloss = log_loss_manual(p, a),
    ece     = compute_calibration_error(p, a)$e_overall,
    n       = sum(complete_rows),
    preds   = preds,   # full (with NAs) for calibration plots
    actuals = actuals
  )
}

# =============================================================================
# 6. Helper: fit + predict (single fit on the full training data)
# =============================================================================
# Every model in this comparison is fit once on the full training seasons, with
# no time split. The time-varying K-segment blend is a separate design step on
# top of the GAM (02_tune_blend_parameters.R), deliberately kept out of this
# model-family comparison.
fit_predict <- function(model_id, formula, fit_fn, predict_fn,
                        train_data, val_data) {
  predict_fn(fit_fn(train_data), val_data)
}

# =============================================================================
# 7. Model definitions (name, fit function, predict function, data vars)
# =============================================================================
model_defs <- list(

  baseline = list(
    name       = "Betting Odds Baseline",
    vars       = vars_baseline,
    fit_fn     = function(data) fit_model("ordinal", data, formula_baseline),
    predict_fn = function(fit, newdata) {
      p <- predict(fit, newdata = newdata, type = "probs")
      # polr returns columns named "-1","0","1" → rename
      colnames(p) <- c("loss", "draw", "win")
      p[, c("win", "draw", "loss")]
    }
  ),

  # THE IN-GAME FLOOR. The betting-odds baseline knows the two teams and nothing
  # about the match in progress; this one is its mirror image, and it is not a
  # model at all. It reads the scoreboard literally as a forecast: whoever is
  # ahead is named the winner, a level score is called a draw. Nothing is
  # estimated and the training seasons are never consulted, which is the point
  # -- it is the floor any in-game forecast has to clear, not a competitor.
  #
  # It states certainty, so its logarithmic score is infinite the first time it
  # is wrong, and it is wrong often. log_loss_manual clips at 1e-15 before
  # taking the logarithm, which would turn that infinity into roughly 34.5 per
  # mistaken possession -- a number that reports the clipping constant rather
  # than the forecast -- so the table shows the score that is actually defined
  # here, which is infinite.
  leader = list(
    name       = "Leading-team baseline",
    vars       = vars_leader,
    fit_fn     = function(data) NULL,
    predict_fn = function(fit, newdata) {
      cell <- lead_cell(newdata$gls_diff_pos_opp)
      data.frame(win  = as.numeric(cell == "1"),
                 draw = as.numeric(cell == "0"),
                 loss = as.numeric(cell == "-1"))
    }
  ),

  # THE TWO REFERENCES, COMBINED. Each of the baselines above is deaf in one
  # ear: the odds know the teams and nothing about the match, the rule knows
  # the match and nothing about the teams. A forecaster with both is the one a
  # reader will actually have in mind, and it is the harder bar. Same
  # proportional-odds family as the odds baseline, with the scoreboard entering
  # as a three-level factor, so the step up from it isolates what the clock,
  # the size of the lead and the rest of the state add.
  odds_leader = list(
    name       = "Odds-and-scoreboard baseline",
    vars       = vars_odds_leader,
    fit_fn     = function(data) fit_model("ordinal", data, formula_odds_leader),
    predict_fn = function(fit, newdata) {
      p <- predict(fit, newdata = newdata, type = "probs")
      colnames(p) <- c("loss", "draw", "win")
      p[, c("win", "draw", "loss")]
    }
  ),

  # baseline_polr = list(
  #   name       = "Baseline Ordered Logit",
  #   vars       = vars_rfxgb,
  #   fit_fn     = function(data) fit_model("ordinal", data, formula_polr),
  #   predict_fn = function(fit, newdata) predict_model(fit, newdata, "ordinal")
  # ),
  # 
  # polr = list(
  #   name       = "Ordered Logit",
  #   vars       = vars_rfxgb,
  #   fit_fn     = function(data) fit_model("ordinal", data, formula_polr),
  #   predict_fn = function(fit, newdata) predict_model(fit, newdata, "ordinal")
  # ),

  # Standard GAM (ocat), single fit on the full training data — no blend here.
  # The blend is developed in 02_tune_blend_parameters.R on top of this model.
  gam = list(
    name       = "GAM (ocat)",
    vars       = vars_models,
    fit_fn     = function(data) fit_model("gam", data, formula_gam),
    predict_fn = function(fit, newdata) predict_model(fit, newdata, "gam")
  ),

  baseline_rf = list(
    name       = "Baseline Random Forest",
    vars       = vars_rfxgb,
    fit_fn     = function(data) fit_model("rf", data, formula_rf),
    predict_fn = function(fit, newdata) predict_model(fit, newdata, "rf")
  ),
  
  # rf = list(
  #   name       = "Random Forest",
  #   vars       = vars_rfxgb,
  #   fit_fn     = function(data) fit_model("rf", data, formula_rf),
  #   predict_fn = function(fit, newdata) predict_model(fit, newdata, "rf")
  # ),

  baseline_rf_ord = list(
    name       = "Baseline Ordinal RF",
    vars       = vars_rfxgb,
    fit_fn     = function(data) fit_ordinal_rf(data, formula_rf),
    predict_fn = function(fit, newdata) predict_ordinal_rf(fit, newdata)
  ),

  # rf_ord = list(
  #   name       = "Ordinal RF",
  #   vars       = vars_rfxgb,
  #   fit_fn     = function(data) fit_ordinal_rf(data, formula_rf),
  #   predict_fn = function(fit, newdata) predict_ordinal_rf(fit, newdata)
  # ),

  baseline_xgb_multi = list(
    name       = "Baseline XGBoost",
    vars       = vars_rfxgb,
    fit_fn     = function(data) fit_model("xgb", data, formula_rf),
    predict_fn = function(fit, newdata) predict_model(fit, newdata, "xgb")
  ),

  baseline_xgb = list(
    name       = "Baseline Ordinal XGBoost",
    vars       = vars_rfxgb,
    fit_fn     = function(data) fit_ordinal_xgb(data, formula_rf),
    predict_fn = function(fit, newdata) predict_ordinal_xgb(fit, newdata)
  )#,

  # xgb = list(
  #   name       = "XGBoost",
  #   vars       = vars_rfxgb,
  #   fit_fn     = function(data) fit_ordinal_xgb(data, formula_rf),
  #   predict_fn = function(fit, newdata) predict_ordinal_xgb(fit, newdata)
  # ),

  # Skellam kept only as a single-fit reference (see header).
  # baseline_skellam = list(
  #   name       = "Skellam",
  #   vars       = vars_skellam,
  #   fit_fn     = function(data) fit_skellam(data, formula_skellam_pos,
  #                                            formula_skellam_opp),
  #   predict_fn = function(fit, newdata) predict_skellam(fit, newdata)
  # )
)

# =============================================================================
# 8. Full LOSO-CV — every season held out once, for every model
# =============================================================================
cat("========== LOSO-CV Model Comparison ==========\n\n")

cv_results <- list()
cv_preds   <- list()   # per model: rbind of out-of-fold predictions, all seasons
cv_actuals <- list()   # per model: matching actuals, same row order as cv_preds

for (m_id in names(model_defs)) {
  m <- model_defs[[m_id]]
  cat(sprintf("--- Model: %s ---\n", m$name))

  fold_metrics <- list()
  fold_preds   <- list()
  fold_actuals <- list()

  for (f in seq_along(cv_seasons)) {
    val_season <- cv_seasons[f]
    cat(sprintf("  Fold %d / %d  (val: %s)  ...\n", f, length(cv_seasons), val_season))

    train_data <- daten |>
      filter(season %in% cv_seasons, season != val_season) |>
      dplyr::select(all_of(m$vars))
    val_data <- daten |>
      filter(season == val_season) |>
      dplyr::select(all_of(m$vars))

    preds  <- fit_predict(m_id, NULL, m$fit_fn, m$predict_fn,
                          train_data, val_data)
    preds  <- as.data.frame(preds)
    if (!all(c("win","draw","loss") %in% names(preds)))
      colnames(preds) <- c("win","draw","loss")

    actuals <- make_actual(val_data)
    metrics <- evaluate_predictions(preds, actuals)

    fold_metrics[[f]] <- tibble(
      model       = m_id,
      model_name  = m$name,
      fold        = val_season,
      rps         = metrics$rps,
      brier       = metrics$brier,
      logloss     = metrics$logloss,
      ece         = metrics$ece,
      n           = metrics$n
    )
    fold_preds[[f]]   <- preds
    fold_actuals[[f]] <- actuals
    cat(sprintf("    RPS = %.5f  |  ECE = %.5f\n", metrics$rps, metrics$ece))
  }

  cv_results[[m_id]] <- bind_rows(fold_metrics)
  cv_preds[[m_id]]    <- bind_rows(fold_preds)
  cv_actuals[[m_id]]  <- unlist(fold_actuals, use.names = FALSE)
  cat("\n")
}

cv_df <- bind_rows(cv_results)

# Time and match id, in the same across-fold row order as cv_preds/cv_actuals
# (row order doesn't depend on which columns a model selects, so these align
# with every model's pooled predictions)
cv_time    <- unlist(lapply(cv_seasons, function(s) daten$time[daten$season == s]))
# The match side each out-of-fold row belongs to: the match, and which of the
# two teams was in possession. The time curve averages over these rather than
# over possessions, so that it does not move with the changing mix of matches
# per window (see rps_by_time_curve).
cv_side    <- unlist(lapply(cv_seasons, function(s)
  paste0(daten$matchid[daten$season == s], "/", daten$home[daten$season == s])))
cv_matchid <- unlist(lapply(cv_seasons,
                            function(s) as.character(daten$matchid[daten$season == s])))

# =============================================================================
# 9. LOSO-CV summary table
# =============================================================================
# Reported point metrics are computed on the POOLED out-of-fold predictions
# (all held-out seasons together), NOT averaged over folds. This is the same
# basis as 02_tune_blend_parameters.R, so ECE in particular is comparable
# across the two scripts; a fold-averaged ECE is noisier and upward-biased by
# per-bin sampling noise. Column names keep the "_mean" suffix for downstream
# compatibility, but they now hold pooled point estimates.
pooled_metrics <- bind_rows(lapply(names(model_defs), function(m_id) {
  p  <- as.data.frame(cv_preds[[m_id]])
  a  <- cv_actuals[[m_id]]
  ok <- complete.cases(p)
  cbind(
    data.frame(model = m_id, stringsAsFactors = FALSE),
    # The nine reported figures, defined once in functions/comparison_table.R
    # so that 05 and 07 cannot drift from this.
    comparison_metrics(p, a, cv_time),
    data.frame(
    # An exact zero on an outcome that happened makes the logarithmic score
    # infinite; log_loss_manual clips first, which hides that behind a number
    # set by the clipping constant. Computed for the console only -- the table
    # does not report it.
    logloss_mean = {
      pm <- as.matrix(p[ok, ])
      realised <- pm[cbind(seq_len(nrow(pm)), match(a[ok], colnames(pm)))]
      if (any(realised <= 0)) Inf else log_loss_manual(p[ok, ], a[ok])
    },
    stringsAsFactors = FALSE
    )
  )
}))

# Fold-to-fold standard deviations (season-to-season variability) are retained
# for the dependence discussion, but are not the reported point metrics.
fold_sd <- cv_df |>
  group_by(model, model_name) |>
  summarize(across(c(rps, brier, logloss, ece), sd, .names = "{.col}_sd"),
            .groups = "drop")

cv_summary <- fold_sd |>
  left_join(pooled_metrics, by = "model") |>
  # Rank by pooled RPS, but treat RPS differences beyond the 4th decimal as
  # noise (they are far within the fold-to-fold spread) and let ECE break
  # such near-ties. Only the sort key is rounded; the printed values stay full.
  arrange(round(rps_mean, 4), ece_mean)

cat("========== LOSO-CV Summary (sorted by mean RPS) ==========\n")
print(cv_summary[,-1], n = Inf)

# ---- Auto-save: model-comparison table for the manuscript --------------------
# Writes paper/generated/model_comparison.tex, \input by the manuscript. The
# Skellam reference is a discussion-only bridge model (see header) and is left
# out of the main comparison table. Display names match the manuscript prose.
# The best (lowest) RPS and ECE cells are bolded via raw LaTeX passthrough.
mc_display <- COMPARISON_DISPLAY
mc <- cv_summary |>
  filter(model %in% names(mc_display)) |>
  transmute(model,
            Model   = unname(mc_display[model]),
            RPS     = rps_mean,
            `RPS 1st`   = rps_h1,
            `RPS 2nd`   = rps_h2,
            Brier   = brier_mean,
            `Brier 1st` = brier_h1,
            `Brier 2nd` = brier_h2,
            `Log-loss` = logloss_mean,
            ECE     = ece_mean,
            `ECE 1st`   = ece_h1,
            `ECE 2nd`   = ece_h2) |>
  arrange(round(RPS, 4), ECE)

fmt3 <- function(x) ifelse(is.finite(x), formatC(x, format = "f", digits = 3), "$\\infty$")
write_comparison_table(mc)

# Headline numbers as reusable macros (\gamRPS, \gamECE, \baselineRPS, ...)
save_scalar("gamRPS",      fmt3(mc$RPS[mc$model == "gam"]))
save_scalar("gamECE",      fmt3(mc$ECE[mc$model == "gam"]))
save_scalar("gamBrier",    fmt3(mc$Brier[mc$model == "gam"]))
save_scalar("baselineRPS", fmt3(mc$RPS[mc$model == "baseline"]))

# The paper compares the GAM against the STRONGEST tree ensemble, so these
# macros must track whichever ensemble that is rather than a fixed model id.
# Which variant wins is not fixed a priori---the ordinal cumulative-link forms
# are not guaranteed to beat the standard multi-class ones---and pointing the
# macros at a weaker ensemble would overstate the GAM's advantage. `mc` is
# already ordered best-first (RPS to 4 decimals, ECE breaking ties).
ens_rows <- mc[grepl("rf|xgb", mc$model), ]
best_ens <- ens_rows[1, ]
cat(sprintf("\nStrongest tree ensemble: %s (RPS %.3f, ECE %.3f)\n",
            best_ens$Model, best_ens$RPS, best_ens$ECE))
save_scalar("bestEnsName",    best_ens$Model)
save_scalar("bestEnsRPS",     fmt3(best_ens$RPS))
save_scalar("bestEnsECE",     fmt3(best_ens$ECE))
save_scalar("bestEnsBrier",   fmt3(best_ens$Brier))

# =============================================================================
# 10. Select best model
# =============================================================================
best_model_id <- cv_summary$model[1]
cat(sprintf("\nBest model: %s  (mean RPS = %.5f)\n\n",
            cv_summary$model_name[1], cv_summary$rps_mean[1]))

# =============================================================================
# 10b. Paired match-clustered bootstrap for the model differences
# =============================================================================
# The fold-level sds in cv_summary measure season-to-season variation of the
# absolute RPS, NOT the uncertainty of the DIFFERENCE between models. Since all
# models are scored on the same out-of-fold observations, the informative
# comparison is paired: the per-possession RPS difference, aggregated by a
# bootstrap that resamples whole MATCHES (not rows), so within-match dependence
# is respected (cf. brill2024exploring; Section "dependence structure" in the
# paper). A mostly-negative delta means the first model is better; p_first_better
# is the share of bootstrap replicates in which it wins.
paired_rps_bootstrap <- function(id_a, id_b, n_boot = 2000, seed = 1) {
  pa <- as.data.frame(cv_preds[[id_a]]); pb <- as.data.frame(cv_preds[[id_b]])
  a  <- cv_actuals[[id_a]]                      # actuals identical across models
  ok <- complete.cases(pa) & complete.cases(pb) # paired: rows both models scored
  d   <- compute_rps(pa[ok, ], a[ok]) - compute_rps(pb[ok, ], a[ok])  # <0 => A better
  mid <- cv_matchid[ok]
  rows_by_match <- split(seq_along(d), mid)     # possession-row indices per match
  matches <- names(rows_by_match)
  set.seed(seed)
  boot <- replicate(n_boot, {
    idx <- unlist(rows_by_match[sample(matches, replace = TRUE)], use.names = FALSE)
    mean(d[idx])
  })
  list(mean_delta     = mean(d),
       ci             = quantile(boot, c(0.025, 0.975)),
       p_first_better = mean(boot < 0))
}

# Compare the best model against the best tree ensemble (the paper's claim).
# best_tree is needed beyond the bootstrap (per-season table and the ensemble's
# calibration figure below), so it is resolved outside the guarded block.
tree_ids  <- intersect(cv_summary$model, grep("rf|xgb", names(model_defs), value = TRUE))
best_tree <- if (length(tree_ids) > 0) {
  cv_summary |> filter(model %in% tree_ids) |> dplyr::slice(1) |> pull(model)
} else NA_character_

if (!is.na(best_tree)) {
  bt <- paired_rps_bootstrap(best_model_id, best_tree)
  cat(sprintf(paste0(
    "Paired RPS difference (%s minus %s), match-clustered bootstrap:\n",
    "  mean delta RPS = %+.5f   95%% CI [%+.5f, %+.5f]\n",
    "  %s better in %.1f%% of bootstrap replicates\n\n"),
    cv_summary$model_name[1],
    cv_summary$model_name[cv_summary$model == best_tree],
    bt$mean_delta, bt$ci[1], bt$ci[2],
    cv_summary$model_name[1], 100 * bt$p_first_better))

  # Auto-save the paired-bootstrap headline numbers as macros. The paper reports
  # the GAM's advantage as a positive number, so flip the sign of the delta
  # (mean_delta < 0 means the first/GAM model is better) and the CI bounds.
  fmt4 <- function(x) formatC(abs(x), format = "f", digits = 4)
  save_scalar("bootDeltaRPS", fmt4(bt$mean_delta))
  save_scalar("bootCIlo",     fmt4(bt$ci[2]))   # flipped: upper bound of |delta|
  save_scalar("bootCIhi",     fmt4(bt$ci[1]))   # flipped: lower bound of |delta|
  save_scalar("bootWinPct",   formatC(100 * bt$p_first_better, format = "f", digits = 1))
  save_scalar("nBoot",        formatC(2000L, format = "d", big.mark = "{,}"))
}

# =============================================================================
# 10c. Per-season comparison of the two leading models (appendix table)
# =============================================================================
# The pooled table in Section 4 reports one number per model, which cannot show
# whether the additive model's edge is systematic or the average of a few
# favorable seasons. This puts the two leading models side by side in every
# held-out season on all three proper scoring rules. The number of comparisons
# the GAM wins is written out as a macro rather than asserted in the prose, so
# the manuscript can never claim a clean sweep the data does not support.
if (!is.na(best_tree)) {
  g <- cv_df |> filter(model == best_model_id) |> arrange(fold)
  x <- cv_df |> filter(model == best_tree)     |> arrange(fold)
  stopifnot(identical(g$fold, x$fold), nrow(g) == length(cv_seasons))

  # Format a pair of values, bolding the smaller one (lower is better on both
  # rules). Emitted as raw LaTeX, hence raw_cols in save_latex_table.
  pair <- function(a, b, digits = 4) {
    fa <- formatC(a, format = "f", digits = digits)
    fb <- formatC(b, format = "f", digits = digits)
    list(a = ifelse(a < b, paste0("\\textbf{", fa, "}"), fa),
         b = ifelse(b < a, paste0("\\textbf{", fb, "}"), fb))
  }
  p_rps <- pair(g$rps, x$rps); p_bri <- pair(g$brier, x$brier)

  # Two reported rules, so two comparisons per season. The logarithmic score is
  # still printed to the console but no longer reported (see the main table).
  wins_rps <- sum(g$rps < x$rps)
  wins_all <- wins_rps + sum(g$brier < x$brier)
  n_comp   <- 2 * nrow(g)
  cat(sprintf(paste0(
    "\nPer-season comparison (%s vs %s):\n",
    "  GAM lower on RPS in %d of %d seasons; %d of %d comparisons overall\n"),
    cv_summary$model_name[1],
    cv_summary$model_name[cv_summary$model == best_tree],
    wins_rps, nrow(g), wins_all, n_comp))
  if (wins_all < n_comp)
    warning("Per-season comparison is not a clean sweep---check the manuscript ",
            "wording, which reads the counts from \\foldWinsAll / \\foldComparisons.")

  # The uncertainty of the comparison, read across the five held-out seasons:
  # the season-wise RPS advantage of the GAM over the ensemble, with its mean,
  # standard deviation and range. Five values are few, so this is a spread, not
  # a confidence interval.
  d_rps <- x$rps - g$rps
  fmt4d <- function(v) formatC(v, format = "f", digits = 4)
  cat(sprintf("  RPS advantage per season: mean %.4f, SD %.4f, range %.4f to %.4f\n",
              mean(d_rps), sd(d_rps), min(d_rps), max(d_rps)))

  fold_tex <- data.frame(
    Season           = as.character(g$fold),
    `RPS (GAM)`      = p_rps$a,  `RPS (XGB)`      = p_rps$b,
    `RPS difference` = fmt4d(d_rps),
    `Brier (GAM)`    = p_bri$a,  `Brier (XGB)`    = p_bri$b,
    check.names = FALSE, stringsAsFactors = FALSE
  )
  save_latex_table(
    fold_tex, "fold_comparison",
    caption = paste0("Leave-one-season-out cross-validation, resolved by ",
                     "held-out season, for the two leading models: the ",
                     "ordered-categorical GAM and the strongest tree ensemble. ",
                     "Lower is better on both proper scoring rules; the ",
                     "better value of each pair is set in bold. RPS difference ",
                     "is the ensemble's RPS minus the GAM's. The pooled ",
                     "figures of Table~\\ref{tab:model_comparison} average over ",
                     "these seasons."),
    label = "tab:fold_comparison",
    align = "lccccc",
    raw_cols = names(fold_tex)[-1])

  save_scalar("foldDiffMean",    fmt4d(mean(d_rps)))
  save_scalar("foldDiffSD",      fmt4d(sd(d_rps)))
  save_scalar("foldDiffMin",     fmt4d(min(d_rps)))
  save_scalar("foldDiffMax",     fmt4d(max(d_rps)))
  save_scalar("foldSeasons",     nrow(g))
  save_scalar("foldWinsRPS",     wins_rps)
  save_scalar("foldWinsAll",     wins_all)
  save_scalar("foldComparisons", n_comp)
}

# Dataset-size macros for the prose (matches, seasons, possessions). LaTeX thin
# comma ({,}) as the thousands separator so they match the manuscript's style.
save_scalar("nMatches", formatC(length(unique(daten$matchid)), format = "d", big.mark = "{,}"))
save_scalar("nSeasons", length(cv_seasons))
save_scalar("nPoss",    formatC(nrow(daten), format = "d", big.mark = "{,}"))
save_scalar("nPossAll", formatC(n_poss_all,  format = "d", big.mark = "{,}"))

# =============================================================================
# 11. Visualisations — all built from pooled, out-of-fold LOSO-CV predictions
# (every prediction below comes from a model that never saw that
# observation's season during training)
# =============================================================================

# 11a. LOSO-CV RPS comparison (box + point per fold)
p_rps_by_model <- ggplot(cv_df, aes(x = reorder(model_name, rps), y = rps)) +
  geom_boxplot(width = 0.4, outlier.shape = NA) +
  geom_jitter(aes(color = fold), width = 0.1, size = 2.5) +
  labs(title = "LOSO-CV RPS by Model",
       subtitle = "Lower is better; points = individual folds, colored by season",
       x = NULL, y = "RPS", color = "Val season") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
print(p_rps_by_model)
save_plot(p_rps_by_model, "cv_rps_by_model")

# 11b. Pooled LOSO-CV metric comparison (all seasons, out-of-fold)
cv_pooled <- bind_rows(lapply(names(model_defs), function(m_id) {
  p  <- cv_preds[[m_id]]
  a  <- cv_actuals[[m_id]]
  ok <- complete.cases(p)
  tibble(model = m_id, model_name = model_defs[[m_id]]$name,
         rps   = mean(compute_rps(p[ok, ], a[ok])),
         brier = mean(compute_brier_score(p[ok, ], a[ok])))
}))

cv_pooled_long <- cv_pooled |>
  pivot_longer(c(rps, brier), names_to = "metric", values_to = "value") |>
  mutate(metric = factor(metric, levels = c("rps","brier"),
                         labels = c("RPS","Brier")))

p_pooled_metrics <- ggplot(cv_pooled_long, aes(x = reorder(model_name, value), y = value,
                           fill = model_name == cv_summary$model_name[1])) +
  geom_col(width = 0.65) +
  scale_fill_manual(values = c("FALSE" = "grey70", "TRUE" = "#1a9850"), guide = "none") +
  facet_wrap(~metric, scales = "free_y") +
  labs(title = "Pooled LOSO-CV Metrics (all seasons, out-of-fold)",
       subtitle = "Green = best model from LOSO-CV; lower is better for all metrics",
       x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
print(p_pooled_metrics)
save_plot(p_pooled_metrics, "cv_pooled_metrics")

# 11c. Pooled calibration comparison
p_calib_cmp <- plot_calibration_comparison(
  results_list = setNames(lapply(names(model_defs), function(m_id) {
    p  <- cv_preds[[m_id]]
    a  <- cv_actuals[[m_id]]
    ok <- complete.cases(p)
    calculate_all_metrics(as.data.frame(p)[ok, ], a[ok])
  }), sapply(model_defs, \(m) m$name))
)
print(p_calib_cmp)
save_plot(p_calib_cmp, "cv_calibration_comparison", height = 4.2)

# 11d. RPS over game time — value of in-game information
# The pre-match baseline cannot react to the score, so its RPS should stay
# flat while the in-game models' RPS shrinks toward 0 as the outcome
# becomes determined. Averaged over all time points the gap is diluted;
# this view shows where the in-game models earn their advantage.
rps_by_time <- bind_rows(lapply(names(model_defs), function(m_id) {
  cbind(data.frame(model = m_id, model_name = model_defs[[m_id]]$name,
                   stringsAsFactors = FALSE),
        rps_by_time_curve(cv_preds[[m_id]], cv_actuals[[m_id]], cv_time, cv_side))
}))

# The figure itself is drawn by 06_rps_by_time.R, which reads the table written
# here. Only this script can produce the numbers -- they need every model's
# out-of-fold predictions, so they cost a full run -- whereas how the figure
# looks is something one wants to change without paying for that again.
# ---- Keep the out-of-fold predictions ----------------------------------------
# Everything this script reports -- the comparison table, the scalars, the time
# curve -- is a summary of these predictions. The fitting is what costs hours;
# the summarising costs seconds. Writing them out means that a later change to
# how a number is reported never again requires refitting: 07 reads this file
# and rebuilds. Not committed (see .gitignore); it is a cache, not an output.
# gzip, not xz: on this data xz takes about 40 seconds against 4 for gzip and
# saves 4 MB, which is a poor trade for a file that is never committed.
#
# Wrapped in try() deliberately. This runs at the very end of a job measured in
# hours, and a cache is a convenience: a full disk or a read-only directory
# must not be allowed to throw away a finished run before the table and the
# curve below have been written.
cache_path <- here("cache", "cv_predictions.rds")
cached_ok <- tryCatch({
  dir.create(dirname(cache_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(list(cv_preds = cv_preds, cv_actuals = cv_actuals,
               cv_time = cv_time, cv_side = cv_side,
               models = names(model_defs), written = Sys.time()),
          cache_path, compress = "gzip")
  TRUE
}, error = function(e) { warning("could not cache the predictions (", 
  conditionMessage(e), "); the run is unaffected, but 07 will need a live ",
  "session until 01 is run again"); FALSE })
if (cached_ok)
  cat(sprintf("  cached preds -> %s (%.1f MB)\n",
              cache_path, file.size(cache_path) / 1024^2))

write.csv(rps_by_time, file.path(OUTPUT_DIR, "rps_by_time.csv"), row.names = FALSE)
cat(sprintf("  saved data   -> %s\n", file.path(OUTPUT_DIR, "rps_by_time.csv")))
cat("  run 06_rps_by_time.R to draw generated/cv_rps_by_time.pdf from it\n")

# 11e. RPS skill score relative to the betting-odds baseline
# skill = 1 - RPS_model / RPS_baseline per time bin (> 0 = beats baseline)
rps_skill <- rps_by_time |>
  left_join(
    rps_by_time |> filter(model == "baseline") |>
      dplyr::select(time_bin, rps_baseline = rps),
    by = "time_bin"
  ) |>
  filter(model != "baseline") |>
  mutate(skill = 1 - rps / rps_baseline)

p_rps_skill <- ggplot(rps_skill, aes(x = time_bin, y = skill, color = model_name)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_y_continuous(labels = scales::label_percent()) +
  labs(title = "RPS Skill Score vs. Betting-Odds Baseline (LOSO-CV)",
       subtitle = "Share of baseline RPS eliminated by in-game information; above 0 = better than baseline",
       x = "Game time [min]", y = "Skill score", color = "Model") +
  theme_minimal()
print(p_rps_skill)
save_plot(p_rps_skill, "cv_rps_skill")

# 11f. Calibration over game time — best model only
# Reliability diagram (observed vs. predicted probability) faceted into
# 5-minute game-time bins, so the winning model's calibration can be read
# phase by phase rather than only pooled over the whole match. Built from the
# same pooled out-of-fold predictions; "CE" per facet is the mean calibration
# error in that time window.
best_p   <- cv_preds[[best_model_id]]
best_a   <- cv_actuals[[best_model_id]]
ok_best  <- complete.cases(best_p)
p_calib_time <- plot_calibration_over_time(
  predictions = as.data.frame(best_p)[ok_best, ],
  actuals     = best_a[ok_best],
  time_var    = cv_time[ok_best] / 60,          # seconds -> minutes
  model_name  = cv_summary$model_name[1]
)
print(p_calib_time)
save_plot(p_calib_time, "cv_calibration_over_time", height = 4.6)

# The same diagram for the strongest tree ensemble. The pooled ECE of the two
# models is nearly identical, which is exactly what a single aggregate number
# can hide: this is what shows where the ensemble's calibration actually goes,
# and it is the appendix counterpart to the figure above.
if (!is.na(best_tree)) {
  tree_p  <- cv_preds[[best_tree]]
  tree_a  <- cv_actuals[[best_tree]]
  ok_tree <- complete.cases(tree_p)
  p_calib_time_tree <- plot_calibration_over_time(
    predictions = as.data.frame(tree_p)[ok_tree, ],
    actuals     = tree_a[ok_tree],
    time_var    = cv_time[ok_tree] / 60,
    model_name  = cv_summary$model_name[cv_summary$model == best_tree]
  )
  print(p_calib_time_tree)
  save_plot(p_calib_time_tree, "cv_calibration_over_time_ens", height = 4.6)

  # Worst-phase competitive ECE of both, so the appendix text can name the gap
  # the pooled ECE conceals rather than only pointing at the figures. Keyed on
  # the literal "gam" id, matching how \gamRPS and the other GAM macros above
  # are written, rather than on best_model_id.
  save_scalar("gamMaxECE",
              formatC(worst_phase_ece(as.data.frame(cv_preds[["gam"]]),
                                      cv_actuals[["gam"]], cv_time),
                      format = "f", digits = 3))
  save_scalar("bestEnsMaxECE",
              formatC(worst_phase_ece(as.data.frame(tree_p), tree_a, cv_time),
                      format = "f", digits = 3))
}

# =============================================================================
# 12. Save best model ID for downstream scripts
# =============================================================================
cat(sprintf(
  "\n# ============================================================\n
  # Best model for 02_tune_blend_parameters.R and
  # 03_goalkeeper_decision_analysis.R:\n
  # Set at the top of those scripts:\n
  #   best_model_id <- \"%s\"   # %s\n
  # ============================================================\n",
  best_model_id, cv_summary$model_name[1]
))
