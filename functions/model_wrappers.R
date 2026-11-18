library(mgcv)
library(nnet)
library(MASS)
library(ranger)
library(xgboost)
library(caret)
library(dplyr)
library(purrr)
library(skellam)

#' Fit a model with optional hyperparameter tuning
#' @param model_type String: "multinom", "ordinal", "gam", "rf", "xgb"
#' @param data Training data frame
#' @param formula Model formula
fit_model <- function(model_type, data, formula, ...) {
  cat("Fitting model type:", model_type, "...\n")

  if (model_type == "multinom") {
    return(nnet::multinom(formula, data = data, trace = FALSE, ...))
  } else if (model_type == "ordinal") {
    # droplevels() removes factor levels with 0 observations that arise from
    # time-filtering (dplyr::filter keeps all original levels). Without this,
    # polr's design matrix is rank-deficient and it silently drops coefficients,
    # causing X %*% coefficients dimension mismatches during prediction.
    data <- droplevels(data)
    return(MASS::polr(formula, data = data, Hess = TRUE, ...))
  } else if (model_type == "gam") {
    m <- mgcv::bam(formula,
      family = ocat(R = 3),
      data = data,
      method = "fREML",
      discrete = TRUE, ...
    )
    # Store training matchid levels so predict_model can substitute unseen
    # levels before building the RE design matrix
    if ("matchid" %in% names(data)) {
      attr(m, "matchid_levels") <- levels(data$matchid)
    }
    return(m)
  } else if (model_type == "rf") {
    x_vars <- all.vars(formula)[-1]

    # RF accuracy is fairly insensitive to min.node.size/num.trees; the one
    # parameter worth a grid search is mtry (variables tried per split).
    # Leaving min.node.size unset uses ranger's own default (10 for
    # probability forests).
    y_var     <- as.character(formula[[2]])
    mtry_grid <- seq(1, min(length(x_vars), 5), by = 1)

    if (!"matchid" %in% names(data)) {
      stop("fit_model (rf): `data` must include `matchid` for match-grouped tuning folds.")
    }

    # Tuning subsample drawn by MATCH, not by row. Sequences from one match are
    # strongly correlated, so a random row sample puts the same match on both
    # sides of every tuning split---the leakage the outer design closes one
    # level up, and which the internal tuning must close too if the comparison
    # is to be fair to the ensembles.
    matches <- unique(as.character(data$matchid))
    tuning_data <- if (length(matches) > 400) {
      data[as.character(data$matchid) %in% sample(matches, 400), , drop = FALSE]
    } else data
    cv_folds <- build_group_folds(tuning_data$matchid, nfold = 5)

    # Score candidates by the Brier score of a PROBABILITY forest---the same
    # kind of model we ultimately fit, and a proper scoring rule. An earlier
    # version tuned a CLASSIFICATION forest on its out-of-bag misclassification
    # rate, which selects for 0-1 accuracy and is blind to calibration; that is
    # the wrong objective for a model whose output is read as a probability, and
    # it would leave the forest tuned against the criterion it is judged on.
    best_error <- Inf
    best_mtry  <- mtry_grid[1]
    for (mtry in mtry_grid) {
      err <- mean(vapply(cv_folds, function(idx) {
        fit_k <- ranger(formula, data = tuning_data[-idx, , drop = FALSE],
                        mtry = mtry, num.trees = 100, probability = TRUE,
                        verbose = FALSE)
        P <- predict(fit_k, data = tuning_data[idx, , drop = FALSE])$predictions
        .brier_of(P, tuning_data[[y_var]][idx])
      }, numeric(1)))
      if (err < best_error) {
        best_error <- err
        best_mtry  <- mtry
      }
    }
    cat(sprintf("Best RF params: mtry = %d (match-grouped CV Brier = %.4f)\n",
                best_mtry, best_error))

    return(ranger(formula,
      data = data,
      mtry = best_mtry,
      probability = TRUE,
      importance = "impurity",
      num.trees = 500, ...
    ))
  } else if (model_type == "xgb") {
    y_var <- as.character(formula[[2]])

    # Guarantee X, y and matchid all reference the same rows: model.matrix()
    # silently drops incomplete cases, which would otherwise misalign the labels
    # and the match-grouping used for the CV folds below (same handling as
    # fit_ordinal_xgb).
    keep <- complete.cases(data[, all.vars(formula), drop = FALSE])
    data <- data[keep, , drop = FALSE]
    y_labels <- as.numeric(as.factor(data[[y_var]])) - 1

    X <- model.matrix(formula, data = data)[, -1]
    dtrain <- xgb.DMatrix(data = X, label = y_labels)

    # Internal tuning folds are grouped by MATCH, not drawn at random over rows.
    # Attacking sequences from one match share an outcome, so random row folds
    # would put correlated rows on both sides of the split and leak information
    # into hyperparameter selection---the same leakage the outer leave-one-
    # season-out design closes one level up. fit_ordinal_xgb does this too; the
    # two XGBoost variants must be tuned under identical conditions for the
    # standard-versus-ordinal comparison to be meaningful.
    if (!"matchid" %in% names(data)) {
      stop("fit_model (xgb): `data` must include `matchid` for match-grouped tuning folds.")
    }
    cv_folds <- build_group_folds(data$matchid, nfold = 5)

    # Simple tuning for eta and max_depth
    tune_grid <- expand.grid(
      max_depth = c(4, 6, 8),
      eta = c(0.1, 0.3)
    )

    best_logloss <- Inf
    best_params_tune <- list()

    cat("Starting XGB hyperparameter tuning...\n")
    for (i in 1:nrow(tune_grid)) {
      params <- list(
        objective = "multi:softprob",
        num_class = 3,
        eval_metric = "mlogloss",
        max_depth = tune_grid$max_depth[i],
        eta = tune_grid$eta[i]
      )

      cv <- xgb.cv(
        params = params, data = dtrain, nrounds = 50,
        folds = cv_folds, early_stopping_rounds = 5, verbose = FALSE
      )

      min_logloss <- min(cv$evaluation_log$test_mlogloss_mean)
      if (min_logloss < best_logloss) {
        best_logloss <- min_logloss
        best_params_tune <- params
        # Fallback to CV max rounds if best_iteration is missing or 0
        best_params_tune$nrounds <- if (!is.null(cv$best_iteration) && cv$best_iteration > 0) cv$best_iteration else 50
      }
    }

    cat(
      "Best XGB params: max_depth =", best_params_tune$max_depth,
      ", eta =", best_params_tune$eta, ", nrounds =", best_params_tune$nrounds, "\n"
    )

    fit <- xgb.train(params = best_params_tune, data = dtrain, nrounds = best_params_tune$nrounds)

    attr(fit, "formula") <- formula
    attr(fit, "feature_names") <- colnames(X)
    return(fit)
  } else {
    stop(sprintf("Unknown model_type: '%s'", model_type))
  }
}

#' Fast XGBoost fit with fixed hyperparameters (no internal CV tuning)
#' Intended for use inside outer hyperparameter search loops where repeated
#' full tuning would be prohibitively slow. Uses sensible defaults.
#' @param data Training data frame
#' @param formula Model formula (outcome must be a factor with levels loss/draw/win)
#' @param max_depth Tree depth (default 6)
#' @param eta Learning rate (default 0.1)
#' @param nrounds Number of boosting rounds (default 100)
fit_model_fast_xgb <- function(data, formula,
                                max_depth = 6,
                                eta       = 0.1,
                                nrounds   = 100) {
  y_var    <- as.character(formula[[2]])
  y_labels <- as.numeric(as.factor(data[[y_var]])) - 1

  X      <- model.matrix(formula, data = data)[, -1]
  dtrain <- xgb.DMatrix(data = X, label = y_labels)

  params <- list(
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    max_depth        = max_depth,
    eta              = eta,
    subsample        = 0.8,
    colsample_bytree = 0.8
  )

  fit <- xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)

  attr(fit, "formula")       <- formula
  attr(fit, "feature_names") <- colnames(X)
  return(fit)
}

# --- Ordinal XGBoost (cumulative link) ---------------------------------------
# Fits two binary classifiers for the two cumulative thresholds:
#   Threshold 1: P(Y >= draw)  →  loss  vs. {draw, win}
#   Threshold 2: P(Y >= win)   →  {loss, draw} vs. win
# Class probabilities are recovered as:
#   P(loss) = 1 - P(>=draw)
#   P(draw) = P(>=draw) - P(>=win)    [clipped to >= 0]
#   P(win)  = P(>=win)

#' Build match-grouped folds for xgb.cv (returns test-set row indices per fold)
#'
#' Possessions from the same match must never be split across an internal
#' CV fold's train/test parts during hyperparameter tuning — xgb.cv's default
#' folds are random row splits, so rows from the same match routinely land on
#' both sides, leaking within-match correlation into early-stopping / grid
#' selection and making it look more generalizable than it is (see Brill,
#' Yurko & Wyner 2026 on the dependence structure of play-by-play sports
#' data). Grouping by match closes that leak for the tuning step specifically
#' — the outer LOSO-CV in 01_model_comparison.R already holds out whole
#' seasons for the actual reported metrics.
#' @param group Vector of grouping keys (e.g. matchid), one per row
#' @param nfold Number of folds
#' Brier score of a probability matrix against observed labels.
#' Columns of `P` are matched to the labels by name, so a fold in which some
#' class is absent cannot silently misalign the one-hot encoding.
.brier_of <- function(P, y) {
  Y <- matrix(0, nrow = length(y), ncol = ncol(P),
              dimnames = list(NULL, colnames(P)))
  Y[cbind(seq_along(y), match(as.character(y), colnames(P)))] <- 1
  mean(rowSums((P - Y)^2))
}

build_group_folds <- function(group, nfold = 5) {
  groups     <- unique(group)
  group_fold <- setNames(sample(rep_len(seq_len(nfold), length(groups))),
                         as.character(groups))
  row_fold   <- group_fold[as.character(group)]
  lapply(seq_len(nfold), function(f) which(row_fold == f))
}

#' Fit ordinal XGBoost with full hyperparameter tuning (grid search + CV)
#' @param data Training data frame; must include `matchid` so the internal
#'   tuning CV can be grouped by match (see build_group_folds)
#' @param formula Model formula (outcome factor with levels loss/draw/win)
fit_ordinal_xgb <- function(data, formula) {
  y_var <- as.character(formula[[2]])

  # Guarantee X, y and matchid all reference the same rows — model.matrix()
  # silently drops incomplete cases, which would otherwise misalign the
  # match-grouping used for the CV folds below.
  keep <- complete.cases(data[, all.vars(formula), drop = FALSE])
  data <- data[keep, ]
  y_fac <- data[[y_var]]

  y1 <- as.integer(y_fac %in% c("draw", "win"))  # threshold 1
  y2 <- as.integer(y_fac == "win")                # threshold 2

  X <- model.matrix(formula, data = data)[, -1]

  if (!"matchid" %in% names(data)) {
    stop("fit_ordinal_xgb: `data` must include `matchid` for match-grouped tuning folds.")
  }
  cv_folds <- build_group_folds(data$matchid, nfold = 5)

  # eta/max_depth alone were undertuned: no regularisation knob (min_child_weight
  # matters a lot on correlated per-match rows) and a 50-round cap starved the
  # slower eta values before they could converge.
  tune_grid <- expand.grid(
    max_depth        = c(4, 6, 8),
    eta              = c(0.03, 0.1, 0.3),
    min_child_weight = c(1, 5, 10)
  )

  fit_binary_tuned <- function(y_bin, label) {
    dtrain       <- xgb.DMatrix(data = X, label = y_bin)
    best_logloss <- Inf
    best_params  <- list()

    for (i in seq_len(nrow(tune_grid))) {
      params <- list(
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        max_depth        = tune_grid$max_depth[i],
        eta              = tune_grid$eta[i],
        min_child_weight = tune_grid$min_child_weight[i],
        subsample        = 0.8,
        colsample_bytree = 0.8
      )
      cv <- xgb.cv(params = params, data = dtrain, nrounds = 300,
                   folds = cv_folds, early_stopping_rounds = 15, verbose = FALSE)
      min_ll <- min(cv$evaluation_log$test_logloss_mean)
      if (min_ll < best_logloss) {
        best_logloss <- min_ll
        best_params  <- params
        best_params$nrounds <- if (!is.null(cv$best_iteration) && cv$best_iteration > 0)
          cv$best_iteration else 300
      }
    }
    cat(sprintf("  [%s] max_depth: %d  eta: %.2f  min_child_weight: %d  nrounds: %d  (logloss: %.5f)\n",
                label, best_params$max_depth, best_params$eta,
                best_params$min_child_weight, best_params$nrounds, best_logloss))
    xgb.train(params = best_params, data = dtrain, nrounds = best_params$nrounds, verbose = 0)
  }

  cat("Fitting ordinal XGB – threshold 1 (loss vs. >=draw)...\n")
  fit1 <- fit_binary_tuned(y1, "threshold 1")
  cat("Fitting ordinal XGB – threshold 2 (<=draw vs. win)...\n")
  fit2 <- fit_binary_tuned(y2, "threshold 2")

  structure(
    list(fit1 = fit1, fit2 = fit2, formula = formula, feature_names = colnames(X)),
    class = "ordinal_xgb"
  )
}

#' Fast ordinal XGBoost with fixed hyperparameters — for use inside CV loops
#' @param data Training data frame
#' @param formula Model formula
#' @param max_depth,eta,nrounds Fixed XGBoost hyperparameters
fit_ordinal_xgb_fast <- function(data, formula,
                                  max_depth = 6,
                                  eta       = 0.1,
                                  nrounds   = 100) {
  y_var <- as.character(formula[[2]])
  y_fac <- data[[y_var]]

  y1 <- as.integer(y_fac %in% c("draw", "win"))
  y2 <- as.integer(y_fac == "win")

  X <- model.matrix(formula, data = data)[, -1]

  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = max_depth,
    eta              = eta,
    subsample        = 0.8,
    colsample_bytree = 0.8
  )

  fit1 <- xgb.train(params = params,
                    data = xgb.DMatrix(X, label = y1),
                    nrounds = nrounds, verbose = 0)
  fit2 <- xgb.train(params = params,
                    data = xgb.DMatrix(X, label = y2),
                    nrounds = nrounds, verbose = 0)

  structure(
    list(fit1 = fit1, fit2 = fit2, formula = formula, feature_names = colnames(X)),
    class = "ordinal_xgb"
  )
}

# --- Ordinal Random Forest (cumulative link) ---------------------------------
# Same construction as ordinal XGBoost: two binary probability forests for the
# cumulative thresholds, recombined into class probabilities.

#' Fit ordinal Random Forest with light hyperparameter tuning per threshold
#' @param data Training data frame
#' @param formula Model formula (outcome factor with levels loss/draw/win)
#' @param num.trees Trees for the final fits (tuning uses 100)
fit_ordinal_rf <- function(data, formula, num.trees = 500) {
  y_var <- as.character(formula[[2]])
  y_fac <- data[[y_var]]
  f_bin <- update(formula, y_bin ~ .)

  # RF accuracy is fairly insensitive to min.node.size/num.trees; only mtry
  # (variables tried per split) is worth tuning per threshold.
  n_predictors <- length(all.vars(formula)) - 1
  mtry_grid <- seq(1, min(n_predictors, 5), by = 1)

  fit_binary <- function(y_bin, label) {
    d <- data
    d$y_bin <- factor(y_bin, levels = c(0, 1))

    # Tuning subsample drawn by MATCH and scored on match-grouped folds, not a
    # random row sample scored out-of-bag: sequences from one match are strongly
    # correlated, so row-level resampling puts the same match on both sides of
    # the split and leaks into hyperparameter selection.
    matches <- unique(as.character(d$matchid))
    tune_d  <- if (length(matches) > 400) {
      d[as.character(d$matchid) %in% sample(matches, 400), , drop = FALSE]
    } else d
    cv_folds <- build_group_folds(tune_d$matchid, nfold = 5)

    best_error <- Inf
    best_mtry  <- mtry_grid[1]
    for (mtry in mtry_grid) {
      err <- mean(vapply(cv_folds, function(idx) {
        fit_k <- ranger(f_bin, data = tune_d[-idx, , drop = FALSE],
                        probability = TRUE, mtry = mtry, num.trees = 100,
                        verbose = FALSE)
        P <- predict(fit_k, data = tune_d[idx, , drop = FALSE])$predictions
        .brier_of(P, tune_d$y_bin[idx])
      }, numeric(1)))
      if (err < best_error) {
        best_error <- err
        best_mtry  <- mtry
      }
    }
    cat(sprintf("  [%s] mtry: %d (match-grouped CV Brier = %.4f)\n",
                label, best_mtry, best_error))
    ranger(f_bin, data = d, probability = TRUE,
           mtry = best_mtry, num.trees = num.trees, verbose = FALSE)
  }

  cat("Fitting ordinal RF – threshold 1 (loss vs. >=draw)...\n")
  fit1 <- fit_binary(as.integer(y_fac %in% c("draw", "win")), "threshold 1")
  cat("Fitting ordinal RF – threshold 2 (<=draw vs. win)...\n")
  fit2 <- fit_binary(as.integer(y_fac == "win"), "threshold 2")

  structure(list(fit1 = fit1, fit2 = fit2, formula = formula),
            class = "ordinal_rf")
}

#' Predict from an ordinal_rf model
#' @return Matrix with columns win, draw, loss (rows sum to 1)
predict_ordinal_rf <- function(fit_ord, newdata) {
  p_geq_draw <- predict(fit_ord$fit1, data = newdata)$predictions[, "1"]
  p_geq_win  <- predict(fit_ord$fit2, data = newdata)$predictions[, "1"]

  # Enforce ordinal constraint: P(win) <= P(Y >= draw)
  p_geq_win <- pmin(p_geq_win, p_geq_draw)

  mat <- cbind(
    win  = p_geq_win,
    draw = p_geq_draw - p_geq_win,
    loss = 1 - p_geq_draw
  )
  mat <- pmax(mat, 0)
  mat / rowSums(mat)
}

# --- Skellam model ------------------------------------------------------------
# Models the REMAINING goals of both teams as Poisson counts whose rates
# depend on the game state (time left, team strength, score). The final goal
# difference is current difference + D, where D = N_pos - N_opp follows a
# Skellam distribution. Outcome probabilities:
#   P(win)  = P(D > -diff),  P(draw) = P(D = -diff),  P(loss) = P(D < -diff)
# Unlike the direct classifiers, this encodes the scoreboard mechanics of the
# game explicitly and needs no separate late-game blend model to become sharp
# near the buzzer — the Poisson rates go to zero as time runs out.

#' Fit the two Poisson rate GAMs for remaining goals
#' @param data Training data (needs rem_goals_pos, rem_goals_opp)
#' @param formula_pos,formula_opp Rate model formulas
fit_skellam <- function(data,
                        formula_pos = rem_goals_pos ~
                          te(time, bet_win_prob, k = c(10, 10)) +
                          s(gls_diff_pos_opp, k = 10) + home,
                        formula_opp = rem_goals_opp ~
                          te(time, bet_win_prob, k = c(10, 10)) +
                          s(gls_diff_pos_opp, k = 10) + home) {
  cat("Fitting Skellam model (Poisson rate GAMs for remaining goals)...\n")
  fit_pos <- mgcv::bam(formula_pos, family = poisson(), data = data,
                       method = "fREML", discrete = TRUE)
  fit_opp <- mgcv::bam(formula_opp, family = poisson(), data = data,
                       method = "fREML", discrete = TRUE)
  structure(list(fit_pos = fit_pos, fit_opp = fit_opp),
            class = "skellam_model")
}

#' Predict outcome probabilities from a skellam_model
#' @return Matrix with columns win, draw, loss (rows sum to 1)
predict_skellam <- function(fit_sk, newdata) {
  lambda_pos <- pmax(as.numeric(predict(fit_sk$fit_pos, newdata,
                                        type = "response")), 1e-8)
  lambda_opp <- pmax(as.numeric(predict(fit_sk$fit_opp, newdata,
                                        type = "response")), 1e-8)
  diff <- as.numeric(newdata$gls_diff_pos_opp)

  # Final diff = diff + D with D ~ Skellam(lambda_pos, lambda_opp)
  p_draw <- skellam::dskellam(-diff,     lambda1 = lambda_pos, lambda2 = lambda_opp)
  p_loss <- skellam::pskellam(-diff - 1, lambda1 = lambda_pos, lambda2 = lambda_opp)
  p_win  <- 1 - p_loss - p_draw

  mat <- cbind(win = p_win, draw = p_draw, loss = p_loss)
  mat <- pmax(mat, 0)
  mat / rowSums(mat)
}

#' Predict from an ordinal_xgb model
#' @return Matrix with columns win, draw, loss (rows sum to 1)
predict_ordinal_xgb <- function(fit_ord, newdata) {
  X     <- model.matrix(fit_ord$formula, data = newdata)[, -1]
  dtest <- xgb.DMatrix(data = X)

  p_geq_draw <- predict(fit_ord$fit1, dtest)  # P(Y >= draw)
  p_geq_win  <- predict(fit_ord$fit2, dtest)  # P(Y >= win) = P(win)

  # Enforce ordinal constraint: P(win) <= P(Y >= draw)
  p_geq_win <- pmin(p_geq_win, p_geq_draw)

  mat <- cbind(
    win  = p_geq_win,
    draw = p_geq_draw - p_geq_win,
    loss = 1 - p_geq_draw
  )
  # Clip negative floating-point residuals and renormalise
  mat <- pmax(mat, 0)
  mat / rowSums(mat)
}

#' Standardized prediction function
#' @param fit Model object
#' @param newdata New data frame
#' @param model_type String
predict_model <- function(fit, newdata, model_type) {
  if (model_type == "multinom") {
    preds <- predict(fit, newdata, type = "probs")
    if (is.null(dim(preds))) preds <- matrix(preds, nrow = 1)
    # nnet::multinom usually returns columns in order of levels: loss, draw, win
    return(preds[, c(3, 2, 1)])
  } else if (model_type == "ordinal") {
    # Align factor levels in newdata to those seen during fitting.
    # polr drops rank-deficient columns silently; unseen levels in prediction
    # data then cause an X %*% coefficients dimension mismatch.
    for (col in names(fit$xlevels)) {
      if (col %in% names(newdata) && is.factor(newdata[[col]])) {
        newdata[[col]] <- factor(as.character(newdata[[col]]),
                                 levels = fit$xlevels[[col]])
      }
    }
    check_cols <- intersect(names(fit$xlevels), names(newdata))
    valid <- complete.cases(newdata[, check_cols, drop = FALSE])
    if (any(!valid)) {
      warning(sprintf(
        "predict_model (ordinal): %d row(s) with factor levels unseen in training predicted as NA.",
        sum(!valid)
      ))
    }
    preds_full <- matrix(NA_real_, nrow = nrow(newdata), ncol = 3)
    preds_full[valid, ] <- predict(fit, newdata[valid, ], type = "probs")
    if (is.null(dim(preds_full))) preds_full <- matrix(preds_full, nrow = 1)
    # MASS::polr returns columns in order of levels: loss, draw, win
    return(preds_full[, c(3, 2, 1)])
  } else if (model_type == "gam") {
    newdata_pred <- newdata

    # --- Fix 1: matchid (random effect smooth) ---
    # Replace unseen match IDs with a known training level so the RE design
    # matrix can be built. exclude = "s(matchid)" zeroes out this contribution,
    # giving population-level predictions — correct for OOS data.
    matchid_levels <- attr(fit, "matchid_levels")
    if (!is.null(matchid_levels) && "matchid" %in% names(newdata_pred)) {
      newdata_pred$matchid <- factor(matchid_levels[1], levels = matchid_levels)
    }

    # --- Fix 2: parametric factor terms (gk_pos, no_field_players, home, …) ---
    # bam() drops unused levels by default. A time-filtered training subset may
    # lack rare levels (e.g. no_field_players = 4 in double-penalty situations).
    # Passing those levels in newdata causes a design-matrix dimension mismatch.
    # Solution: restrict newdata levels to fit$xlevels (what the model knows).
    # Rows with unknown levels become NA and are predicted as NA.
    for (col in names(fit$xlevels)) {
      if (col %in% names(newdata_pred) && is.factor(newdata_pred[[col]])) {
        newdata_pred[[col]] <- factor(as.character(newdata_pred[[col]]),
                                      levels = fit$xlevels[[col]])
      }
    }
    check_cols <- intersect(names(fit$xlevels), names(newdata_pred))
    valid <- complete.cases(newdata_pred[, check_cols, drop = FALSE])
    if (any(!valid)) {
      warning(sprintf(
        "predict_model (gam): %d row(s) with factor levels unseen in training predicted as NA.",
        sum(!valid)
      ))
    }
    preds_full <- matrix(NA_real_, nrow = nrow(newdata_pred), ncol = 3)
    preds_full[valid, ] <- predict(fit, newdata_pred[valid, , drop = FALSE],
                                   type = "response",
                                   discrete = FALSE, exclude = "s(matchid)")
    # mgcv ocat family returns columns in order: loss, draw, win. drop = FALSE
    # keeps a 1x3 matrix when newdata has a single row (e.g. the per-state
    # goalkeeper-decision valuation in 04); otherwise it would collapse to a
    # length-3 vector and break the K-segment blend's matrix accumulation.
    return(preds_full[, c(3, 2, 1), drop = FALSE])
  } else if (model_type == "rf") {
    preds <- predict(fit, data = newdata)$predictions
    # ranger probability model returns ordered columns: loss, draw, win
    return(preds[, c(3, 2, 1), drop = FALSE])
  } else if (model_type == "xgb") {
    formula <- attr(fit, "formula")
    X <- model.matrix(formula, data = newdata)[, -1]
    # model.matrix() silently drops incomplete cases. If that happened the
    # returned probabilities would no longer line up row-for-row with newdata,
    # and every downstream metric would be computed against shifted actuals --
    # a silent, hard-to-spot error, so fail loudly instead.
    if (nrow(X) != nrow(newdata)) {
      stop(sprintf(paste0("predict_model (xgb): %d of %d newdata rows dropped by ",
                          "model.matrix (incomplete cases); predictions would ",
                          "misalign with the actuals."),
                   nrow(newdata) - nrow(X), nrow(newdata)))
    }
    dtest <- xgb.DMatrix(data = X)
    preds <- predict(fit, newdata = dtest)
    preds_mat <- matrix(preds, ncol = 3, byrow = TRUE)
    # xgboost label order depends on factor levels: 0=loss, 1=draw, 2=win
    colnames(preds_mat) <- c("loss", "draw", "win")
    return(preds_mat[, c("win", "draw", "loss"), drop = FALSE])
  } else {
    stop("Unknown model_type")
  }
}
