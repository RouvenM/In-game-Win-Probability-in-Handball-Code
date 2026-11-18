library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)

# --- Core Metrics ---

#' Compute Ranked Probability Score (RPS)
#' @param predictions DataFrame with columns 'win', 'draw', 'loss' (probabilities)
#' @param actual_outcome Vector with true results ("win", "draw" or "loss")
compute_rps <- function(predictions, actual_outcome) {
    # Sequence: loss, draw, win for cumulative probabilities
    outcomes <- c("loss", "draw", "win")
    pred_cdf <- t(apply(predictions[, outcomes], 1, cumsum))

    # True results in CDF-form: 1 at and above the outcome's ordinal position.
    # Must compare positions, not strings — `outcomes >= x` collates
    # alphabetically ("draw" < "loss" < "win") and corrupts the CDF for
    # loss and draw outcomes.
    true_idx <- match(actual_outcome, outcomes)
    true_cdf <- 1 * outer(true_idx, seq_along(outcomes), "<=")

    # RPS = sum of squared differences / (M - 1)
    rps_values <- rowSums((pred_cdf - true_cdf)^2) / (length(outcomes) - 1)
    return(rps_values)
}

#' Compute Brier Score
#' @param predictions DataFrame with columns = classes
#' @param actual_outcome Vector with the true results
compute_brier_score <- function(predictions, actual_outcome) {
    classes <- colnames(predictions)
    if (!all(actual_outcome %in% classes)) {
        stop("actual_outcome contains classes not found in predictions.")
    }

    # One-Hot-Encoding of actual results
    actual_matrix <- sapply(classes, function(cls) as.numeric(actual_outcome == cls))

    # Brier Score = sum of squared differences
    brier_scores <- rowSums((predictions - actual_matrix)^2)
    return(brier_scores)
}

#' Manual Log Loss calculation
log_loss_manual <- function(pred_probs, true_labels) {
    eps <- 1e-15
    # Must be a matrix before clipping: pmax/pmin on a data.frame recycle the
    # scalar against length(df) = ncol and silently inject NA wherever a value
    # actually needs clipping (exact 0/1 probabilities, e.g. from RF or GAM)
    pred_probs <- as.matrix(pred_probs)
    pred_probs <- pmin(pmax(pred_probs, eps), 1 - eps)
    idx <- cbind(seq_len(nrow(pred_probs)), match(true_labels, colnames(pred_probs)))
    -mean(log(pred_probs[idx]))
}

# --- Calibration Functions ---

#' Compute overall calibration error and per-class data
compute_calibration_error <- function(predictions, actual_outcome, bins = 20) {
    classes <- colnames(predictions)
    if (!all(actual_outcome %in% classes)) {
        stop("actual_outcome contains classes not found in predictions.")
    }

    n <- length(actual_outcome)
    breaks <- seq(0, 1, length.out = bins + 1)
    plot_data_list <- list()

    per_class <- sapply(classes, function(cls) {
        p_cls <- as.numeric(predictions[, cls])
        bin_id <- cut(p_cls, breaks = breaks, include.lowest = TRUE, labels = FALSE)

        e_yb <- numeric(bins)
        n_yb <- numeric(bins)
        pred_means <- numeric(bins)
        obs_rates <- numeric(bins)

        for (b in seq_len(bins)) {
            idx_b <- which(bin_id == b)
            if (length(idx_b) > 0) {
                pred_means[b] <- mean(p_cls[idx_b])
                obs_rates[b] <- mean(actual_outcome[idx_b] == cls)
                e_yb[b] <- abs(pred_means[b] - obs_rates[b])
                n_yb[b] <- length(idx_b)
            } else {
                pred_means[b] <- NA
                obs_rates[b] <- NA
            }
        }

        plot_data_list[[cls]] <<- data.frame(
            class = cls,
            bin = seq_len(bins),
            pred_mean = pred_means,
            obs_rate = obs_rates,
            n_bin = n_yb
        )

        n_y_obs <- sum(n_yb * !is.na(e_yb))
        if (n_y_obs > 0) sum(n_yb * e_yb, na.rm = TRUE) / n_y_obs else NA
    })

    n_y_all <- table(actual_outcome)
    e_overall <- sum(n_y_all[names(n_y_all) %in% names(per_class)] * per_class, na.rm = TRUE) / n

    return(list(
        e_overall = e_overall,
        per_class = per_class,
        plot_data = do.call(rbind, plot_data_list)
    ))
}

#' Compute calibration error faceted by time
compute_calibration_error_facet_time <- function(predictions, actual_outcome, time_var,
                                                 breaks_minutes = seq(0, 60, by = 5),
                                                 bins_prob = 20) {
    
    classes <- colnames(predictions)
    time_bins <- cut(time_var,
        breaks = breaks_minutes, include.lowest = TRUE,
        labels = paste(head(breaks_minutes, -1), "-", tail(breaks_minutes, -1), "min")
    )

    prob_breaks <- seq(0, 1, length.out = bins_prob + 1)
    plot_data_list <- list()

    for (tb in levels(time_bins)) {
        idx_t <- which(time_bins == tb)
        if (length(idx_t) == 0) next

        per_class_tb <- lapply(classes, function(cls) {
            p_cls <- as.numeric(predictions[idx_t, cls])
            bin_id <- cut(p_cls, breaks = prob_breaks, include.lowest = TRUE, labels = FALSE)

            pred_means <- numeric(bins_prob)
            obs_rates <- numeric(bins_prob)
            n_yb <- numeric(bins_prob)

            for (b in seq_len(bins_prob)) {
                idx_b <- which(bin_id == b)
                if (length(idx_b) > 0) {
                    pred_means[b] <- mean(p_cls[idx_b])
                    obs_rates[b] <- mean(actual_outcome[idx_t][idx_b] == cls)
                    n_yb[b] <- length(idx_b)
                } else {
                    pred_means[b] <- NA
                    obs_rates[b] <- NA
                    n_yb[b] <- 0
                }
            }

            data.frame(
                class = cls, time_bin = tb, prob_bin = seq_len(bins_prob),
                pred_mean = pred_means, obs_rate = obs_rates, n_bin = n_yb,
                cal_error_class_tb = sum(n_yb * abs(pred_means - obs_rates), na.rm = TRUE) / sum(n_yb)
            )
        })
        plot_data_list[[tb]] <- do.call(rbind, per_class_tb)
    }

    out <- do.call(rbind, plot_data_list)
    # Keep the chronological order of the windows. `tb` is stored as a character
    # above, so without this the facets sort alphabetically and "5 - 10 min"
    # lands between "45 - 50 min" and "50 - 55 min".
    out$time_bin <- factor(out$time_bin, levels = levels(time_bins))
    return(out)
}

#' Expected Calibration Error (ECE) as per paper
compute_ece_paper <- function(predictions, actual_outcome, bins = 20) {
    classes <- colnames(predictions)
    n <- length(actual_outcome)
    breaks <- seq(0, 1, length.out = bins + 1)

    p_hat <- apply(predictions, 1, max)
    class_hat <- classes[apply(predictions, 1, which.max)]
    correct <- as.integer(class_hat == actual_outcome)
    bin_id <- cut(p_hat, breaks = breaks, include.lowest = TRUE, labels = FALSE)

    e_yb <- numeric(bins)
    n_yb <- numeric(bins)
    pred_mean <- numeric(bins)
    obs_rate <- numeric(bins)

    for (b in seq_len(bins)) {
        idx_b <- which(bin_id == b)
        if (length(idx_b) > 0) {
            pred_mean[b] <- mean(p_hat[idx_b])
            obs_rate[b] <- mean(correct[idx_b])
            e_yb[b] <- abs(pred_mean[b] - obs_rate[b])
            n_yb[b] <- length(idx_b)
        } else {
            pred_mean[b] <- NA
            obs_rate[b] <- NA
        }
    }

    list(
        ece = sum((n_yb / n) * e_yb, na.rm = TRUE),
        plot_data = data.frame(bin = seq_len(bins), pred_mean = pred_mean, obs_rate = obs_rate, n_bin = n_yb)
    )
}

# --- High-level Wrapper Functions ---

#' Calculate multiple metrics for a set of predictions
calculate_all_metrics <- function(predictions, actuals) {
    rps <- mean(compute_rps(predictions, actuals))
    cal <- compute_calibration_error(as.data.frame(predictions), actuals)
    brier <- mean(compute_brier_score(predictions, actuals))
    logloss <- log_loss_manual(as.matrix(predictions), actuals)

    return(list(rps = rps, ece = cal$e_overall, brier = brier, logloss = logloss, calibration_details = cal))
}

#' Plot comparison of calibration across models
plot_calibration_comparison <- function(results_list) {
    plot_data <- map_df(names(results_list), function(name) {
        df <- results_list[[name]]$calibration_details$plot_data
        df$model <- name
        df
    })

    ggplot(plot_data, aes(x = pred_mean, y = obs_rate, size = n_bin, color = model)) +
        geom_point(alpha = 0.6) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
        facet_wrap(~class) +
        theme_minimal() +
        labs(
            title = "Calibration Comparison", subtitle = "Observed vs Predicted Probabilities by Class",
            x = "Predicted Probability", y = "Observed Rate", size = "Number of Observations", color = "Model"
        )
}

#' Plot calibration error over time for a model
plot_calibration_over_time <- function(predictions, actuals, time_var, model_name = "") {
    df_plot <- compute_calibration_error_facet_time(as.data.frame(predictions), actuals, time_var)
    ce_summary <- df_plot %>%
        group_by(time_bin) %>%
        summarise(cal_error = mean(cal_error_class_tb, na.rm = TRUE), .groups = "drop")

    ggplot(df_plot, aes(x = pred_mean, y = obs_rate)) +
        geom_point(aes(size = n_bin, color = class), alpha = 0.6) +
        geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
        facet_wrap(~time_bin) +
        geom_text(
            data = ce_summary, aes(x = 0.9, y = 0.05, label = sprintf("CE: %.3f", cal_error)),
            inherit.aes = FALSE, hjust = 1, size = 3, color = "black"
        ) +
        # Twelve facets share the page width, so the axis text has very little
        # room. Three breaks instead of the default five, with short hand-written
        # labels: ggplot's default formatter renders these breaks as
        # "0.00"/"0.50"/"1.00", which is what ran the labels together.
        scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"),
                           limits = c(0, 1), expand = expansion(mult = 0.08)) +
        scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"),
                           limits = c(0, 1), expand = expansion(mult = 0.08)) +
        labs(
            x = "Predicted Probability", y = "Observed Rate",
            title = paste("Calibration per Time Interval", ifelse(model_name != "", paste("-", model_name), "")),
            size = "Num Obs", color = "Class"
        ) +
        theme_minimal() +
        theme(axis.text = element_text(size = 7),
              plot.title = element_text(size = 11))
}

#' Variable importance plot for an ordinal_xgb model
#' Gain is averaged across both cumulative threshold models.
#' A per-threshold facet is also shown for comparison.
#' @param fit_ord An ordinal_xgb object from fit_ordinal_xgb() / fit_ordinal_xgb_fast()
#' @param title Plot title
plot_variable_importance_ordinal <- function(fit_ord, title = "Variable Importance") {
  imp1 <- xgb.importance(model = fit_ord$fit1,
                          feature_names = fit_ord$feature_names) |>
    as.data.frame() |>
    mutate(threshold = "loss \u2192 \u2265draw")

  imp2 <- xgb.importance(model = fit_ord$fit2,
                          feature_names = fit_ord$feature_names) |>
    as.data.frame() |>
    mutate(threshold = "\u2264draw \u2192 win")

  imp_avg <- bind_rows(imp1, imp2) |>
    group_by(Feature) |>
    summarise(Gain = mean(Gain), .groups = "drop") |>
    arrange(Gain)

  # Panel 1: averaged importance
  p_avg <- ggplot(imp_avg, aes(x = Gain, y = reorder(Feature, Gain))) +
    geom_col() +
    scale_x_continuous(labels = scales::label_percent()) +
    labs(title = title,
         subtitle = "Mean Gain across both cumulative thresholds",
         x = "Mean Gain", y = NULL) +
    theme_minimal()

  # Panel 2: per-threshold comparison
  imp_both <- bind_rows(imp1, imp2) |>
    mutate(Feature = factor(Feature, levels = imp_avg$Feature))

  p_facet <- ggplot(imp_both, aes(x = Gain, y = Feature)) +
    geom_col() +
    scale_x_continuous(labels = scales::label_percent()) +
    facet_wrap(~threshold, scales = "free_x") +
    labs(title = title,
         subtitle = "Gain per cumulative threshold",
         x = "Gain", y = NULL) +
    theme_minimal()

  print(p_avg)
  print(p_facet)
  invisible(list(avg = imp_avg, by_threshold = imp_both))
}

#' Create a summary table of performance metrics
create_performance_leaderboard <- function(results_list) {
    map_df(names(results_list), function(name) {
        res <- results_list[[name]]
        data.frame(Model = name, RPS = res$rps, ECE = res$ece, Brier = res$brier, LogLoss = res$logloss)
    }) %>% arrange(RPS)
}

# =============================================================================
# Worst-phase calibration on competitive states
# =============================================================================
# Shared by 01_model_comparison.R and 02_tune_blend_parameters.R so the two
# cannot drift apart: a worst-phase figure computed one way for the model
# comparison and another way for the blend tuning would not be comparable, and
# the manuscript quotes them side by side.
#
# Rationale for the "competitive" restriction. Late in a match most games are
# decided, so predicted probabilities pile up near 0 and 1. Those near-certain
# predictions are trivially well calibrated and, because ECE is population
# weighted, they drag a late bin's ECE DOWN. A raw cross-phase comparison is
# therefore confounded by how decided the games are rather than by genuine
# miscalibration. Restricting to states in which no outcome is yet near-certain
# removes that confound and focuses on the decision-relevant situations.
COMPET_THRESH <- 0.90   # a state is competitive if no class probability reaches this
BIN_WIDTH     <- 300    # seconds per game-time bin
MIN_BIN_N     <- 200    # bins with fewer rows are too noisy to score

#' TRUE for rows where no predicted class probability reaches COMPET_THRESH.
is_competitive <- function(P, thresh = COMPET_THRESH) {
  do.call(pmax, as.list(as.data.frame(P))) < thresh
}

#' ECE within each game-time bin over the rows selected by `keep`.
#' Returns bin index, row count and ECE (NA for bins below `min_n`).
ece_by_time_bin <- function(P, A, time, keep,
                            bin_width = BIN_WIDTH, min_n = MIN_BIN_N) {
  Pk <- as.data.frame(P)[keep, ]; Ak <- A[keep]
  tb <- floor(time[keep] / bin_width)
  bins <- sort(unique(tb))
  data.frame(
    bin = bins,
    n   = vapply(bins, function(b) sum(tb == b), integer(1)),
    ece = vapply(bins, function(b) {
      idx <- which(tb == b)
      if (length(idx) < min_n) return(NA_real_)
      compute_calibration_error(Pk[idx, ], Ak[idx])$e_overall
    }, numeric(1))
  )
}

#' Max-norm calibration criterion: the ECE of the worst-calibrated game-time
#' bin, computed on competitive states only. Lower means the model is uniformly
#' trustworthy across the match rather than merely on average.
worst_phase_ece <- function(P, A, time, competitive_only = TRUE, ...) {
  ok   <- complete.cases(as.data.frame(P))
  keep <- if (competitive_only) ok & is_competitive(P) else ok
  v <- ece_by_time_bin(P, A, time, keep, ...)$ece
  v <- v[is.finite(v)]
  # A forecaster that never leaves a state competitive -- the leading-team rule
  # states certainty on every possession -- retains no window at all, and
  # max() of nothing is -Inf, which then prints as a very good score and sorts
  # ahead of every real one. The criterion is undefined there, so say so.
  if (!length(v)) NA_real_ else max(v)
}
