# 02_tune_blend_parameters.R --------------------------------------------------
# Paper workflow - Step 2: data-driven blend design for the chosen GAM
#
# Step 1 (01_model_comparison.R) establishes that the ordered-categorical GAM
# beats the tree ensembles out of sample. Given that the GAM is the model we
# carry forward, this script works out the *right* way to blend it over game
# time, rather than assuming a single fixed two-segment split.
#
# What is new relative to the earlier version of this file:
#   * The blend is generalised from two segments to up to THREE. A K-segment
#     blend fits K copies of the GAM, each on its own (overlapping) time
#     window, and combines their predictions with a smooth partition of unity
#     built from cosine ramps (each weight has zero derivative at its ends, so
#     the blended win-probability curve stays smooth). K = 1 is the no-blend
#     GAM; K = 2 reproduces the previous main/late split; K = 3 adds a distinct
#     middle-of-match fit.
#   * Blend configurations are selected data-driven by FULL leave-one-season-
#     out cross-validation (every season held out once), the same protocol as
#     01, scoring pooled out-of-fold predictions by RPS. Across this grid RPS
#     turns out to be flat, so the configurations are ranked on the worst-phase
#     competitive ECE instead (see the selection rule below).
#
# NOTE on the model spec. We tune the blend on the win-probability GAM
# (`formula_gam`), evaluated at the state each row records---the state AFTER
# that possession, the convention set in pre_process_data. It deliberately
# excludes the goalkeeper decision, which is a tactical choice valued separately
# in 04, not part of the situation a team faces. The blend lets the GAM's
# covariate effects differ between the early and the closing phase of a match.
# -----------------------------------------------------------------------------
rm(list = ls())
library(here)
library(dplyr)
library(ggplot2)

source(here("functions", "general_functions.R"))
source(here("functions", "pre_processing.R"))
source(here("functions", "model_wrappers.R"))
source(here("functions", "evaluation_functions.R"))
source(here("functions", "momentum_features.R"))
source(here("functions", "blend_functions.R"))
source(here("functions", "output_helpers.R"))

# =============================================================================
# CONFIGURATION
# =============================================================================
HALF_WIDTH      <- 300     # cosine-ramp half-width in seconds (transition = 2x)
LATE_HALF_WIDTH <- 60      # narrower ramp, for break points in the closing minutes
momentum_window <- 5       # must match 01_model_comparison.R
# =============================================================================

# =============================================================================
# 1. Data preparation (mirrors 01_model_comparison.R)
# =============================================================================
daten <- read.csv(here("raw_data", "preprocessed_data.csv"))
daten <- pre_process_data(daten)
daten <- add_momentum_features(daten, window = momentum_window)

daten <- daten |>
  ungroup() |>
  mutate(
    final_int = as.integer(factor(final, levels = c(-1, 0, 1))),
    final_ord = ordered(final, levels = c("-1", "0", "1")),
    matchid   = as.factor(matchid),
    final_rf  = factor(final, levels = c("-1", "0", "1"),
                       labels = c("loss", "draw", "win")),
    HZ        = ifelse(as.character(HZ) == "30", 1, 0),
    # Ten-second tolerance, as in 01 (see pre_process_data).
    gk_pulled = as.integer(as.character(gk_pos_10s) == "0"),
    player_diff_pos_opp = as.integer(as.character(no_field_players_pos_10s)) -
                          as.integer(as.character(no_field_players_opp)),
    # suspension-only man-up/man-down difference (keeper pull removed)
    susp_diff = player_diff_pos_opp - gk_pulled
  ) |>
  filter(
    no_field_players_pos_10s %in% c("5", "6", "7"),
    no_field_players_opp     %in% c("5", "6", "7")
  )

make_actual <- function(data) {
  ifelse(data$final == "1", "win",
         ifelse(data$final == "-1", "loss", "draw"))
}

# =============================================================================
# 2. Model spec — the win-probability GAM (no goalkeeper decision)
# =============================================================================
momentum_covariates <- c("mom_conversion_diff_pos_opp", "mom_shot_pct_diff_pos_opp",
                         "mom_turnover_diff_pos_opp",   "mom_penalty_diff_pos_opp")
responses  <- c("final", "final_int", "final_ord", "final_rf")
covariates <- c("bet_win_prob", "gls_diff_pos_opp", "time", "home",
                "susp_diff", "HZ", momentum_covariates)

formula_gam <- final_int ~
  te(time, gls_diff_pos_opp, bs = c("tp", "tp"), k = c(10, 10)) +
  te(time, bet_win_prob,           bs = c("tp", "tp"), k = c(10, 10)) +
  s(matchid, bs = "re") +
  mom_conversion_diff_pos_opp +
  mom_shot_pct_diff_pos_opp +
  mom_turnover_diff_pos_opp +
  mom_penalty_diff_pos_opp +
  home + susp_diff + HZ

tune_formula <- formula_gam
vars_model   <- c(responses, covariates, "matchid")

fit_fn     <- function(d)       fit_model("gam", d, tune_formula)
predict_fn <- function(fit, nd) predict_model(fit, nd, "gam")

# Display name for plot titles and console output. The lowercase "gam" above is
# the dispatch key of fit_model()/predict_model() and must stay as it is; this
# is what the reader sees, and it matches the naming used in 01.
MODEL_LABEL <- "GAM (ocat)"

cat(sprintf("Tuning blend for model: %s | ramp half-width: %d s (%d min), %d s for late break points\n\n",
            MODEL_LABEL, HALF_WIDTH, HALF_WIDTH %/% 60, LATE_HALF_WIDTH))

# =============================================================================
# 3. K-segment blend machinery
# =============================================================================
# ramp_up(), seg_bounds() and segment_weights() come from the shared helper
# functions/blend_functions.R (sourced above); this script drives them over a
# grid of break points, so we pass its own HALF_WIDTH rather than the deployed
# BLEND_HALF_WIDTH default.

# =============================================================================
# 4. Candidate blend configurations (K = 1, 2, 3)
# =============================================================================
# Break points on a 5-minute grid in the second half, with ramps that must not
# overlap and must complete before the final whistle (blend_is_valid).
#
# WHY THE GRID REACHES INTO THE LAST MINUTES. With a 5-minute ramp the latest
# usable break point is 55, whose ramp completes exactly at 60:00; the standard
# grid therefore cannot give the closing seconds a segment of their own. That is
# a problem now that the data reach the final whistle: within about ten seconds
# of it the result is settled---level scorelines end drawn 94% of the time, a
# one-goal deficit is a loss essentially always---while a tensor smooth spread
# over 45 to 60 minutes has roughly ninety seconds of resolution and cannot bend
# that sharply at the edge of its own domain. The selection criterion could not
# see it either, since a 5-minute ECE bin buries the last seconds among the
# preceding minutes.
#
# Late candidates therefore come with LATE_HALF_WIDTH rather than HALF_WIDTH, so
# their segment reaches full weight before the whistle: a break at 58 with a
# 1-minute ramp is fitted from 57:00 (~7,900 rows per training set) and owns the
# match outright from 59:00. Whether any of this earns its place is left to the
# grid; the point is that the grid can now express it.
build_configs <- function() {
  # Break points at 57 or 58 always carry LATE_HALF_WIDTH and no others do, so
  # the centres alone identify a configuration and the label stays short.
  lab <- function(bp) {
    if (length(bp) == 0L) return("no blend")
    sprintf("%d-seg @ %s", length(bp) + 1L, paste(bp %/% 60, collapse = "/"))
  }
  add <- function(cfgs, bp, hw) {
    if (!blend_is_valid(bp, hw)) return(cfgs)
    cfgs[[length(cfgs) + 1L]] <- list(bp = bp, hw = .recycle_hw(bp, hw),
                                      K = length(bp) + 1L, label = lab(bp))
    cfgs
  }

  centers <- c(35, 40, 45, 50, 55) * 60
  late    <- c(57, 58) * 60

  cfgs <- list(list(bp = numeric(0), hw = numeric(0), K = 1L, label = "no blend"))
  for (c1 in c(40, 45, 50, 55) * 60) cfgs <- add(cfgs, c1, HALF_WIDTH)
  for (c1 in late)                   cfgs <- add(cfgs, c1, LATE_HALF_WIDTH)

  for (c1 in centers) for (c2 in centers)
    cfgs <- add(cfgs, c(c1, c2), HALF_WIDTH)
  for (c1 in centers) for (c2 in late)
    cfgs <- add(cfgs, c(c1, c2), c(HALF_WIDTH, LATE_HALF_WIDTH))

  for (c1 in centers) for (c2 in centers) for (c3 in late)
    cfgs <- add(cfgs, c(c1, c2, c3), c(HALF_WIDTH, HALF_WIDTH, LATE_HALF_WIDTH))

  cfgs
}
cfgs  <- build_configs()
n_cfg <- length(cfgs)
cat(sprintf("Evaluating %d blend configurations by leave-one-season-out CV.\n\n", n_cfg))

# =============================================================================
# 5. Full LOSO-CV over all configurations (with a per-fold fit cache)
# =============================================================================
cv_seasons <- sort(unique(daten$season))
cat("LOSO-CV seasons:", paste(cv_seasons, collapse = ", "), "\n\n")

oof_pred <- replicate(n_cfg, list(), simplify = FALSE)  # per config: fold preds
oof_act  <- replicate(n_cfg, list(), simplify = FALSE)

for (f in seq_along(cv_seasons)) {
  val_season <- cv_seasons[f]
  cat(sprintf("Fold %d/%d (val: %s)\n", f, length(cv_seasons), val_season))

  train_data <- daten |> filter(season != val_season) |>
    dplyr::select(all_of(vars_model))
  val_data   <- daten |> filter(season == val_season) |>
    dplyr::select(all_of(vars_model))
  actuals    <- make_actual(val_data)

  # Cache one GAM fit per distinct time window; many configs share windows
  # (e.g. the full-data fit, or the same late window), so we never refit one.
  fit_cache <- list()
  get_fit <- function(lo, hi) {
    key <- paste0(lo, "_", hi)
    if (is.null(fit_cache[[key]])) {
      seg <- train_data |> filter(time >= lo, time <= hi) |>
        realign_factor_levels(train_data)
      lo_lab <- if (is.infinite(lo)) "start" else as.character(lo %/% 60)
      hi_lab <- if (is.infinite(hi)) "end"   else as.character(hi %/% 60)
      cat(sprintf("  fitting window [%s, %s] min (%d rows)\n",
                  lo_lab, hi_lab, nrow(seg)))
      fit_cache[[key]] <<- fit_fn(seg)
    }
    fit_cache[[key]]
  }

  for (ci in seq_len(n_cfg)) {
    bp <- cfgs[[ci]]$bp
    hw <- cfgs[[ci]]$hw
    sb <- seg_bounds(bp, hw)
    W  <- segment_weights(val_data$time, bp, hw)
    acc <- matrix(0, nrow = nrow(val_data), ncol = 3)
    for (k in seq_len(sb$K)) {
      p_k <- as.matrix(predict_fn(get_fit(sb$lo[k], sb$hi[k]), val_data))
      acc <- acc + W[, k] * p_k
    }
    colnames(acc) <- c("win", "draw", "loss")
    oof_pred[[ci]][[f]] <- acc
    oof_act[[ci]][[f]]  <- actuals
  }
}
cat("\n")

# =============================================================================
# 6. Pool out-of-fold predictions and score each configuration
# =============================================================================
# Game time per pooled row, in the same fold/row order as every config's pooled
# predictions (each fold contributes all of its season's rows in order), so it
# aligns with oof_pred for every configuration.
pooled_time <- unlist(lapply(cv_seasons, function(s) daten$time[daten$season == s]))

# BIN_WIDTH, MIN_BIN_N, COMPET_THRESH, is_competitive() and ece_by_time_bin()
# now live in functions/evaluation_functions.R (sourced above). They are shared
# with 01_model_comparison.R, which reports the same worst-phase figure for the
# tree ensembles: the manuscript places those numbers next to these, so a second
# local copy that could drift would make the comparison meaningless.

# For each configuration: overall RPS/ECE, ECE on competitive states, and the
# worst-phase (max-over-bins) ECE computed both over ALL states (confounded, for
# contrast) and over COMPETITIVE states (the de-confounded selection criterion).
blend_summary <- lapply(seq_len(n_cfg), function(ci) {
  P  <- as.data.frame(do.call(rbind, oof_pred[[ci]]))
  A  <- unlist(oof_act[[ci]])
  ok <- complete.cases(P)
  comp <- ok & is_competitive(P)
  data.frame(
    config       = cfgs[[ci]]$label,
    n_seg        = cfgs[[ci]]$K,
    rps          = mean(compute_rps(P[ok, ], A[ok])),
    ece          = compute_calibration_error(P[ok, ], A[ok])$e_overall,
    ece_comp     = compute_calibration_error(P[comp, ], A[comp])$e_overall,
    ece_max      = max(ece_by_time_bin(P, A, pooled_time, ok)$ece,   na.rm = TRUE),
    ece_max_comp = max(ece_by_time_bin(P, A, pooled_time, comp)$ece, na.rm = TRUE),
    brier        = mean(compute_brier_score(P[ok, ], A[ok])),
    logloss      = log_loss_manual(P[ok, ], A[ok]),
    stringsAsFactors = FALSE
  )
}) |>
  bind_rows()

# Selection rule. RPS is the primary criterion in principle, but it only decides
# anything if the configurations actually differ on it. Here they do not: the
# pooled RPS spans a few 1e-4 across the whole grid, far below cross-validation
# noise. We therefore rank on the worst-phase competitive ECE instead---the
# de-confounded max-norm calibration criterion.
#
# That is a claim about THESE results, not a standing rule, so the span is
# checked rather than assumed. If a future run produced configurations that did
# differ materially in accuracy, ranking them on calibration alone would no
# longer be justified, and the warning below says so instead of silently
# selecting on the wrong criterion.
RPS_FLAT_TOL <- 1e-3          # span above which RPS would be considered informative
rps_span <- diff(range(blend_summary$rps))
if (rps_span > RPS_FLAT_TOL) {
  warning(sprintf(paste0(
    "pooled RPS spans %.5f across the grid, above the %.0e treated as negligible: ",
    "the configurations differ on accuracy, so ranking them by calibration alone ",
    "is not justified. Inspect the table before trusting the selection."),
    rps_span, RPS_FLAT_TOL))
}
# Ranked on the worst-phase competitive ECE, then on PARSIMONY, and only then on
# RPS. The middle term matters because ties on the criterion are common and RPS
# cannot break them honestly: it spans a few 1e-4 across the whole grid, below
# cross-validation noise, as this script reports a few lines below. Letting it
# break ties lets noise choose the architecture---and it did: three
# configurations reached the best competitive ECE and RPS picked the one with
# four segments over an equally calibrated one with three. Preferring fewer
# segments deploys the simpler model whenever the criterion cannot separate them.
blend_summary <- blend_summary |> arrange(ece_max_comp, n_seg, rps)

cat("=== Blend configurations, pooled LOSO-CV ===\n")
cat(sprintf("   pooled RPS span across the grid: %.5f (treated as negligible); ranked by max ECE across %d-min bins on competitive states (top prob < %.2f)\n",
            rps_span, BIN_WIDTH %/% 60, COMPET_THRESH))
print(blend_summary, row.names = FALSE, digits = 5)

# ---- Auto-save: blend-tuning table for the manuscript appendix ---------------
# Writes paper/generated/blend_tuning.tex (\input by the manuscript). blend_summary
# is already sorted by competitive worst-phase ECE, then by segment count, so
# row 1 is the selected configuration; its label and competitive worst-phase ECE
# are bolded via raw LaTeX passthrough.
bt_tex <- data.frame(
  Configuration        = .latex_escape(blend_summary$config),
  `Seg.`               = as.character(blend_summary$n_seg),
  RPS                  = formatC(blend_summary$rps,          format = "f", digits = 5),
  ECE                  = formatC(blend_summary$ece,          format = "f", digits = 4),
  `max ECE (all)`      = formatC(blend_summary$ece_max,      format = "f", digits = 4),
  `max ECE (comp.)`    = formatC(blend_summary$ece_max_comp, format = "f", digits = 4),
  check.names = FALSE, stringsAsFactors = FALSE
)
bt_tex$Configuration[1]     <- paste0("\\textbf{", bt_tex$Configuration[1], "}")
bt_tex$`max ECE (comp.)`[1] <- paste0("\\textbf{", bt_tex$`max ECE (comp.)`[1], "}")
save_latex_table(
  bt_tex, "blend_tuning",
  caption = paste0("Blend-segment tuning: pooled leave-one-season-out cross-",
                   "validation over the configuration grid. RPS and ECE are ",
                   "overall, on the pooled out-of-fold predictions; ``max ECE'' ",
                   "is the worst five-minute game-time bin, over all states and ",
                   "---the selection criterion---over competitive states only ",
                   "(no predicted outcome probability reaching $0.9$). Because ",
                   "pooled RPS spans only a few $10^{-4}$ across the grid (below ",
                   "cross-validation noise), configurations within that band are ",
                   "treated as tied on accuracy and ranked by the competitive ",
                   "worst-phase ECE, ties on which are broken in favor of ",
                   "fewer segments rather than by an RPS difference that lies ",
                   "below cross-validation noise; the selected configuration ",
                   "is in bold. ",
                   "Break points up to $55$ minutes use a five-minute ramp ",
                   "half-width; those at $57$ or $58$ use a one-minute one, ",
                   "since a five-minute ramp centered that late would not ",
                   "complete before the final whistle and its segment would ",
                   "never reach full weight."),
  label = "tab:blend_tuning",
  align = "lccccc",
  raw_cols = c("Configuration", "max ECE (comp.)"))

# Headline blend macros: worst-phase competitive ECE, no-blend vs. selected.
noblend_row <- blend_summary[blend_summary$n_seg == 1L, ][1, ]
save_scalar("blendWorstPhaseComp",   formatC(blend_summary$ece_max_comp[1], format = "f", digits = 3))
save_scalar("noblendWorstPhaseComp", formatC(noblend_row$ece_max_comp,      format = "f", digits = 3))
save_scalar("blendSelectedRPS",      formatC(blend_summary$rps[1],           format = "f", digits = 5))
save_scalar("blendSelectedConfig",   .latex_escape(blend_summary$config[1]))

best <- blend_summary[1, ]
best_cfg <- cfgs[[ which(vapply(cfgs, function(c) c$label, "") == best$config) ]]
cat(sprintf(paste0("\nBest configuration: %s\n",
                   "  overall RPS = %.5f | overall ECE = %.5f | competitive ECE = %.5f\n",
                   "  worst-phase ECE: all states = %.5f, competitive = %.5f\n"),
            best$config, best$rps, best$ece, best$ece_comp, best$ece_max, best$ece_max_comp))

# Per-bin ECE profile of the winner: all states vs competitive states, with
# counts, so the decided-game effect (all-states ECE falling late while the
# competitive count thins out) is plainly visible.
best_i    <- which(vapply(cfgs, function(c) c$label, "") == best$config)
best_P    <- as.data.frame(do.call(rbind, oof_pred[[best_i]]))
best_A    <- unlist(oof_act[[best_i]])
best_ok   <- complete.cases(best_P)
prof_all  <- ece_by_time_bin(best_P, best_A, pooled_time, best_ok)
prof_comp <- ece_by_time_bin(best_P, best_A, pooled_time, best_ok & is_competitive(best_P))
fmt_ece <- function(x) ifelse(is.na(x), "  NA  ", sprintf("%.4f", x))
cat("  per-bin ECE (min : all-states ECE [n] | competitive ECE [n]):\n")
for (i in seq_len(nrow(prof_all))) {
  b <- prof_all$bin[i]; cj <- match(b, prof_comp$bin)
  cat(sprintf("    %2d-%2d : %s [%6d] | %s [%6d]\n",
              b * (BIN_WIDTH %/% 60), (b + 1) * (BIN_WIDTH %/% 60),
              fmt_ece(prof_all$ece[i]), prof_all$n[i],
              fmt_ece(if (is.na(cj)) NA_real_ else prof_comp$ece[cj]),
              if (is.na(cj)) 0L else prof_comp$n[cj]))
}
if (best_cfg$K == 1L) {
  cat("  -> the no-blend GAM wins; blending does not improve OOS prediction here.\n")
} else {
  cat(sprintf("  -> %d segments, break point(s) at %s min, ramp half-width(s) %s min.\n",
              best_cfg$K, paste(best_cfg$bp %/% 60, collapse = ", "),
              paste(best_cfg$hw / 60, collapse = ", ")))
}

# =============================================================================
# 7. Plots: RPS by configuration, and calibration-over-time of the winner
# =============================================================================
blend_summary <- blend_summary |>
  mutate(config = factor(config, levels = rev(config)))
p_blend_rps <- ggplot(blend_summary, aes(x = config, y = rps, fill = factor(n_seg))) +
    geom_col() +
    coord_flip(ylim = c(min(blend_summary$rps) - 0.0005,
                        max(blend_summary$rps) + 0.0005)) +
    geom_text(aes(label = sprintf("%.5f", rps)), hjust = -0.1, size = 3) +
    scale_fill_brewer(palette = "Set2", name = "segments") +
    labs(title = sprintf("Blend design for %s: pooled LOSO-CV RPS", MODEL_LABEL),
         subtitle = "Lower is better; ties broken on ECE",
         x = NULL, y = "RPS") +
    theme_minimal()
print(p_blend_rps)
save_plot(p_blend_rps, "blend_rps_by_config", height = 4.4)

best_ci <- which(vapply(cfgs, function(c) c$label, "") == as.character(best$config))
best_P  <- as.data.frame(do.call(rbind, oof_pred[[best_ci]]))
best_A  <- unlist(oof_act[[best_ci]])
best_T  <- unlist(lapply(cv_seasons, function(s) daten$time[daten$season == s]))
okb     <- complete.cases(best_P)
p_blend_calib <- plot_calibration_over_time(
  predictions = best_P[okb, ],
  actuals     = best_A[okb],
  time_var    = best_T[okb] / 60,
  model_name  = sprintf("%s, %s", MODEL_LABEL, best$config)
)
print(p_blend_calib)
save_plot(p_blend_calib, "blend_calibration_over_time", height = 4.6)

# =============================================================================
# 7b. Blend-architecture illustration (manuscript Figure 1)
# =============================================================================
# Built from the SELECTED configuration and the shared segment_weights() helper,
# so the figure can never drift from the blend the paper actually deploys---the
# earlier hand-drawn two-segment version did not match it. Two panels:
#   (a) the segment weights over game time: a partition of unity, each segment
#       handing over to the next across a cosine ramp of its own half-width;
#   (b) an illustrative trajectory showing how the per-segment fits combine into
#       one smooth curve. Panel (b) uses synthetic curves---it illustrates the
#       architecture, not an estimate.
t_grid <- seq(0, 3600, by = 5)
W_ill  <- segment_weights(t_grid, best_cfg$bp, best_cfg$hw)
K_ill  <- ncol(W_ill)

seg_label <- if (K_ill == 1L) "single fit" else
  c("early segment",
    # numbered only when there is more than one middle segment
    if (K_ill == 3L) "middle segment" else
      if (K_ill > 3L) paste0("middle segment ", seq_len(K_ill - 2L)) else NULL,
    "late segment")

w_long <- data.frame(
  minute  = rep(t_grid / 60, times = K_ill),
  weight  = as.vector(W_ill),
  segment = factor(rep(seg_label, each = length(t_grid)), levels = seg_label)
)

seg_cols <- setNames(c("#0077BB", "#EE7733", "#CC3311")[seq_len(K_ill)], seg_label)

p_w <- ggplot(w_long, aes(minute, weight, colour = segment, linetype = segment)) +
  geom_vline(xintercept = best_cfg$bp / 60, colour = "grey70", linewidth = 0.3) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = seg_cols, name = NULL) +
  scale_linetype_discrete(name = NULL) +
  scale_x_continuous(breaks = seq(0, 60, 10), limits = c(0, 60)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1)) +
  labs(subtitle = sprintf(
         "(a) Segment weights: %d segments, break points at %s min, ramp half-width %s min",
         K_ill, paste(best_cfg$bp %/% 60, collapse = " and "),
         paste(best_cfg$hw / 60, collapse = "/")),
       x = NULL, y = expression(w[k](t))) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "top", panel.grid.minor = element_blank())

# Synthetic per-segment win-probability curves: all segments describe the same
# match, but each is fit on its own window, so they disagree where they are
# extrapolating. The blend follows whichever segment currently owns the clock.
set.seed(1)
base_curve  <- 0.5 + 0.28 * sin(2 * pi * t_grid / 5200) - 0.12 * (t_grid / 3600)
seg_curves  <- sapply(seq_len(K_ill), function(k)
  base_curve + (k - (K_ill + 1) / 2) * 0.10 * (t_grid / 3600)^2)
blend_curve <- rowSums(W_ill * seg_curves)

traj <- rbind(
  data.frame(minute = rep(t_grid / 60, times = K_ill),
             p = as.vector(seg_curves),
             series = factor(rep(seg_label, each = length(t_grid)), levels = seg_label),
             kind = "segment fit"),
  data.frame(minute = t_grid / 60, p = blend_curve,
             series = factor("blended", levels = "blended"), kind = "blended")
)

p_t <- ggplot() +
  geom_vline(xintercept = best_cfg$bp / 60, colour = "grey70", linewidth = 0.3) +
  geom_line(data = subset(traj, kind == "segment fit"),
            aes(minute, p, colour = series), linewidth = 0.5, alpha = 0.55,
            linetype = "22") +
  geom_line(data = subset(traj, kind == "blended"),
            aes(minute, p), colour = "#222222", linewidth = 0.9) +
  scale_colour_manual(values = seg_cols, guide = "none") +
  scale_x_continuous(breaks = seq(0, 60, 10), limits = c(0, 60)) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  labs(subtitle = "(b) Illustrative trajectory: dashed = per-segment fits, solid = blended forecast",
       x = "game time [min]", y = "P(win)") +
  theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank())

p_blend_ill <- if (requireNamespace("patchwork", quietly = TRUE)) {
  patchwork::wrap_plots(p_w, p_t, ncol = 1, heights = c(1, 1))
} else {
  print(p_w); p_t
}
print(p_blend_ill)
save_plot(p_blend_ill, "blend_illustration", width = 6.2, height = 4.4)

# =============================================================================
# 8. Hand-off note for downstream scripts (03 / 04)
# =============================================================================
cat(sprintf(
"\n# ============================================================
# Blend chosen for the %s (reuse in 03 / 04):
#   n_segments  = %d
#   break_points = c(%s)   # seconds  (%s min)
#   half_width   = c(%s)   # seconds  (%s min)
# ============================================================\n",
  MODEL_LABEL, best_cfg$K,
  paste(best_cfg$bp, collapse = ", "),
  ifelse(best_cfg$K == 1L, "none", paste(best_cfg$bp %/% 60, collapse = ", ")),
  paste(best_cfg$hw, collapse = ", "),
  ifelse(best_cfg$K == 1L, "none", paste(best_cfg$hw / 60, collapse = ", "))))
