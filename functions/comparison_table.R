# functions/comparison_table.R -------------------------------------------------
# One place where the model-comparison table is defined: the metrics that go in
# it, and the LaTeX that comes out.
#
# Three callers need to agree on it exactly -- 01_model_comparison.R, which
# produces it from a full run; 05_leader_baseline.R, which splices one row into
# an existing table; and 07_rebuild_comparison_table.R, which rebuilds it from a
# finished session's predictions. They used to carry three copies of the column
# list, the header and the caption, and every change to the table meant editing
# all three in step. They now call the two functions below.
# -----------------------------------------------------------------------------

# The nine reported figures for one model: three metrics, each over the first
# half, the second half and the whole match.
#
# RPS and the Brier score are per-possession scores, so the halves are read off
# the same vectors the overall mean is taken over -- split at the interval
# rather than recomputed. The expected calibration error is a binned diagnostic
# and not a per-row score, so each half has to be recomputed on its own rows;
# the overall figure is therefore NOT the weighted mean of the two halves, and
# is not meant to be.
#
# `times` is game time in seconds, aligned with the rows of `p`.
comparison_metrics <- function(p, a, times) {
  p  <- as.data.frame(p)
  ok <- complete.cases(p)
  pk <- p[ok, ]; ak <- a[ok]
  stopifnot(length(times) == nrow(p))
  h1 <- times[ok] <= 1800
  if (!any(h1) || all(h1))
    stop("comparison_metrics: the rows fall entirely in one half, so the split ",
         "would be empty. Check that `times` is game time in seconds and is ",
         "aligned with the rows of `p`.")
  rps_i   <- compute_rps(pk, ak)
  brier_i <- compute_brier_score(pk, ak)
  ece_of  <- function(keep) compute_calibration_error(pk[keep, ], ak[keep])$e_overall
  data.frame(
    # The selection criterion of the paper: the calibration error of the
    # worst-calibrated five-minute window among competitive states. Reported for
    # every model, not only the two the discussion quotes, because it is the
    # figure the methodological claim rests on and an aggregate ECE on which
    # models cannot be told apart is no substitute for it.
    ece_max    = worst_phase_ece(p, a, times),
    rps_mean   = mean(rps_i),   rps_h1   = mean(rps_i[h1]),   rps_h2   = mean(rps_i[!h1]),
    brier_mean = mean(brier_i), brier_h1 = mean(brier_i[h1]), brier_h2 = mean(brier_i[!h1]),
    ece_mean   = ece_of(rep(TRUE, length(ak))),
    ece_h1     = ece_of(h1),    ece_h2   = ece_of(!h1),
    stringsAsFactors = FALSE
  )
}

# Display names, in no particular order: the rows are sorted by score, not by
# this list. A model id missing from here is left out of the table.
COMPARISON_DISPLAY <- c(
  gam                = "GAM (ocat)",
  baseline_xgb       = "XGBoost (ordinal)",
  baseline_xgb_multi = "XGBoost (multiclass)",
  baseline_rf        = "Random forest (multiclass)",
  baseline_rf_ord    = "Random forest (ordinal)",
  baseline           = "Betting-odds baseline",
  odds_leader        = "Odds-and-scoreboard baseline",
  leader             = "Leading-team baseline"
)

COMPARISON_CAPTION <- paste0(
  "Leave-one-season-out cross-validation, pooled over the five held-out ",
  "seasons, for the first half, the second half and the whole match (lower ",
  "is better). The whole-match ECE is computed on the pooled predictions and ",
  "is therefore not the average of the two halves. All models are single ",
  "fits without time segmentation. Bold marks the best value in each column.")

# Writes generated/model_comparison.tex. `mc` carries a Model column and the
# nine metric columns under the names used below, already in the row order the
# table should print.
#
# The logarithmic score is deliberately absent. Three rows would score infinity
# -- the leading-team rule by design, the two ordinal ensembles through the
# clipping of their cumulative-link decomposition -- and a column that cannot
# rank half of what it contains is not worth a column.
write_comparison_table <- function(mc, caption = COMPARISON_CAPTION) {
  cols <- c("RPS 1st", "Brier 1st", "ECE 1st",
            "RPS 2nd", "Brier 2nd", "ECE 2nd",
            "RPS", "Brier", "ECE")
  missing <- setdiff(c("Model", cols), names(mc))
  if (length(missing))
    stop("write_comparison_table: missing column(s) ",
         paste(missing, collapse = ", "))

  # A cell can be undefined (the worst-phase criterion for a forecaster with no
  # competitive states) or infinite. Neither is a number to print as one, and
  # neither may win a column: which.min would hand the bold to -Inf or to NA.
  fmt3 <- function(x)
    ifelse(is.na(x), "---",
           ifelse(is.finite(x), formatC(x, format = "f", digits = 3), "$\\infty$"))
  bold_min <- function(x) {
    s <- fmt3(x)
    if (any(is.finite(x))) {
      i <- which(is.finite(x))[which.min(x[is.finite(x)])]
      s[i] <- paste0("\\textbf{", s[i], "}")
    }
    s
  }
  out <- data.frame(Model = mc$Model, check.names = FALSE,
                    stringsAsFactors = FALSE)
  for (cn in cols) out[[cn]] <- bold_min(mc[[cn]])

  # Ten columns want a two-deck header: the period spans its three columns, and
  # the deck beneath names the metrics. Passed verbatim (escape_header = FALSE)
  # because the header is LaTeX rather than text.
  header <- paste0(
    " & \\multicolumn{3}{c}{First half} & \\multicolumn{3}{c}{Second half}",
    " & \\multicolumn{3}{c}{Whole match} \\\\\n",
    "\\cline{2-4}\\cline{5-7}\\cline{8-10}\n",
    "Model & RPS & Brier & ECE & RPS & Brier & ECE & RPS & Brier & ECE")

  save_latex_table(
    out, "model_comparison",
    caption = caption,
    label   = "tab:model_comparison",
    align   = "lccccccccc",
    header  = header, escape_header = FALSE,
    # ten columns overrun the text block by 27pt at the default column
    # separation and fit comfortably at 3.2pt, so the body stays at \small
    prelude = "\\setlength{\\tabcolsep}{3.2pt}",
    raw_cols = cols)
}

# The ranked probability score by five-minute game-time bin, for one model.
#
# Two weightings, because they answer different questions and only one of them
# belongs in a figure drawn against game time.
#
#   rps     averages over POSSESSIONS. It is the expected score of a randomly
#           drawn possession in that window, and it is what the pooled table
#           reports, so the two agree by construction.
#
#   rps_eq  averages over MATCH SIDES: each side's mean score within the window
#           first, then those means with equal weight. A match contributes a
#           different number of possessions to each window -- the closing five
#           minutes hold around nine or ten per match against under eight in
#           the opening five -- so a possession-weighted curve moves with the
#           changing mix of matches as well as with the models. The pre-match
#           baseline makes that unmistakable: its forecast is literally
#           constant within a match side, so its curve must be flat, and under
#           possession weighting it is not. Weighting sides equally removes the
#           composition effect and flattens it exactly.
#
# `sides` identifies the match side a row belongs to -- the match, and which of
# the two teams was in possession -- because the forecast is read from the
# possessing team's perspective and possession alternates.
rps_by_time_curve <- function(preds, actuals, times, sides) {
  p  <- as.data.frame(preds)
  ok <- complete.cases(p)
  stopifnot(length(times) == nrow(p), length(sides) == nrow(p))
  d <- data.frame(
    time_bin = pmin(floor(times[ok] / 300) * 5, 55),
    side     = as.character(sides[ok]),
    rps      = compute_rps(p[ok, ], actuals[ok]),
    stringsAsFactors = FALSE)
  per_side <- d |>
    dplyr::group_by(time_bin, side) |>
    dplyr::summarize(side_mean = mean(rps), .groups = "drop")
  dplyr::left_join(
    d |> dplyr::group_by(time_bin) |>
      dplyr::summarize(rps = mean(rps), n = dplyr::n(), .groups = "drop"),
    per_side |> dplyr::group_by(time_bin) |>
      dplyr::summarize(rps_eq = mean(side_mean), n_sides = dplyr::n(),
                       .groups = "drop"),
    by = "time_bin")
}
