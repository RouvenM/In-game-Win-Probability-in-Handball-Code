# 07_rebuild_comparison_table.R ------------------------------------------------
# Rebuilds generated/model_comparison.tex from a finished run of
# 01_model_comparison.R, WITHOUT refitting anything.
#
# Why this exists: the expensive part of 01 is the leave-one-season-out fitting
# that fills cv_preds. Everything the table reports is a summary of those
# predictions, so if the table definition changes after a run has started, the
# run is still worth finishing -- the predictions are what took the hours, and
# they are all this script needs.
#
# HOW TO USE IT. Run 01 in an interactive session with source(), not Rscript, so
# that the objects survive the script. When it has finished, in the SAME
# session:
#
#     source("07_rebuild_comparison_table.R")
#
# It reads cv_preds, cv_actuals and cv_time out of the session, recomputes every
# reported figure and overwrites the table. Nothing is fitted, so it takes
# seconds.
#
# If the session is gone, this cannot help: 01 does not persist its predictions,
# so there is nothing to rebuild from and 01 has to run again.
# -----------------------------------------------------------------------------

library(here)
library(dplyr)
source(here("functions", "evaluation_functions.R"))
source(here("functions", "output_helpers.R"))
source(here("functions", "comparison_table.R"))

# The predictions come from the session if it is still open, and from the cache
# 01 writes otherwise -- which is the case that matters, since a session rarely
# outlives the decision to change a table.
needed <- c("cv_preds", "cv_actuals", "cv_time")
absent <- needed[!vapply(needed, exists, logical(1), envir = globalenv())]
if (length(absent)) {
  cache_path <- here("cache", "cv_predictions.rds")
  if (!file.exists(cache_path))
    stop("07_rebuild_comparison_table.R found neither ",
         paste(absent, collapse = ", "), " in the session nor a cache at ",
         cache_path, ". A run of 01_model_comparison.R from before the cache ",
         "was added leaves nothing to rebuild from, so 01 has to run once more; ",
         "from then on this script works whether or not the session survives.")
  cached <- readRDS(cache_path)
  cat(sprintf("Reading predictions from %s (written %s).\n",
              cache_path, format(cached$written, "%Y-%m-%d %H:%M")))
  for (nm in c("cv_preds", "cv_actuals", "cv_time", "cv_side"))
    if (!is.null(cached[[nm]])) assign(nm, cached[[nm]], envir = globalenv())
}

ids <- intersect(names(COMPARISON_DISPLAY), names(cv_preds))
if (!length(ids))
  stop("none of the models in COMPARISON_DISPLAY are present in cv_preds; ",
       "cv_preds holds: ", paste(names(cv_preds), collapse = ", "))
left_out <- setdiff(names(COMPARISON_DISPLAY), ids)
if (length(left_out))
  warning("not in this run, so left out of the table: ",
          paste(left_out, collapse = ", "))

cat(sprintf("Rebuilding the comparison table from %d models in the session.\n",
            length(ids)))
mc <- bind_rows(lapply(ids, function(m_id) {
  cat(sprintf("  %-30s", COMPARISON_DISPLAY[[m_id]]))
  m <- comparison_metrics(cv_preds[[m_id]], cv_actuals[[m_id]], cv_time)
  cat(sprintf("RPS %.4f (1st %.4f, 2nd %.4f)\n",
              m$rps_mean, m$rps_h1, m$rps_h2))
  data.frame(model = m_id, Model = unname(COMPARISON_DISPLAY[[m_id]]),
             RPS = m$rps_mean, `RPS 1st` = m$rps_h1, `RPS 2nd` = m$rps_h2,
             Brier = m$brier_mean, `Brier 1st` = m$brier_h1,
             `Brier 2nd` = m$brier_h2,
             ECE = m$ece_mean, `ECE 1st` = m$ece_h1, `ECE 2nd` = m$ece_h2,
             check.names = FALSE, stringsAsFactors = FALSE)
})) |>
  # Same sort key as 01: pooled RPS to four decimals, ECE breaking near-ties.
  arrange(round(RPS, 4), ECE)

write_comparison_table(mc)

# The figure's data comes from the same predictions, so rebuild it here too.
# cv_side was added after some runs; where it is absent it can be reconstructed
# from daten and cv_seasons, which a finished interactive session still holds.
if (!exists("cv_side", envir = globalenv()) &&
    all(vapply(c("daten", "cv_seasons"), exists, logical(1), envir = globalenv()))) {
  cv_side <- unlist(lapply(cv_seasons, function(s)
    paste0(daten$matchid[daten$season == s], "/", daten$home[daten$season == s])))
  cat("\nReconstructed cv_side from daten/cv_seasons.\n")
}
if (exists("cv_side", envir = globalenv())) {
  rps_by_time <- bind_rows(lapply(ids, function(m_id)
    cbind(data.frame(model = m_id, model_name = unname(COMPARISON_DISPLAY[[m_id]]),
                     stringsAsFactors = FALSE),
          rps_by_time_curve(cv_preds[[m_id]], cv_actuals[[m_id]], cv_time, cv_side))))
  write.csv(rps_by_time, file.path(OUTPUT_DIR, "rps_by_time.csv"), row.names = FALSE)
  cat(sprintf("Rewrote %s -- run 06_rps_by_time.R to redraw the figure.\n",
              file.path(OUTPUT_DIR, "rps_by_time.csv")))
} else {
  warning("cv_side could not be built (daten/cv_seasons are gone), so ",
          "rps_by_time.csv was left alone and its reference lines will not be ",
          "exactly flat.")
}

cat("\nTable rewritten. Row order:\n")
print(as.data.frame(mc[, c("Model", "RPS", "RPS 1st", "RPS 2nd")]),
      row.names = FALSE, digits = 4)
