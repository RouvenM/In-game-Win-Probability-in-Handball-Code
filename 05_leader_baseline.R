# 05_leader_baseline.R ---------------------------------------------------------
# Runs ONLY the leading-team baseline and folds its row into the comparison
# table that 01_model_comparison.R has already written.
#
# WHY THIS FILE EXISTS. The baseline was added after 01 had been run, and 01
# refits every model on every fold, which is hours of work to obtain one row
# that costs seconds. This script reproduces 01's data preparation and its
# cross-validation exactly -- same seasons, same filters, same metrics -- runs
# the one model, and rewrites paper/generated/model_comparison.tex with the new
# row inserted in its proper place. Nothing else in the table is recomputed:
# the other models' figures are read back out of the file 01 wrote, so they are
# the same numbers, not new ones.
#
# It is idempotent. Running it twice replaces the leader row rather than adding
# a second one, and running 01 in full afterwards overwrites the whole table
# from scratch, including this model, which is then the authoritative version.
# -----------------------------------------------------------------------------

rm(list = ls())
library(here)
library(dplyr)

source(here("functions", "general_functions.R"))
source(here("functions", "pre_processing.R"))
source(here("functions", "model_wrappers.R"))
source(here("functions", "evaluation_functions.R"))
source(here("functions", "momentum_features.R"))
source(here("functions", "output_helpers.R"))
source(here("functions", "comparison_table.R"))

# =============================================================================
# 1. Data preparation -- identical to 01_model_comparison.R
# =============================================================================
daten <- read.csv(here("raw_data", "preprocessed_data.csv"))
daten <- pre_process_data(daten)
daten <- add_momentum_features(daten, window = 5)

daten <- daten |>
  ungroup() |>
  mutate(
    final_int = as.integer(factor(final, levels = c(-1, 0, 1))),
    final_ord = ordered(final, levels = c("-1", "0", "1")),
    matchid   = as.factor(matchid),
    final_rf  = factor(final, levels = c("-1", "0", "1"),
                       labels = c("loss", "draw", "win"))
  ) |>
  filter(
    no_field_players_pos_10s %in% c("5", "6", "7"),
    no_field_players_opp     %in% c("5", "6", "7")
  )

make_actual <- function(data) {
  ifelse(data$final == "1",  "win",
         ifelse(data$final == "-1", "loss", "draw"))
}

cv_seasons <- sort(unique(daten$season))
cat(sprintf("Leading-team baseline over %d LOSO folds: %s\n\n",
            length(cv_seasons), paste(cv_seasons, collapse = ", ")))

# =============================================================================
# 2. The rule
# =============================================================================
# The scoreboard read literally as a forecast: whoever is ahead is named the
# winner, a level score is called a draw. There is no model here and nothing is
# estimated. The rule never sees the training seasons at all -- it sees the
# score at the moment it speaks and nothing else -- so it is the same rule in
# every fold, and the cross-validation below exists only to score it on the
# same possessions the other models were scored on.
lead_cell <- function(x) {
  d <- suppressWarnings(as.numeric(as.character(x)))
  ifelse(is.na(d), "0", ifelse(d > 0, "1", ifelse(d < 0, "-1", "0")))
}

leader_predict <- function(newdata) {
  cell <- lead_cell(newdata$gls_diff_pos_opp)
  data.frame(win  = as.numeric(cell == "1"),
             draw = as.numeric(cell == "0"),
             loss = as.numeric(cell == "-1"))
}

# A forecast that claims certainty has an infinite logarithmic score the first
# time it is wrong, and this one is wrong often. log_loss_manual clips at 1e-15
# before taking the logarithm, which keeps it from returning -Inf but turns the
# infinity into roughly 34.5 per mistaken possession -- a number that says what
# the clipping constant is, not how good the forecast is. We therefore report
# the score that is actually defined here, which is infinite, and say so.
log_loss_honest <- function(p, a) {
  realised <- p[cbind(seq_len(nrow(p)), match(a, colnames(p)))]
  if (any(realised <= 0)) Inf else -mean(log(realised))
}

# =============================================================================
# 3. Cross-validation
# =============================================================================
cv_preds <- list(); cv_actuals <- list(); cv_times <- list()
for (f in seq_along(cv_seasons)) {
  val_season <- cv_seasons[f]
  train_data <- daten |> filter(season != val_season)
  val_data   <- daten |> filter(season == val_season)

  preds <- leader_predict(val_data)
  act   <- make_actual(val_data)

  if (f == 1) {
    cat(sprintf("  prediction frame: %d x %d, columns %s, any NA: %s\n",
                nrow(preds), ncol(preds), paste(names(preds), collapse = "/"),
                anyNA(preds)))
    cat(sprintf("  states called: %s\n",
                paste(sprintf("%s %.1f%%", c("loss", "draw", "win"),
                              100 * colMeans(preds[, c("loss", "draw", "win")])),
                      collapse = " | ")))
    cat(sprintf("  rule is right on %.1f%% of possessions in this fold\n\n",
                100 * mean(preds[cbind(seq_len(nrow(preds)), match(act, names(preds)))] == 1)))
  }
  stopifnot(nrow(preds) == length(act),
            all(c("win", "draw", "loss") %in% names(preds)),
            !anyNA(preds))

  cv_preds[[f]]   <- preds
  cv_actuals[[f]] <- act
  cv_times[[f]]   <- val_data$time
  cat(sprintf("  fold %d / %d (val: %s): %d possessions, RPS %.4f\n",
              f, length(cv_seasons), val_season, nrow(preds),
              mean(compute_rps(preds, act))))
}

p <- bind_rows(cv_preds)
a <- unlist(cv_actuals)
ok <- complete.cases(p)
# The logarithmic score is computed for the console only. The manuscript no
# longer reports it: it is infinite here by construction, and for the two
# ordinal ensembles through their clipping, so the column could rank neither.
rps_i   <- compute_rps(p[ok, ], a[ok])
brier_i <- compute_brier_score(p[ok, ], a[ok])
first_half <- unlist(cv_times)[ok] <= 1800
pk <- p[ok, ]; ak <- a[ok]
ece_half <- function(keep) compute_calibration_error(pk[keep, ], ak[keep])$e_overall
res <- list(
  RPS          = mean(rps_i),
  `RPS 1st`    = mean(rps_i[first_half]),
  `RPS 2nd`    = mean(rps_i[!first_half]),
  Brier        = mean(brier_i),
  `Brier 1st`  = mean(brier_i[first_half]),
  `Brier 2nd`  = mean(brier_i[!first_half]),
  `Log-loss`   = log_loss_honest(p[ok, ], a[ok]),
  ECE          = ece_half(rep(TRUE, length(ak))),
  `ECE 1st`    = ece_half(first_half),
  `ECE 2nd`    = ece_half(!first_half))
cat(sprintf("\nPooled over all folds: RPS %.4f | Brier %.4f | Log-loss %s | ECE %.4f\n",
            res$RPS, res$Brier,
            if (is.finite(res$`Log-loss`)) sprintf("%.4f", res$`Log-loss`) else "infinite",
            res$ECE))

save_scalar("leaderRPS",     formatC(res$RPS,        format = "f", digits = 3))
save_scalar("leaderBrier",   formatC(res$Brier,      format = "f", digits = 3))
save_scalar("leaderECE",     formatC(res$ECE,        format = "f", digits = 3))

# =============================================================================
# 4. Fold the row into the table 01 already wrote
# =============================================================================
tex_path <- here("paper", "generated", "model_comparison.tex")
stopifnot(file.exists(tex_path))
lines <- readLines(tex_path, warn = FALSE)

cap <- sub("^\\\\caption\\{", "", grep("^\\\\caption\\{", lines, value = TRUE)[1])
cap <- sub("\\}$", "", cap)

body <- grep("^[^\\\\].*\\\\\\\\$", lines, value = TRUE)   # data rows end in \\
body <- body[!grepl("^Model &", body)]
parse_row <- function(l) {
  cells <- trimws(strsplit(sub("\\s*\\\\\\\\$", "", l), "&", fixed = TRUE)[[1]])
  cells <- gsub("\\\\textbf\\{|\\}", "", cells)
  cells <- ifelse(grepl("infty", cells), "Inf", cells)
  data.frame(Model = cells[1],
             `RPS 1st`   = as.numeric(cells[2]),
             `Brier 1st` = as.numeric(cells[3]),
             `ECE 1st`   = as.numeric(cells[4]),
             `RPS 2nd`   = as.numeric(cells[5]),
             `Brier 2nd` = as.numeric(cells[6]),
             `ECE 2nd`   = as.numeric(cells[7]),
             RPS         = as.numeric(cells[8]),
             Brier       = as.numeric(cells[9]),
             ECE         = as.numeric(cells[10]),
             check.names = FALSE, stringsAsFactors = FALSE)
}
mc <- bind_rows(lapply(body, parse_row))
mc <- mc[mc$Model != "Leading-team baseline", ]          # idempotent
mc <- bind_rows(mc, data.frame(Model = "Leading-team baseline",
                               RPS = res$RPS,
                               `RPS 1st` = res$`RPS 1st`, `RPS 2nd` = res$`RPS 2nd`,
                               Brier = res$Brier,
                               `Brier 1st` = res$`Brier 1st`,
                               `Brier 2nd` = res$`Brier 2nd`,
                               ECE = res$ECE,
                               `ECE 1st` = res$`ECE 1st`, `ECE 2nd` = res$`ECE 2nd`,
                               check.names = FALSE, stringsAsFactors = FALSE))
mc <- mc |> arrange(round(RPS, 4), ECE)

write_comparison_table(mc, caption = cap)

cat("\nTable rewritten with the leading-team baseline in place:\n")
print(as.data.frame(mc), row.names = FALSE, digits = 4)
