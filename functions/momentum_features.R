library(data.table)

# Momentum features ------------------------------------------------------------
# Handball analog of basketball's "Four Factors" (Dean Oliver), computed as
# rolling stats over each team's recent possessions and used as a proxy for
# current in-game momentum/form, independent of the cumulative score:
#
#   conversion  = goals / possessions           (overall scoring efficiency;
#                                                 stands in for Oliver's ORB%,
#                                                 for which this play-by-play
#                                                 has no rebound-equivalent)
#   shot_pct    = goals / shot attempts          (pure finishing quality,
#                 (attempts = goal | missed_shot,  ~ eFG%)
#                  turnovers excluded)
#   turnover    = turnovers / possessions        (ball security, ~ TOV%)
#   penalty     = 7m-attempts / possessions      (rate of earning high-value
#                                                  chances, ~ FT rate; 7m
#                                                  shots convert at ~79% vs.
#                                                  a ~54% baseline in this data)
#
# All four are computed for both the possessing team ("_pos") and the
# opponent ("_opp") over a trailing window of `window` attacks.

#' Add rolling momentum (four-factors-style) features to possession-level data
#'
#' Both sides are read at the instant the row records, which is the state AFTER
#' the possession (see pre_process_data): mom_*_pos covers the possessing team's
#' last `window` attacks INCLUDING the one just played, mom_*_opp the opponent's
#' last `window` completed attacks, which is everything of theirs up to this
#' point in the match. Nothing here looks ahead, so no lag is applied on either
#' side; a lag would push the features back to the start of the possession and
#' reintroduce the mixture of two instants that the post-state convention exists
#' to remove.
#'
#' @param data   Possession-level data frame with matchid, pos_no (a single
#'               ascending sequence per match shared by both teams), team,
#'               opp, goal, missed_shot, turnover, penalty
#' @param window Number of trailing own attacks used for the rolling stats
#'               (default 5; fewer are used at the start of a team's
#'               sequence in a match)
#' @return `data` with 8 raw (mom_<metric>_pos/opp) and 4 diff
#'         (mom_<metric>_diff_pos_opp = pos - opp) columns added. Rows where
#'         the rolling window is undefined (the opponent has not attacked yet,
#'         or — for shot_pct — no shot was taken in the window) are imputed
#'         with the dataset-wide mean of that column (under 1% of rows).
add_momentum_features <- function(data, window = 5) {
  dt <- as.data.table(data)
  dt[, .orig_row_id := .I]
  setorder(dt, matchid, pos_no)

  dt[, .attempt := as.integer(goal == 1 | missed_shot == 1)]

  # Trailing partial-window sum: last `w` values ending at each position,
  # fewer at the start of the group. Vectorised via cumsum (no explicit loop).
  roll_sum_partial <- function(x, w) {
    n  <- length(x)
    cs <- cumsum(c(0, x))
    idx <- seq_len(n)
    lo  <- pmax(0L, idx - w)
    cs[idx + 1L] - cs[lo + 1L]
  }
  roll_n_partial <- function(n_total, w) {
    idx <- seq_len(n_total)
    idx - pmax(0L, idx - w)
  }

  # Rolling stats over each team's own possessions, ending at and including the
  # possession the row records---the post-attack state, matching every other
  # column.
  dt[, `:=`(
    .n_win        = roll_n_partial(.N, window),
    .goals_win    = roll_sum_partial(goal,       window),
    .attempts_win = roll_sum_partial(.attempt,   window),
    .tos_win      = roll_sum_partial(turnover,   window),
    .pens_win     = roll_sum_partial(penalty,    window)
  ), by = .(matchid, team)]

  dt[, `:=`(
    .post_conversion = .goals_win / .n_win,
    .post_shot_pct   = ifelse(.attempts_win > 0, .goals_win / .attempts_win, NA_real_),
    .post_turnover   = .tos_win / .n_win,
    .post_penalty    = .pens_win / .n_win
  )]

  # Own momentum: the window ending with the attack just completed.
  dt[, `:=`(
    mom_conversion_pos = .post_conversion,
    mom_shot_pct_pos    = .post_shot_pct,
    mom_turnover_pos    = .post_turnover,
    mom_penalty_pos     = .post_penalty
  )]

  # Opponent momentum: the opponent's most recent completed attack as of this
  # possession, found via a rolling (last-observation-carried-forward) join.
  # pos_no is a single sequence shared by both teams, so the opponent never
  # holds this row's pos_no and the roll lands on their previous attack.
  team_form <- dt[, .(matchid, team, pos_no,
                       .post_conversion, .post_shot_pct, .post_turnover, .post_penalty)]
  setkey(team_form, matchid, team, pos_no)

  dt[, .join_pos := pos_no]
  opp_vals <- team_form[dt, on = .(matchid, team = opp, pos_no = .join_pos), roll = TRUE,
                         .(mom_conversion_opp = .post_conversion,
                           mom_shot_pct_opp    = .post_shot_pct,
                           mom_turnover_opp    = .post_turnover,
                           mom_penalty_opp     = .post_penalty)]
  dt[, `:=`(mom_conversion_opp = opp_vals$mom_conversion_opp,
            mom_shot_pct_opp    = opp_vals$mom_shot_pct_opp,
            mom_turnover_opp    = opp_vals$mom_turnover_opp,
            mom_penalty_opp     = opp_vals$mom_penalty_opp)]

  # Impute with the dataset-wide mean wherever the rolling window is
  # undefined: before the opponent's first attack, and for shot_pct
  # specifically whenever the window's attacks were all turnovers (no shot
  # taken to compute a percentage from). Both are rare (well under 1% of rows)
  # but not exactly zero.
  mom_cols <- c("mom_conversion_pos", "mom_conversion_opp",
                "mom_shot_pct_pos",   "mom_shot_pct_opp",
                "mom_turnover_pos",   "mom_turnover_opp",
                "mom_penalty_pos",    "mom_penalty_opp")
  for (col in mom_cols) {
    col_mean <- mean(dt[[col]], na.rm = TRUE)
    dt[is.na(get(col)), (col) := col_mean]
  }

  # Momentum swing: own minus opponent (positive = better recent form than
  # the opponent right now)
  dt[, `:=`(
    mom_conversion_diff_pos_opp = mom_conversion_pos - mom_conversion_opp,
    mom_shot_pct_diff_pos_opp   = mom_shot_pct_pos   - mom_shot_pct_opp,
    mom_turnover_diff_pos_opp   = mom_turnover_pos   - mom_turnover_opp,
    mom_penalty_diff_pos_opp    = mom_penalty_pos    - mom_penalty_opp
  )]

  # Drop internal helper columns, restore the original row order
  dt[, c(".attempt", ".n_win", ".goals_win", ".attempts_win", ".tos_win", ".pens_win",
         ".post_conversion", ".post_shot_pct", ".post_turnover", ".post_penalty",
         ".join_pos") := NULL]
  setorder(dt, .orig_row_id)
  dt[, .orig_row_id := NULL]

  as.data.frame(dt)
}
