# 04_use_cases.R --------------------------------------------------------------
# Paper workflow – Step 4: Applications of the win-probability model
#
# Three applications built on the deployed blended win-probability GAM (which
# excludes the goalkeeper decision — that is a tactical choice, valued in C).
# Every row of `daten` is the state AFTER the possession it records (see
# pre_process_data); `time` is that possession's terminating stamp and
# `time_prev` the clock it started from, which is the instant a coach decides at
# and therefore what use case C reads its frequencies at:
#   (A) Fan engagement — win probability as a "story stat": reconstruct the
#       most dramatic matches by the size of their win-probability swings and
#       comebacks, and plot example trajectories.
#   (B) Value added (VA) — value each possession/goal by the EXPECTED LEAGUE
#       POINTS it adds for the attacking team, P(win) + 0.5*P(draw); rank the
#       highest-leverage moments and players. Expected points rather than win
#       probability for the same reason as in (C): a draw is worth a point.
#   (C) Goalkeeper-substitution decision — a fourth-down-style decomposition:
#       compare V(pull) vs V(keep) by valuing the states the exchange leads to,
#       where V is EXPECTED LEAGUE POINTS, P(win) + 0.5*P(draw) — a draw is
#       worth a point and is much of the payoff to a late pull. In the closing
#       seconds the exchange runs to the whistle, so those states are settled by
#       their lead and the decision reduces to a break-even on the scoring
#       chance whose downside observed frequencies can bound. Away from the
#       whistle the states have to be valued by the model, and a full treatment
#       needs the companion paper's state-space policy.
#
# Predictions are OUT-OF-FOLD (leave-one-season-out): every match is scored by
# a model that never saw its season during training. This avoids in-sample
# over-fit artefacts in the trajectories — important because the use cases read
# the curves at face value. (Set USE_INSAMPLE <- TRUE for a quick single-fit
# pass during iteration.)
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
source(here("functions", "blend_functions.R"))
source(here("functions", "output_helpers.R"))

USE_INSAMPLE <- FALSE   # FALSE = leave-one-season-out OOF predictions (default)

# =============================================================================
# 1. Data preparation — mirrors 01_model_comparison.R
# =============================================================================
daten <- read.csv(here("raw_data", "preprocessed_data.csv"))
daten <- pre_process_data(daten)
daten <- add_momentum_features(daten, window = 5)

daten <- daten |>
  ungroup() |>
  mutate(
    final_int = as.integer(factor(final, levels = c(-1, 0, 1))),
    matchid   = as.factor(matchid),
    HZ        = ifelse(as.character(HZ) == "30", 1, 0),
    win       = as.integer(as.character(final) == "1"),   # 1 if pos team won
    # Ten-second tolerance, as in 01 (see pre_process_data).
    gk_pulled = as.integer(as.character(gk_pos_10s) == "0"),
    player_diff_pos_opp = as.integer(as.character(no_field_players_pos_10s)) -
                          as.integer(as.character(no_field_players_opp)),
    # suspension-only man-up/man-down difference (keeper pull removed); this is
    # the only numerical-strength term in the win-probability model.
    susp_diff = player_diff_pos_opp - gk_pulled,
    # possession-team perspective flag (1 = the team in possession is the home
    # team). home is built in pre_process_data as team == team_home.
    is_home_pos = as.integer(as.character(home) == "1")
  ) |>
  filter(
    no_field_players_pos_10s %in% c("5", "6", "7"),
    no_field_players_opp     %in% c("5", "6", "7")
  )

# Stable within-match ordering. pos_no is a single ascending sequence per match
# shared by both teams (verified in the momentum-feature work).
daten <- daten |> arrange(matchid, pos_no)

# ---- How the matches themselves ended ----------------------------------------
# Counted at MATCH level, not possession level: the introduction and Section 2
# both quote how often a handball match is drawn, which is the reason the
# response is three-class rather than binary, and it is the kind of number that
# should not be typed into the text by hand. Home and away shares come from the
# same tally so the three always add up.
match_ends <- daten |>
  group_by(matchid) |>
  summarize(home = as.character(dplyr::last(team_home)),
            win  = as.character(dplyr::last(winner)), .groups = "drop") |>
  mutate(outcome = ifelse(win == "draw", "draw",
                   ifelse(win == home, "home", "away")))
n_matches_all <- nrow(match_ends)
n_draw  <- sum(match_ends$outcome == "draw")
cat(sprintf("\nMatch outcomes over %d matches: %d drawn (%.1f%%), home %.1f%%, away %.1f%%\n",
            n_matches_all, n_draw, 100 * mean(match_ends$outcome == "draw"),
            100 * mean(match_ends$outcome == "home"),
            100 * mean(match_ends$outcome == "away")))
save_scalar("nDrawMatches", format(n_draw, big.mark = ","))
save_scalar("pctDraw", formatC(100 * mean(match_ends$outcome == "draw"), format = "f", digits = 1))
save_scalar("pctHomeWin", formatC(100 * mean(match_ends$outcome == "home"), format = "f", digits = 1))
save_scalar("pctAwayWin", formatC(100 * mean(match_ends$outcome == "away"), format = "f", digits = 1))

# =============================================================================
# 2. Selected model — the deployed blended win-probability GAM (as in 01/02)
# =============================================================================
# The K-segment blend machinery and the deployed configuration
# (BLEND_BREAKPOINTS / BLEND_HALF_WIDTH, selected in 02) live in
# functions/blend_functions.R. The model excludes the goalkeeper decision (a
# tactical choice valued in Use Case C); its only numerical-strength term is
# susp_diff, the suspension-only man-up/man-down difference. Clock and score
# difference are the POST-possession ones, matching 01/02.
formula_gam <- final_int ~
  te(time, gls_diff_pos_opp, bs = c("tp", "tp"), k = c(10, 10)) +
  te(time, bet_win_prob,           bs = c("tp", "tp"), k = c(10, 10)) +
  s(matchid, bs = "re") +
  mom_conversion_diff_pos_opp + mom_shot_pct_diff_pos_opp +
  mom_turnover_diff_pos_opp   + mom_penalty_diff_pos_opp +
  home + susp_diff + HZ

model_vars <- c("final_int", "time", "gls_diff_pos_opp", "bet_win_prob",
                "susp_diff",
                "mom_conversion_diff_pos_opp", "mom_shot_pct_diff_pos_opp",
                "mom_turnover_diff_pos_opp", "mom_penalty_diff_pos_opp",
                "home", "HZ", "matchid")

# Fit the deployed K-segment blend on a training set and predict on newdata
# (win/draw/loss for the team in possession).
fit_blend <- function(train, formula = formula_gam) {
  fit_kblend(train, function(d) fit_model("gam", d, formula))
}
predict_blend <- function(fit, newdata) {
  p <- predict_kblend(fit, newdata, function(f, nd) predict_model(f, nd, "gam"))
  colnames(p) <- c("win", "draw", "loss")
  as.data.frame(p)
}

# =============================================================================
# 3. Out-of-fold win-probability predictions for every possession
# =============================================================================
cat("Fitting model and predicting win probabilities...\n")
preds_pos <- matrix(NA_real_, nrow = nrow(daten), ncol = 3,
                    dimnames = list(NULL, c("win", "draw", "loss")))

if (USE_INSAMPLE) {
  fit <- fit_blend(daten |> dplyr::select(all_of(model_vars)))
  preds_pos[] <- as.matrix(predict_blend(fit, daten))
} else {
  for (s in sort(unique(daten$season))) {
    cat("  OOF fold: holding out season", s, "\n")
    tr  <- daten |> filter(season != s) |> dplyr::select(all_of(model_vars))
    te  <- daten |> filter(season == s)
    fit <- fit_blend(tr)
    preds_pos[daten$season == s, ] <- as.matrix(predict_blend(fit, te))
  }
}

# =============================================================================
# 4. Convert to a fixed (home-team) perspective
# =============================================================================
# The model predicts from the possessing team's view. Re-express every row as
# the HOME team's win/draw/loss probability so a match has one coherent curve:
#   home in possession  -> home_win = pos_win,  home_loss = pos_loss
#   away in possession  -> home_win = pos_loss, home_loss = pos_win
# (draw is symmetric).
daten$p_home_win  <- ifelse(daten$is_home_pos == 1, preds_pos[, "win"],  preds_pos[, "loss"])
daten$p_home_draw <- preds_pos[, "draw"]
daten$p_home_loss <- ifelse(daten$is_home_pos == 1, preds_pos[, "loss"], preds_pos[, "win"])
# Win probability for the team IN POSSESSION (kept for the story-stat curve)
daten$p_pos_win   <- preds_pos[, "win"]
# Expected league points of the HOME team, normalized to [0, 1]: a draw is worth
# a point, so a possession that turns a certain loss into a draw has real value
# even though it moves the win probability not at all. Valuing possessions by
# P(win) alone would score that as zero---the same blind spot the goalkeeper
# decision has (Use Case C), and the reason the measure below is value added
# rather than win-probability added. Note v_home + v_away = 1, so the quantity
# stays zero-sum and the possessing-team sign flip below is unchanged.
daten$v_home      <- daten$p_home_win + 0.5 * daten$p_home_draw

# =============================================================================
# 5. USE CASE A — Fan engagement: win probability as a story stat
# =============================================================================
# Per-match summaries of drama, all from the home-team win-probability curve.
match_drama <- daten |>
  group_by(matchid) |>
  arrange(pos_no, .by_group = TRUE) |>
  summarize(
    season          = dplyr::first(season),
    home_team       = dplyr::first(team_home),
    n_pos           = dplyr::n(),
    # biggest single-possession swing in home win probability (all matches)
    max_swing       = max(abs(diff(p_home_win)), na.rm = TRUE),
    final_home_wp   = dplyr::last(p_home_win),
    # ~9% of matches end in a draw (winner == "draw"); "comeback" needs a
    # winner, so it is defined only for decided matches.
    decided         = dplyr::last(winner) != "draw",
    home_won        = as.integer(dplyr::last(winner) == dplyr::last(team_home)),
    # comeback size: how low did the eventual winner's win prob fall?
    #   home won -> min(p_home_win);  away won -> min(p_home_loss)
    winner_min_wp   = ifelse(dplyr::last(winner) == dplyr::last(team_home),
                             min(p_home_win,  na.rm = TRUE),
                             min(p_home_loss, na.rm = TRUE)),
    .groups = "drop"
  ) |>
  # comeback score: how far below a coin flip the eventual winner sank
  # (larger = more dramatic recovery); NA for drawn matches
  mutate(comeback = ifelse(decided, pmax(0, 0.5 - winner_min_wp), NA_real_))

cat("\n=== Most dramatic matches by largest single-possession WP swing ===\n")
match_drama |> arrange(desc(max_swing)) |> head(10) |> print()

cat("\n=== Biggest comebacks (eventual winner's lowest win probability) ===\n")
match_drama |> filter(!is.na(comeback)) |> arrange(desc(comeback)) |> head(10) |> print()

# ---- Auto-save: the biggest comebacks -----------------------------------------
# Ranked by how low the eventual winner's win probability sank, which is the
# story-stat a reader recognises: not "the sharpest single moment" but "how dead
# were they". The single-possession swing was tried first and is a poorer table:
# under the post-possession convention the largest swing of a close match is
# almost always its final possession, so every row came out at minute 60 and the
# column carried no information.
#
# Defined only for decided matches---a comeback needs a winner---and measured
# on the winner's own curve, so a home and an away comeback are comparable.
cb_series <- daten |>
  group_by(matchid) |>
  arrange(pos_no, .by_group = TRUE) |>
  mutate(winner_name    = as.character(dplyr::last(winner)),
         winner_is_home = dplyr::last(winner) == dplyr::last(team_home),
         decided        = dplyr::last(winner) != "draw",
         # ifelse() needs a full-length test, so winner_is_home is recycled by
         # mutate first; a scalar test would silently return a scalar.
         winner_wp      = ifelse(winner_is_home, p_home_win, p_home_loss)) |>
  ungroup() |>
  filter(decided, !is.na(winner_wp))

N_COMEBACKS <- 5                  # rows in Table "comebacks"; drives the caption
top_comebacks <- cb_series |>
  group_by(matchid) |>
  slice_min(winner_wp, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(winner_wp) |>
  head(N_COMEBACKS) |>
  transmute(
    match   = paste0(as.character(team_home), " vs. ", as.character(team_away)),
    season  = as.character(season),
    winner  = winner_name,
    # The other club, and whether the winner was at home---both needed for a
    # table that names each team once instead of twice.
    loser   = ifelse(winner_name == as.character(team_home),
                     as.character(team_away), as.character(team_home)),
    winner_home = winner_name == as.character(team_home),
    minute  = round(time / 60, 1),
    low_pp  = round(100 * winner_wp, 1)
  )

cat("\n=== Biggest comebacks: the winner's lowest win probability ===\n")
print(top_comebacks)

# NAMING THE TWO TEAMS ONCE. An earlier layout carried a "Match" column reading
# "home vs. away" AND a "Winner" column, so one club was printed twice and the
# row ran off the page: Bundesliga names are long ("Handball Sport Verein
# Hamburg" alone is twenty-nine characters). Winner and opponent as their own
# columns say the same thing in two names rather than three, and a bracketed H
# or A on the winner keeps the home information the old ordering carried.
cb_tex <- data.frame(
  Winner             = paste0(top_comebacks$winner,
                              ifelse(top_comebacks$winner_home, " (H)", " (A)")),
  Opponent           = top_comebacks$loser,
  Season             = top_comebacks$season,
  Minute             = formatC(top_comebacks$minute, format = "f", digits = 1),
  `Lowest P(win), \\%` = formatC(top_comebacks$low_pp, format = "f", digits = 1),
  check.names = FALSE, stringsAsFactors = FALSE
)
save_latex_table(
  cb_tex, "comebacks",
  caption = paste0("The ", N_COMEBACKS, " biggest comebacks in the five seasons: ",
                   "matches ranked by how far the eventual winner's win ",
                   "probability fell, with the minute at which it bottomed out. ",
                   "The probability is read on the winner's own curve, so home ",
                   "and away comebacks are on the same scale, and drawn matches ",
                   "are excluded since a comeback needs a winner. (H) and (A) ",
                   "mark whether the winning side played at home or away."),
  label = "tab:comebacks",
  align = "lllcc",
  # Club names are long enough that \small still overruns the text block.
  size = "footnotesize",
  # The last header carries a literal \% , which .latex_escape would turn into
  # \textbackslash{}\% ; the other four are plain words, so the whole header is
  # safe to pass through verbatim.
  escape_header = FALSE)
save_scalar("cbLowest",  formatC(min(top_comebacks$low_pp), format = "f", digits = 1))
save_scalar("cbTopMatch", top_comebacks$match[1])
save_scalar("cbTopSeason", top_comebacks$season[1])
save_scalar("cbTopWinner", top_comebacks$winner[1])

# Trajectory plot helper for a chosen match. Two stacked panels (win
# probability above, actual home score lead below) rather than a dual axis.
plot_wp_trajectory <- function(match_id) {
  d <- daten |> filter(matchid == match_id) |> arrange(pos_no) |>
    mutate(minute     = time / 60,
           home_lead  = (gls_pos - gls_opp) * ifelse(is_home_pos == 1, 1, -1))
  home <- as.character(d$team_home[1]); away <- as.character(d$team_away[1])

  # THE CURVE HAS TO REACH THE WHISTLE. The last recorded possession ends a few
  # seconds short of 60:00 (the median is 59:57), and the model evaluated there
  # is close to certain but not certain---so without this the trajectory stops
  # early at something like 0.94 for a match that was over. We append the
  # terminal point: at the final whistle the home team has won or it has not,
  # and P(home win) is 1 or 0 by definition. This is the same boundary condition
  # the goalkeeper decomposition imposes (see state_probs); it is the realized
  # result, not a prediction, and the score panel extends with it so both panels
  # now run the full sixty minutes.
  fin            <- d[nrow(d), ]
  fin$minute     <- 60
  fin$p_home_win <- as.numeric(fin$home_lead > 0)
  d              <- rbind(d, fin)

  # Club names are long, so the default title size runs past the panel and gets
  # clipped. Wrap onto a second line and shrink the title a little.
  wp_title <- paste(strwrap(sprintf("Win-probability trajectory: %s vs. %s",
                                    home, away), width = 60), collapse = "\n")

  p_wp <- ggplot(d, aes(minute, p_home_win)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60") +
    geom_line(linewidth = 0.9, color = "#0077BB") +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = wp_title,
         subtitle = "Model P(home win) over the match",
         x = NULL, y = "P(home win)") +
    theme_minimal() +
    theme(plot.title = element_text(size = 10),
          plot.subtitle = element_text(size = 8))

  p_score <- ggplot(d, aes(minute, home_lead)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
    geom_step(color = "grey40", linewidth = 0.6) +
    # The score lead is a count of goals, so only integer breaks are meaningful;
    # pretty() will happily return half-goal steps on a narrow range.
    scale_y_continuous(breaks = function(lims) {
      b <- pretty(lims)
      b[b == round(b)]
    }) +
    labs(x = "game time (min)", y = "home score lead") +
    theme_minimal()

  # stacked; if patchwork is available, combine into one grob, else return a list
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- patchwork::wrap_plots(p_wp, p_score, ncol = 1, heights = c(2, 1))
    print(combined)
    return(invisible(combined))
  }
  print(p_wp); print(p_score)
  invisible(list(wp = p_wp, score = p_score))
}

# Example: plot the biggest comeback. Saved for the manuscript's fan-engagement
# figure (fig:trajectory -> generated/win_prob_trajectory.pdf).
biggest_comeback_id <- match_drama |> filter(!is.na(comeback)) |>
  arrange(desc(comeback)) |> dplyr::slice(1) |> pull(matchid)
p_trajectory <- plot_wp_trajectory(biggest_comeback_id)
if (inherits(p_trajectory, "ggplot") || inherits(p_trajectory, "patchwork"))
  save_plot(p_trajectory, "win_prob_trajectory", height = 4.6)

# =============================================================================
# 6. USE CASE B — Value added (VA)
# =============================================================================
# VALUE ADDED (VA) of a possession = the change in the ATTACKING team's expected
# league points the possession itself produced: the value of the state it leaves
# behind less the value of the state it was played from. Every row already holds
# a post-possession state, so in home-team terms that is
# delta_home = v_home[i] - v_home[i-1]; the possessing team's VA is +delta_home
# if it is the home team, -delta_home otherwise (v_home + v_away = 1, so the
# flip is exact).
daten <- daten |>
  group_by(matchid) |>
  arrange(pos_no, .by_group = TRUE) |>
  mutate(
    # Value added BY this possession: its post-state minus the post-state of
    # the one before. Under the post-possession convention every row already
    # records the value the possession produced, so the difference looks
    # BACKWARD. It looked forward while rows held pre-possession states, which
    # left the deciding attack of a match---the row with no successor---with
    # no value added at all. Now it is the opening possession of a match that
    # has no predecessor and drops out, which costs nothing: at 0:0 in the
    # first minute there is no leverage to measure.
    delta_home = v_home - dplyr::lag(v_home),
    va_pos     = ifelse(is_home_pos == 1, delta_home, -delta_home)
  ) |>
  ungroup()

# THE EVALUATION SEASON. The player figures below are computed on the LAST
# season only, from a model that never saw it. No extra fit is needed for that:
# the leave-one-season-out loop of Section 3 already holds out each season in
# turn, so the predictions carried by the final season's rows come from exactly
# the model trained on the four seasons before it. Restricting to that season is
# what makes the leaderboard an out-of-sample statement rather than a
# description of the data the model was fit on---and it stops a player's figure
# from being an average over five seasons of different squads and roles.
VA_SEASON <- max(as.character(daten$season))
if (USE_INSAMPLE)
  stop("Use case B needs out-of-fold predictions; set USE_INSAMPLE <- FALSE.")
cat(sprintf("\nValue added evaluated on %s (model trained on the %d earlier seasons).\n",
            VA_SEASON, length(unique(daten$season)) - 1L))

# SHOTS, NOT GOALS, are the unit. A goal always adds value, so averaging VA over
# a player's goals measures only the leverage of the moments they scored in, and
# nothing about whether they converted the chances they took. Averaging over
# every SHOT---goals and misses alike, since a miss hands the ball over and
# carries negative VA---makes the quantity an efficiency: the expected league
# points a player adds per attempt. Seven-metre throws are included; they are
# shots the player took and converted or did not.
shots <- daten |>
  filter(season == VA_SEASON, goal == 1 | missed_shot == 1, !is.na(va_pos))
goals <- daten |> filter(goal == 1, !is.na(va_pos))

cat("\n=== Highest-leverage goals by VA (value added, expected league points) ===\n")
goals |>
  transmute(matchid, season, minute = round(time / 60, 1),
            scoring_team = team, score = paste0(gls_pos, ":", gls_opp),
            va = round(va_pos, 3)) |>
  arrange(desc(va)) |>
  head(15) |>
  print()

# Average VA of a goal by game phase and score closeness
goal_value <- goals |>
  mutate(time_bin  = pmin(floor(time / 300) * 5, 55),
         close     = abs(gls_diff_pos_opp) <= 2) |>
  group_by(time_bin, close) |>
  summarize(mean_va = mean(va_pos, na.rm = TRUE), n = dplyr::n(), .groups = "drop")

p_goal_value <- ggplot(goal_value, aes(x = time_bin, y = mean_va,
                       color = ifelse(close, "close (<=2)", "not close"))) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.6) +
  labs(title = "What a goal is worth over the course of a match",
       subtitle = "Mean value added by a goal, by game time and score closeness",
       x = "game time (min)", y = "mean VA per goal", color = "score state") +
  theme_minimal()
# Printed for interactive inspection but deliberately NOT saved: the manuscript
# makes the same point in the text (leverage is driven by the game state, not by
# the scorer), and the figure added little beyond it.
print(p_goal_value)

# Player leaderboard. VA per shot is already a meaningful quantity on its own --
# expected league points added per attempt, which rewards converting chances and
# charges for missing them---but it still carries one thing that is not the
# player's: how competitive their TEAM's matches tend to be. A side that wins
# comfortably takes most of its shots in already-decided games, where nothing a
# player does moves the needle much.
#
# We therefore also report each shot against the leverage available in its own
# context: the mean VA per shot of the team it was taken for. The resulting
#
#     clutch_index = mean(player VA per shot) / mean(team baseline for those shots)
#
# is on a scale where 1 means "this player's shots came at exactly the leverage
# typical of a shot for their side", above 1 that they came in tighter
# situations and were converted well, below 1 the reverse. Because the
# denominator carries the same team context as the numerator, the
# team-competitiveness confound cancels. The baseline is attached per shot
# rather than per player, so a player who changed clubs mid-season is measured
# against whichever side they were shooting for at the time.
#
# VA_RANK_BY switches which of the two orders the table. "index" removes the
# club context; "per_shot" ranks on the raw efficiency, which is easier to
# explain but partly ranks clubs.
VA_RANK_BY  <- "index"     # or "per_shot"
VA_MIN_SHOTS <- 100        # ~110 players clear this in a single season
if ("name" %in% names(daten)) {
  shots <- shots |>
    group_by(team) |>
    mutate(team_va_per_shot = mean(va_pos, na.rm = TRUE)) |>
    ungroup()

  player_va <- shots |>
    filter(!is.na(name)) |>
    group_by(name) |>
    summarize(
      shots        = dplyr::n(),
      goals        = sum(goal == 1, na.rm = TRUE),
      total_va     = sum(va_pos, na.rm = TRUE),
      va_per_shot  = mean(va_pos, na.rm = TRUE),
      # "opportunity": the leverage a typical shot carried in the contexts this
      # player actually played in---low for dominant clubs, high for sides in
      # many tight matches. This is the quantity a raw ranking confounds with
      # individual performance.
      opportunity  = mean(team_va_per_shot, na.rm = TRUE),
      clutch_index = mean(va_pos, na.rm = TRUE) / mean(team_va_per_shot, na.rm = TRUE),
      # principal club = the team the player most often shot for this season
      club         = names(sort(table(as.character(team)), decreasing = TRUE))[1],
      .groups = "drop"
    )
  # Kept unfiltered as well: the club table below uses a lower bar, because a
  # single squad has only a handful of players clearing the league-wide one.
  player_va_all <- player_va
  player_va <- player_va_all |>
    filter(shots >= VA_MIN_SHOTS)     # minimum volume for a stable estimate

  # Does the normalization actually remove the confound? The raw per-goal figure
  # should track the opportunity level; the index should not.
  cor_raw_opp   <- cor(player_va$va_per_shot,  player_va$opportunity)
  cor_index_opp <- cor(player_va$clutch_index, player_va$opportunity)
  cat(sprintf("\nTeam-context confound: cor(VA/shot, opportunity) = %.2f | cor(clutch index, opportunity) = %.2f\n",
              cor_raw_opp, cor_index_opp))

  # ---- SANITY CHECK ON THE INDEX ---------------------------------------------
  # The index is a player's mean VA per shot over their team's mean VA per shot.
  # That makes it an identity, not an estimate: shot-weighting the index over
  # ALL of a team's named shooters must give exactly 1, because
  #
  #   sum_p n_p (VAbar_p / VAbar_T) / sum_p n_p
  #     = (sum_p VA_p) / (VAbar_T * N_T) = 1,
  #
  # provided every shot is attributed to a named player. Anything else means a
  # coding error, or shots whose shooter is unknown (they enter the team mean
  # but no player), so the share of those is printed too.
  #
  # WHY THIS MATTERS FOR THE REVIEW. It shows that the index cannot, on average,
  # favour or penalise any team: every club is centred on 1 by construction. So
  # "players at strong clubs are held back by dividing through their own team's
  # mean" cannot be the explanation for who appears in the table---the
  # numerator and the denominator carry the same club context and it cancels.
  # What can differ across clubs is the SPREAD, and a ratio spreads out when its
  # denominator is small. A side that loses most of its matches adds little per
  # shot on aggregate, so VAbar_T sits near zero and the ratio becomes
  # explosive. That is the candidate explanation, and it predicts something the
  # level story does not: weak clubs should crowd BOTH tails, not just the top.
  cat("\n=== Sanity check: is the index centred on 1 for every club? ===\n")
  all_shooters <- shots |>
    filter(!is.na(name)) |>
    group_by(team) |>
    summarize(
      shooters   = dplyr::n_distinct(name),
      shots      = dplyr::n(),
      va_bar_team = dplyr::first(team_va_per_shot),
      strength   = mean(bet_win_prob, na.rm = TRUE),
      .groups = "drop")
  per_player <- shots |>
    filter(!is.na(name)) |>
    group_by(team, name) |>
    summarize(n = dplyr::n(), idx = mean(va_pos) / dplyr::first(team_va_per_shot),
              .groups = "drop")
  chk <- per_player |>
    group_by(team) |>
    summarize(wtd_mean_index = sum(n * idx) / sum(n),
              unwtd_mean     = mean(idx),
              sd_index       = sd(idx),
              idx_min        = min(idx),
              idx_max        = max(idx),
              .groups = "drop") |>
    left_join(all_shooters, by = "team") |>
    arrange(va_bar_team)
  print(as.data.frame(chk |> transmute(team = substr(as.character(team), 1, 22),
                                       shots, va_bar_team, strength,
                                       wtd_mean_index, sd_index, idx_min, idx_max)),
        digits = 3, row.names = FALSE)
  cat(sprintf("\n  shots with no named shooter: %.1f%% (these enter the team mean but no player)\n",
              100 * mean(is.na(shots$name))))
  if (any(abs(chk$wtd_mean_index - 1) > 0.02))
    warning("The shot-weighted index does not come out at 1 for every club. ",
            "With every shot named it is an algebraic identity, so either some ",
            "shots have no shooter or the index is computed wrongly.")
  # THE DENOMINATOR ITSELF. A ratio is only meaningful while its denominator
  # stays well away from zero, and nothing guarantees that here: a club that
  # loses most of its matches adds little per shot on aggregate. Should any
  # club's mean come out negative, the index inverts for that club---a player
  # who added value would be scored below zero---and the statistic would be
  # unusable rather than merely noisy.
  if (any(chk$va_bar_team <= 0))
    warning("At least one club has a non-positive mean VA per shot, so its ",
            "players' index values are inverted and meaningless. The ratio ",
            "form of the index cannot be used as it stands.")
  else if (min(chk$va_bar_team) < 0.2 * stats::median(chk$va_bar_team))
    warning("The smallest club denominator is less than a fifth of the median, ",
            "so the index is far more dispersed at that club than elsewhere ",
            "for arithmetic rather than sporting reasons.")

  # Does the spread track club strength? If weak clubs have small denominators
  # and wide spreads, the table is picking up ratio instability, not clutch.
  cat(sprintf("  cor(club strength, VA per shot of the club) = %+.2f\n",
              cor(chk$strength, chk$va_bar_team, use = "complete.obs")))
  cat(sprintf("  cor(club strength, spread of the index)     = %+.2f\n",
              cor(chk$strength, chk$sd_index, use = "complete.obs")))

  # BOTH TAILS. If the ratio is unstable at weak clubs, their players should
  # crowd the bottom of the index as well as the top. If instead the bottom is
  # made up of strong clubs, the level story would be the right one after all.
  cat("\n=== Both tails of the index (>= VA_MIN_SHOTS shots) ===\n")
  tails <- player_va |> transmute(name = substr(name, 1, 20), club = substr(club, 1, 20),
                                  shots, va_per_shot = round(va_per_shot, 4),
                                  index = round(clutch_index, 2))
  cat("  top 10:\n");    print(as.data.frame(tails |> arrange(desc(index)) |> head(10)), row.names = FALSE)
  cat("  bottom 10:\n"); print(as.data.frame(tails |> arrange(index)      |> head(10)), row.names = FALSE)
  # How many clubs appear at BOTH ends. This is the number the manuscript
  # quotes against the reading that the index favours weaker sides: a club that
  # places a player at the top and another at the bottom is not being favoured,
  # it is simply spread out around the one it is centred on.
  top_clubs <- unique((player_va |> arrange(desc(clutch_index)) |> head(10))$club)
  bot_clubs <- unique((player_va |> arrange(clutch_index)       |> head(10))$club)
  n_both <- length(intersect(top_clubs, bot_clubs))
  cat(sprintf("  clubs in BOTH tails: %d of %d distinct clubs in the top ten\n",
              n_both, length(top_clubs)))
  save_scalar("vaTailBoth",  n_both)
  save_scalar("vaTailTopN",  length(top_clubs))

  # Correlations behind the two aggregation stories (reported in the manuscript):
  # total VA tracks shot volume; VA per shot is nearly orthogonal to it.
  cor_total_vol   <- cor(player_va$total_va,    player_va$shots)
  cor_pershot_vol <- cor(player_va$va_per_shot, player_va$shots)
  cat(sprintf("\nVA correlations (players with >= %d shots): total~shots = %.2f | per-shot~shots = %.2f\n",
              VA_MIN_SHOTS, cor_total_vol, cor_pershot_vol))

  N_VA_LEADERS <- 10              # rows in Table "va"; drives the caption too
  va_leader <- player_va |>
    arrange(desc(if (VA_RANK_BY == "index") clutch_index else va_per_shot)) |>
    head(N_VA_LEADERS)

  cat(sprintf("\n=== Leading players, %s, ranked by %s (>= %d shots) ===\n",
              VA_SEASON, VA_RANK_BY, VA_MIN_SHOTS))
  print(va_leader |>
          transmute(name, club, shots, goals,
                    va_per_shot  = round(va_per_shot, 4),
                    clutch_index = round(clutch_index, 3)))
  cat("\nNote: the index is relative to the player's own team context; it does not\n",
      "separate skill from sampling noise, and a per-shot average over one season\n",
      "remains noisy.\n")

  # ---- Auto-save: VA leaderboard table for the manuscript -------------------
  va_tex <- data.frame(
    `#`         = as.character(seq_len(nrow(va_leader))),
    Player      = va_leader$name,
    Club        = va_leader$club,
    Shots       = as.character(va_leader$shots),
    Goals       = as.character(va_leader$goals),
    `VA/shot`   = formatC(va_leader$va_per_shot, format = "f", digits = 4),
    Index       = formatC(va_leader$clutch_index, format = "f", digits = 2),
    check.names = FALSE, stringsAsFactors = FALSE
  )
  save_latex_table(
    va_tex, "va",
    caption = paste0("Leading players in ", VA_SEASON, ", the season held out of ",
                     "the model's training data, among those with at least $",
                     VA_MIN_SHOTS, "$ shots. ``VA/shot'' is the mean value added ",
                     "over every shot the player took, goals and misses alike, ",
                     "in expected league points; the index is that figure ",
                     "divided by the mean over all shots taken for the same ",
                     "club, so that how competitive a side's matches tend to be ",
                     "cancels out. Ranked by ",
                     ifelse(VA_RANK_BY == "index", "the index", "VA per shot"), "."),
    label = "tab:va",
    align = "rllrrrr")

  # ---- Auto-save: the same index inside the champion's squad -----------------
  # A reviewer read the league-wide leaderboard as showing that players at strong
  # clubs are held back by the division through their own club's mean. The
  # sanity check above rules that out arithmetically---every club is centred on
  # one---but the leaderboard still looks the way it looks, and the reason is
  # worth showing rather than asserting: the strongest squads are the most
  # evenly spread, so their players sit near one and reach neither tail. The
  # league's top four by strength have among the smallest index spreads of the
  # eighteen.
  #
  # This table therefore reads the index INSIDE the champion's squad. It is the
  # direct answer to the question the leaderboard provokes---who carried the
  # decisive shots for the side that actually won the league---and it shows the
  # index doing its job at a club whose matches are rarely close.
  #
  # The champion is derived from the season's results rather than named here, so
  # the table follows the data if the season ever changes.
  VA_CLUB_MIN_SHOTS <- 60
  season_results <- daten |>
    filter(season == VA_SEASON) |>
    group_by(matchid) |>
    summarize(home = as.character(dplyr::last(team_home)),
              away = as.character(dplyr::last(team_away)),
              win  = as.character(dplyr::last(winner)), .groups = "drop")
  standings <- bind_rows(
      season_results |> transmute(club = home,
                                  pts = ifelse(win == "draw", 1L, ifelse(win == home, 2L, 0L))),
      season_results |> transmute(club = away,
                                  pts = ifelse(win == "draw", 1L, ifelse(win == away, 2L, 0L)))) |>
    group_by(club) |>
    summarize(points = sum(pts), .groups = "drop") |>
    arrange(desc(points))
  VA_CHAMPION <- standings$club[1]
  cat(sprintf("\n=== The index inside the champion's squad: %s (%d points) ===\n",
              VA_CHAMPION, standings$points[1]))

  champ_va <- player_va_all |>
    filter(club == VA_CHAMPION, shots >= VA_CLUB_MIN_SHOTS) |>
    arrange(desc(clutch_index))
  print(as.data.frame(champ_va |>
          transmute(name, shots, goals,
                    va_per_shot  = round(va_per_shot, 4),
                    clutch_index = round(clutch_index, 2))), row.names = FALSE)

  champ_tex <- data.frame(
    `#`       = as.character(seq_len(nrow(champ_va))),
    Player    = champ_va$name,
    Shots     = as.character(champ_va$shots),
    Goals     = as.character(champ_va$goals),
    `VA/shot` = formatC(champ_va$va_per_shot, format = "f", digits = 4),
    Index     = formatC(champ_va$clutch_index, format = "f", digits = 2),
    check.names = FALSE, stringsAsFactors = FALSE)
  save_latex_table(
    champ_tex, "va_champion",
    caption = paste0("The same measure read inside one squad: every ",
                     VA_CHAMPION, " player with at least $", VA_CLUB_MIN_SHOTS,
                     "$ shots in ", VA_SEASON, ", the season they won the ",
                     "league. The bar is lower than in Table~\\ref{tab:va} ",
                     "because a single squad has few players clearing the ",
                     "league-wide one. By construction the shot-weighted index ",
                     "over a club's shooters is exactly $1$, so these values ",
                     "say which of the champion's players took the shots that ",
                     "mattered most, not how the club compares with others. ",
                     "That the spread here is narrow is the point: at a side ",
                     "whose matches are rarely close, the decisive shots are ",
                     "shared out rather than carried by one player, which is ",
                     "why such squads are thinly represented at either end of ",
                     "Table~\\ref{tab:va}."),
    label = "tab:va_champion",
    align = "rlrrrr")

  save_scalar("vaChampion",       VA_CHAMPION)
  save_scalar("vaChampionPoints", standings$points[1])
  save_scalar("vaClubMinShots",   VA_CLUB_MIN_SHOTS)
  save_scalar("vaChampionN",      nrow(champ_va))
  save_scalar("vaChampionHi", formatC(max(champ_va$clutch_index), format = "f", digits = 2))
  save_scalar("vaChampionLo", formatC(min(champ_va$clutch_index), format = "f", digits = 2))

  save_scalar("vaSeason",         VA_SEASON)
  save_scalar("vaMinShots",       VA_MIN_SHOTS)
  save_scalar("vaCorTotalVol",    formatC(cor_total_vol,   format = "f", digits = 2))
  save_scalar("vaCorPershotVol",  formatC(cor_pershot_vol, format = "f", digits = 2))
  save_scalar("vaCorRawOpp",       formatC(cor_raw_opp,       format = "f", digits = 2))
  save_scalar("vaCorIndexOpp",     formatC(cor_index_opp,     format = "f", digits = 2))
  save_scalar("vaIndexLo", formatC(min(va_leader$clutch_index), format = "f", digits = 2))
  save_scalar("vaIndexHi", formatC(max(va_leader$clutch_index), format = "f", digits = 2))
}

# =============================================================================
# 7. USE CASE C — Goalkeeper-substitution decision
# =============================================================================
# Primary: a win-probability DECOMPOSITION, exactly as American-football
# fourth-down "go for it" models work. We do NOT toggle the gk_pulled covariate
# (that coefficient is confounded). Instead we compare the two actions through
# their consequences: pulling raises (here: barely) the own scoring rate but
# raises the opponent's next-possession scoring rate (empty-net risk), so
#   WP(a, s) = sum_o P(o | a, s) * WP(resulting state),
# with the resulting states valued by the fitted WP model with the keeper back
# in goal. The action enters only through the outcome rates, which sidesteps the
# confounding and gives the prospective decision at every state.
# Complement: the retrospective residual analysis (further below).
cat("\n=== Use case C: goalkeeper-substitution decision (WP decomposition) ===\n")

# ---- 7A. The two outcomes an exchange can turn on ----------------------------
# Own goal this attack, and whether the opponent scores on the very next
# possession. NOTE the downside is just "the opponent scores next", NOT
# "scores into an empty net": teams often re-insert the keeper before the
# opponent shoots. So opp_scores_next is the realized opponent-scoring rate
# under each action (already reflecting keeper re-insertion); its keep-vs-pull
# DIFFERENCE is the true downside of pulling, whether the goal came into an
# empty net or not. Whether the net was in fact still empty when the reply came
# is built separately below as net_empty_next, and the worst-case charge of 7C-1
# needs it: charging every reply as an empty-net goal is only the worst case for
# a team that stays pulled, and most do not.
#
# CONFOUNDING WARNING: mostly weak, trailing teams pull, so raw marginal rates
# conflate the action with team strength---comparing them directly is wrong. We
# condition every rate on the same game state the win-probability model uses
# (game time, score difference, matchup strength, home) and read each rate at
# the DECISION team's own state. Even so, strong teams essentially never pull,
# so evaluating the decision at high strength is extrapolation (common support).
nxt <- daten |>
  arrange(matchid, pos_no) |>
  group_by(matchid) |>
  transmute(pos_no,
            opp_scores_next = as.integer(!is.na(lead(as.character(team))) &
              lead(as.character(team)) == as.character(opp) & lead(goal) == 1)) |>
  ungroup()
# select(-any_of(...)) so that re-sourcing this section into a live session is
# safe: joining a column that `daten` already carries would give
# opp_scores_next.x / .y and silently leave opp_scores_next NULL.
daten <- daten |>
  dplyr::select(-dplyr::any_of("opp_scores_next")) |>
  left_join(nxt, by = c("matchid", "pos_no"))
# Whether a next possession exists at all, kept separately: opp_scores_next
# deliberately codes "no next possession" as 0 (the score is unchanged either
# way), so it cannot be used to measure how often the match simply ends.
daten <- daten |> group_by(matchid) |> arrange(pos_no, .by_group = TRUE) |>
  mutate(lead_pos_no = lead(pos_no)) |> ungroup()

# WAS THE NET ACTUALLY STILL EMPTY WHEN THE REPLY CAME? This is the quantity the
# worst-case charge below stands or falls on, and it is observable after all.
# On the reply row the opponent is in possession, so no_field_players_opp on
# that row counts OUR outfield players while we defend: seven of them means the
# keeper had not made it back by the time the reply ended, six that they had.
# Read strictly and without the ten-second tolerance, which is right here --
# what matters is the court at the moment the reply concluded, not a window
# around it.
#
# The one thing it cannot separate is a suspension: a team that is a man down
# and empty-net also defends with six outfield players. Both arms are matched at
# full strength during our own possession, so this can only enter through a
# penalty awarded during the reply itself, which is rare.
nxt_net <- daten |>
  arrange(matchid, pos_no) |>
  group_by(matchid) |>
  transmute(pos_no,
            net_empty_next = ifelse(
              is.na(lead(as.character(team))) |
                lead(as.character(team)) != as.character(opp),
              NA_integer_,
              as.integer(as.character(lead(no_field_players_opp)) == "7"))) |>
  ungroup()
daten <- daten |>
  dplyr::select(-dplyr::any_of("net_empty_next")) |>
  left_join(nxt_net, by = c("matchid", "pos_no"))

# Exchange duration: the clock advance across our attack plus the opponent's
# reply. Used, averaged by action, as a data-based proxy for how far the clock
# runs---better than a fixed constant, though the number and timing of the
# REMAINING possessions still needs a state-space model.
daten <- daten |> group_by(matchid) |>
  # The exchange starts when our attack starts (time_prev) and ends when the
  # opponent's reply ends, which is the terminating stamp of the NEXT row.
  mutate(exch_dur = lead(time, 1) - time_prev) |> ungroup()

# The well-defined decision here is the STANDARD seven-against-six attacking
# choice: at full strength against a 6-outfield defense, keep the keeper
# (6-vs-6) or pull for a 7th attacker (7-vs-6, empty net). We match BOTH arms to
# that context (opponent at 6) and exclude the penalty-driven "refill to equal
# numbers" pull (own suspension: 5 outfield + keeper vs 6 outfield with the net
# empty), which is a different decision left to the companion paper.
#
# The two rates below are descriptive frequencies, not fitted models. The
# win-probability model of 01/Section 2 enters only in 7B, as the value function
# that scores the states the exchange leads to.
# Both arms read with the ten-second tolerance, so an attack played seven
# against six for all but its closing moments counts as one. Read strictly it
# would not, and the pulled arm---the scarce one---loses about a third of its
# possessions to that alone: 9,518 against 6,977.
keep_set <- daten |> filter(gk_pos_10s == 1, no_field_players_pos_10s == "6", no_field_players_opp == "6")
pull_set <- daten |> filter(gk_pos_10s == 0, no_field_players_pos_10s == "7", no_field_players_opp == "6")

# Median exchange duration by action. Not used by the arithmetic below---the
# closing-seconds argument has the exchange run to the whistle---but printed
# because it says how long the two possessions it covers actually take.
cat(sprintf("  exchange duration (s): keep = %.0f | pull = %.0f\n",
            median(keep_set$exch_dur, na.rm = TRUE),
            median(pull_set$exch_dur, na.rm = TRUE)))
cat("  NOTE: the marginal rates below are confounded by team strength (weak teams\n")
cat("  pull). The like-for-like set-attack comparison that follows is the usable one.\n")
cat(sprintf("  own-goal rate      keep = %.3f | pull = %.3f  (marginal, confounded)\n",
            mean(keep_set$goal), mean(pull_set$goal)))
# The marginal scoring rates are reported in the paper alongside the conditioned
# ones, because the two disagree and the disagreement is the point. Pooled over
# every possession the pull arm looks WORSE, but that comparison is between
# populations rather than actions: pulled possessions are set attacks almost by
# construction, kept ones are a mixture that is roughly a third fast breaks,
# and fast breaks convert best. Conditioning on set attacks removes that, and
# the confounding by team strength---it is mostly weaker teams that pull --
# still runs against the pull afterwards.
save_scalar("rawQkeep", formatC(mean(keep_set$goal), format = "f", digits = 3))
save_scalar("rawQpull", formatC(mean(pull_set$goal), format = "f", digits = 3))
# Like for like. A keeper can only be pulled in a set attack, so the comparison
# excludes fast breaks---no pulling team could have played one, and they
# convert well, so leaving them in flatters the keep arm.
#
# SEVEN-METRE THROWS STAY IN. An earlier version dropped them, reasoning that
# the extra attacker cannot matter once the throw is awarded. That confuses the
# throw with the possession: winning a seven-metre is one of the ways an attack
# can succeed, and whether seven attackers make it more or less likely is
# exactly the sort of thing the comparison should be measuring, not defining
# away. The classification is sound for these rows too, since the recorded
# goalkeeper state at a seven-metre refers to the moment of the foul, which is
# the in-attack configuration we are conditioning on.
#
# What the data then say is that the channel is close to neutral: the two arms
# draw seven-metre throws at nearly the same rate and convert them at nearly
# the same rate, so including them neither flatters nor penalizes the pull. The
# printed rates are saved as scalars so the manuscript can state whichever way
# they come out rather than asserting a direction.
set_keep <- keep_set |> filter(time_pos > 20)
set_pull <- pull_set |> filter(time_pos > 20)
cat(sprintf("  own-goal rate, SET ATTACKS (>20 s, 7m included): keep = %.3f (n=%d) | pull = %.3f (n=%d)\n",
            mean(set_keep$goal), nrow(set_keep), mean(set_pull$goal), nrow(set_pull)))
cat(sprintf("  seven-metre throws per set attack:               keep = %.3f | pull = %.3f\n",
            mean(set_keep$penalty == 1), mean(set_pull$penalty == 1)))
save_scalar("setQkeep", formatC(mean(set_keep$goal), format = "f", digits = 3))
save_scalar("setQpull", formatC(mean(set_pull$goal), format = "f", digits = 3))
# The seven-metre channel, reported because it is the one a reader will ask
# about and it does not run the way intuition suggests.
save_scalar("setPenKeep", formatC(mean(set_keep$penalty == 1), format = "f", digits = 3))
save_scalar("setPenPull", formatC(mean(set_pull$penalty == 1), format = "f", digits = 3))
cat(sprintf("  opp-scores-next    keep = %.3f | pull = %.3f  (marginal, confounded)\n",
            mean(keep_set$opp_scores_next), mean(pull_set$opp_scores_next)))
cat(sprintf("  odds-implied win prob: pulling teams median = %.2f vs all = %.2f\n",
            median(pull_set$bet_win_prob, na.rm = TRUE),
            median(daten$bet_win_prob, na.rm = TRUE)))

# ---- 5b. Does the VALUE FUNCTION fit the endgame? ----------------------------
# The whole decomposition is valued with the win-probability model, so how sharp
# that model is in the closing seconds decides how much the exchange arithmetic
# is worth. The segmentation of 02 was selected on the worst-calibrated
# FIVE-MINUTE window, which is a resolution at which a failure confined to the
# last seconds can hide entirely. This checks it directly, on the same
# out-of-fold predictions the use cases read, against the realized outcome.
cat("\n=== Value function: predicted vs realized, closing windows (OOF) ===\n")
vf_chk <- bind_rows(lapply(c(60, 30, 20), function(s) {
  d <- daten |> filter(time > 3600 - s, !is.na(p_pos_win))
  data.frame(window = sprintf("last %d s", s), n = nrow(d),
             pred_win  = mean(d$p_pos_win),
             obs_win   = mean(as.character(d$final) == "1"),
             pred_draw = mean(preds_pos[daten$time > 3600 - s &
                                        !is.na(daten$p_pos_win), "draw"]),
             obs_draw  = mean(as.character(d$final) == "0"))
}))
print(vf_chk, digits = 3, row.names = FALSE)
vf20 <- vf_chk[vf_chk$window == "last 20 s", ]
save_scalar("vfPredWin", formatC(vf20$pred_win, format = "f", digits = 3))
save_scalar("vfObsWin",  formatC(vf20$obs_win,  format = "f", digits = 3))
save_scalar("vfPredDraw", formatC(vf20$pred_draw, format = "f", digits = 3))
save_scalar("vfObsDraw",  formatC(vf20$obs_draw,  format = "f", digits = 3))


# ---- 7B. The value function the exchange is scored with -----------------------
# Value function: the deployed blended gk GAM, fit once (in-sample is fine---we
# use it as a calibrated value function, not to measure predictive accuracy).
fit_C <- fit_blend(daten |> dplyr::select(all_of(model_vars)))

# Resulting state at the given time and score difference. The win-probability
# model has no keeper term, so nothing goalkeeper-specific need be set; the
# suspension state (susp_diff) is carried over from the decision row unchanged.
make_state <- function(base, t, d) {
  base$time             <- t
  base$gls_diff_pos_opp <- d
  base
}

# VALUE FUNCTION. Crucially this is NOT the win probability alone. A drawn
# handball match awards a league point (2 for a win, 1 for a draw), and a large
# part of why coaches pull the keeper late is to turn a certain LOSS into a
# DRAW. Valuing states by P(win) only makes that upside invisible: one goal down
# with seconds left, scoring yields a draw, which P(win) scores identically to
# losing---so pulling could never look worthwhile no matter how much it raised
# the scoring rate. We therefore value states by expected league points,
# normalized to [0, 1]:
#     v = (2*P(win) + 1*P(draw)) / 2 = P(win) + 0.5*P(draw).
# Set VALUE_DRAW_WEIGHT <- 0 to recover the win-probability-only value function.
VALUE_DRAW_WEIGHT <- 0.5

# THE FINAL WHISTLE IS NOT A FORECASTING PROBLEM. At t = 3600 the lead settles
# the match: ahead is a win, level is a draw, behind is a loss, with certainty.
# Asking the GAM for those states gets an answer that is merely close: the check
# printed below values a level scoreline at the buzzer well short of 0.50 and a
# one-goal deficit well above 0.00. The smooth cannot be perfectly sharp at the
# edge of its own domain, and the segmentation was tuned on five-minute windows,
# so the last seconds sit inside a window rather than being resolved.
#
# That error is not neutral for the decision this section is about. It shrinks
# the gap between equalizing and not equalizing---the entire payoff a late pull
# plays for (the diagnostic below prints the two values it is computed from).
# Small once the closing segment is fine enough to resolve the last seconds, but
# one-sided, and it runs AGAINST the pull in
# exactly the states it is used in. Imposing the terminal values is also what
# makes the closing-seconds argument below exact rather than approximate: with
# them, V(a) = 1/2 q_a (1 - r_a) is an identity, not a fit.
#
# Imposing the terminal values is not a correction applied to a forecast; it is a
# boundary condition the process satisfies by definition, and every state-space
# treatment of an endgame starts from it. `state_probs` returns it at the whistle
# and the fitted model everywhere else, so the break-even arithmetic and the
# value-function figure read the same function.
terminal_probs <- function(d) data.frame(
  win  = as.numeric(d > 0),
  draw = as.numeric(d == 0),
  loss = as.numeric(d < 0))
state_probs <- function(base, t, d) {
  if (t >= 3600) return(terminal_probs(d))
  predict_blend(fit_C, make_state(base, t, d))
}
state_value <- function(base, t, d) {
  p <- state_probs(base, t, d)
  p$win + VALUE_DRAW_WEIGHT * p$draw
}

# ---- 7C. The state the decision is read at -----------------------------------
# A strength TYPICAL OF TEAMS THAT ACTUALLY PULL (median odds among pulls).
# This is the state the FIGURE is drawn at, and the warning it used to carry --
# that an "even matchup" reading extrapolates to strong teams that never pull --
# is about the states a figure spanning a two-goal deficit covers. It does not
# carry over to the level-score case study below, which overrides this to an
# even matchup deliberately: a tied match with half a minute left is not a state
# only weak teams reach, and the two rate models are adjusted for strength
# rather than restricted to it. The difference between the two is real and worth
# knowing when reading the figure against the numbers: pulling teams sit near
# 0.35 on the odds-implied scale, the case study reads at 0.50.
pull_bwp <- median(pull_set$bet_win_prob, na.rm = TRUE)
neutral <- daten[1, model_vars, drop = FALSE]
neutral$bet_win_prob <- pull_bwp
neutral$home <- factor("1", levels = levels(daten$home))
neutral$HZ   <- 0
for (m in c("mom_conversion_diff_pos_opp", "mom_shot_pct_diff_pos_opp",
            "mom_turnover_diff_pos_opp", "mom_penalty_diff_pos_opp")) neutral[[m]] <- 0

# How far the fitted model is from the boundary condition imposed in
# `state_probs`. Printed and saved because the size of this gap is the whole
# argument for imposing it, and because it should shrink once 02 is re-tuned:
# under the post-possession convention the whistle is now IN the training data,
# where under the previous one the last row of a match preceded the final attack
# and the smooth had to extrapolate to reach the buzzer at all.
cat("\n=== Value function at the final whistle: fitted vs. definitional ===\n")
whistle_chk <- do.call(rbind, lapply(c(-2, -1, 0, 1), function(d) {
  p <- predict_blend(fit_C, make_state(neutral, 3600, d))
  data.frame(lead         = d,
             fitted       = p$win + VALUE_DRAW_WEIGHT * p$draw,
             definitional = as.numeric(d > 0) + VALUE_DRAW_WEIGHT * as.numeric(d == 0))
}))
print(whistle_chk, digits = 3, row.names = FALSE)
save_scalar("whistleFitLevel",
            formatC(whistle_chk$fitted[whistle_chk$lead == 0], format = "f", digits = 3))
save_scalar("whistleFitDown",
            formatC(whistle_chk$fitted[whistle_chk$lead == -1], format = "f", digits = 3))

# ---- 7C-1. The decision in the closing seconds, without estimating a contrast -
# This is a case study, and it is built to need as little estimation as it can.
# Earlier drafts fitted the two rates as functions of the full game state and
# read the pull-versus-keep contrast off them at a point. That does not work
# here and the reason is worth stating: what such a model is asked for is not an
# average rate over many possessions, where estimation noise cancels, but the
# DIFFERENCE of two predictions at ONE point of a six-dimensional space, from
# roughly seventy late pulls per arm. Refits differed in the SIGN of the
# contrast. No amount of averaging turns seventy possessions into an answer.
#
# The endgame, however, does not require one. Take the decision at the moment a
# possession begins with the clock nearly out and the team one goal down. The
# exchange then runs to the whistle, so the resulting states are settled by
# their lead---1/2 for the equaliser, 0 otherwise---and Equation (gk_decomp)
# collapses to
#
#     V(a) = 1/2 * q_a * (1 - r_a),
#
# so pulling pays exactly when q_pull (1 - r_pull) > q_keep (1 - r_keep). Only
# two things matter: how much the seventh attacker adds to the scoring chance,
# and how much the empty net adds to the opponent's.
#
# The second of those is bounded by something we can simply count. The empty net
# can only be punished on a reply that actually happens, and with fifteen
# seconds left it almost never does. So take the WORST CASE the data allow:
# every reply that occurs becomes a goal because the net is empty. That gives
# r_pull <= P(a next possession exists) and hence a break-even
#
#     dq* = q_keep * (r_pull - r_keep) / (1 - r_pull),
#
# the smallest scoring benefit that still makes pulling worthwhile. It needs no
# model of the pull at all---only observed frequencies---and it is deliberately
# pessimistic, since it charges the pull for every reply as if the keeper being
# absent guaranteed a goal.
#
# The comparison for dq* is then the like-for-like descriptive gap above
# (set attacks, no seven-metre throws), which rests on thousands of possessions
# rather than seventy and is not a fitted contrast at all.
#
# The table also carries windows further from the whistle, where the reply
# becomes certain, the bound goes vacuous and the decision genuinely turns on
# estimating the empty-net cost. That is where this case study stops, and it is
# the honest boundary of it.
cat("\n=== The decision in the closing seconds (observed frequencies only) ===\n")

GK_WINDOWS <- data.frame(
  key   = c("fifteen", "thirty", "minute", "tenmin"),
  label = c("under 15 s left", "15--30 s left", "30--60 s left",
            "about 10 min left"),
  lo    = c(3585, 3570, 3540, 2940),
  hi    = c(3600, 3585, 3570, 3060),
  stringsAsFactors = FALSE)

# Computed on the KEEP arm: q_keep and r_keep are what happens playing normally,
# which is what the pull has to beat. `reply` is the share of those possessions
# after which the opponent gets the ball at all.
# WHERE THE WIN-PROBABILITY MODEL ENTERS. The break-even follows from the
# decomposition, and the decomposition values the states the exchange leads to.
# Writing A for what equalizing is worth, V(d+1) - V(d), and B for what falling
# further behind costs, V(d) - V(d-1), both read at the clock the exchange ends
# on, the value of an action is V(a) = V(d) + q_a (1 - r_a) A - (1 - q_a) r_a B,
# and setting V(pull) = V(keep) gives
#
#     dq* = (r_pull - r_keep) * [q A + (1 - q) B] / [(1 - r_pull) A + r_pull B].
#
# Only in the closing seconds does this simplify. There the exchange runs past
# the whistle, so A = 1/2 (the equaliser buys a point where a defeat was worth
# none), B = 0 (a second goal against costs nothing that is not already lost),
# and the break-even reduces to q (r_pull - r_keep) / (1 - r_pull) with no model
# in it at all. Further out both A and B come from the fitted model, and B in
# particular stops being zero---which is why the earlier windows have a much
# higher bar than the arithmetic of the last fifteen seconds would suggest.
GK_LEAD <- -1                    # one goal down throughout

# HOW OFTEN THE EMPTY NET IS ACTUALLY THERE TO BE PUNISHED. The charge "every
# reply becomes a goal because the net is empty" is the worst case only for a
# team that STAYS pulled. It is not a conservative reading for a team that
# brings the keeper back---for those possessions there is no empty net at the
# reply at all, and charging one is simply wrong. That matters because the
# benchmark gk_dq is measured on the tolerant arm, which is exactly the arm that
# includes them: charging the full empty net against a scoring gap measured on
# a population that mostly refilled the goal compares two different tactics.
#
# The obvious refinement is to charge the empty net only where the data say
# there was one: with gk_e the share of replies that met an unguarded goal, the
# bound would read r_pull <= gk_e * reply + (1 - gk_e) * r_keep. WE DO NOT DO
# THIS, and the per-window figures printed below are why. The exposure is not a
# constant of the tactic, it is a function of the clock: 12 per cent ten minutes
# out, 38 per cent in the last minute, and every one of the (two) observed cases
# in the 15--30 s window. The league-wide 11 per cent is essentially the
# mid-match value---of roughly nine thousand pulled set attacks with a reply,
# only a few hundred fall in the closing ten minutes---so applying it to an
# endgame decision would price the risk of a pull taken with ten minutes left
# against a decision taken with seconds left, and would do so in the direction
# that flatters the tactic.
#
# In the window the argument actually rests on it cannot be measured at all:
# a possession ending under 15 s from time almost never draws a reply, so there
# is no exposure to observe. That is not a gap in the data but the same fact the
# bound already uses---no reply, no empty net to punish.
#
# So the full worst case stays. It is also not the inconsistency it would be
# further from the whistle: with seconds left there is no time to send the
# keeper back, which is what the rising per-window exposure says, so the
# tolerant pull arm and a committed pull are nearly the same population there.
gk_e_set <- set_pull |> filter(!is.na(net_empty_next))
gk_e <- mean(gk_e_set$net_empty_next)
cat(sprintf("\n=== Empty-net exposure on the pull arm ===\n"))
cat(sprintf("  replies to pulled set attacks that met an EMPTY net: %.3f (n=%d)\n",
            gk_e, nrow(gk_e_set)))
# Printed per window as a diagnostic: exposure plausibly depends on the clock,
# and if it moves a lot across these the league-wide figure used in the table is
# the wrong summary. The late windows are thin, which is why the table uses the
# league-wide value---the same choice, and for the same reason, as gk_dq.
for (i in seq_len(nrow(GK_WINDOWS))) {
  wi <- gk_e_set |> filter(time_prev >= GK_WINDOWS$lo[i], time_prev < GK_WINDOWS$hi[i])
  cat(sprintf("    %-20s %s\n", GK_WINDOWS$label[i],
              if (nrow(wi) == 0) "no pulled set attacks"
              else sprintf("%.3f (n=%d)", mean(wi$net_empty_next), nrow(wi))))
}

gk_win <- do.call(rbind, lapply(seq_len(nrow(GK_WINDOWS)), function(i) {
  w <- keep_set |> filter(time_prev >= GK_WINDOWS$lo[i],
                          time_prev <  GK_WINDOWS$hi[i],
                          !is.na(opp_scores_next))
  q     <- mean(w$goal)
  reply <- mean(!is.na(w$lead_pos_no))
  r     <- mean(w$opp_scores_next)

  # The clock the exchange ends on: this window's typical start plus its typical
  # duration for our attack and the reply, capped at the whistle.
  t_end <- min(mean(w$time_prev) + median(w$exch_dur, na.rm = TRUE), 3600)
  A <- state_value(neutral, t_end, GK_LEAD + 1) - state_value(neutral, t_end, GK_LEAD)
  B <- state_value(neutral, t_end, GK_LEAD)     - state_value(neutral, t_end, GK_LEAD - 1)

  # Worst case the data permit: the empty net converts every reply that happens.
  # See gk_e above for the refinement this deliberately does NOT apply.
  r_pull <- reply
  be <- if (reply >= 0.95) NA_real_
        else (r_pull - r) * (q * A + (1 - q) * B) / ((1 - r_pull) * A + r_pull * B)
  data.frame(key = GK_WINDOWS$key[i], window = GK_WINDOWS$label[i],
             n = nrow(w), q = q, reply = reply, r = r,
             t_end = t_end, A = A, B = B, be_pp = 100 * be)
}))
print(gk_win[, c("window", "n", "q", "reply", "r", "A", "B", "be_pp")],
      digits = 3, row.names = FALSE)
cat("  (A = what equalizing is worth, B = what a further goal against costs,\n",
    "   both from the win-probability model at the clock the exchange ends on)\n")

# The benchmark the break-even is judged against: the like-for-like descriptive
# gap between pulled and kept SET attacks, league-wide. Not a fitted contrast,
# and not restricted to the endgame---which is the point, since the endgame
# sample is exactly what cannot support one.
gk_dq <- mean(set_pull$goal) - mean(set_keep$goal)
cat(sprintf("\n  like-for-like scoring gap, set attacks league-wide: %+.1f pp (%.3f vs %.3f)\n",
            100 * gk_dq, mean(set_pull$goal), mean(set_keep$goal)))

# SENSITIVITY TO THE KEEPER CODING. The arms above use the ten-second tolerance,
# so a possession played seven against six for all but its closing moments
# counts as pulled. Read strictly---keeper out and seven outfield players for
# the whole possession---the pulled arm loses the attacks in which the keeper
# came back, and those are disproportionately the ones that were already going
# wrong, so the strict arm scores better. Neither coding is the clean one: the
# strict reading selects the pulls that ran smoothly, the tolerant one includes
# the aborted ones. The gap straddles the break-even between them, which is why
# the manuscript states a threshold in the clock rather than a sign at fifteen
# seconds. Reported so the reader can see the spread rather than take the
# tolerant figure on trust.
strict_keep <- daten |> filter(gk_pos == 1, no_field_players_pos == "6",
                               no_field_players_opp == "6", time_pos > 20)
strict_pull <- daten |> filter(gk_pos == 0, no_field_players_pos == "7",
                               no_field_players_opp == "6", time_pos > 20)
gk_dq_strict <- mean(strict_pull$goal) - mean(strict_keep$goal)
cat(sprintf("  same gap under the STRICT keeper coding:            %+.1f pp (%.3f vs %.3f, n=%d)\n",
            100 * gk_dq_strict, mean(strict_pull$goal), mean(strict_keep$goal),
            nrow(strict_pull)))

gk15 <- gk_win[gk_win$key == "fifteen", ]
cat(sprintf("  break-even at 15 s: %+.1f pp  ->  pulling pays if the seventh attacker is worth more than that\n",
            gk15$be_pp))
# The two claims Section 5.1 actually makes, checked against what came out.
# Neither is a claim about the sign at fifteen seconds; the section deliberately
# does not make one.
#
# (1) At thirty seconds and beyond the break-even is far above the scoring gap
#     under EITHER keeper coding, so pulling clearly does not pay there.
gk_dq_max <- max(100 * gk_dq, 100 * gk_dq_strict)
if (any(is.na(gk_win$be_pp[gk_win$key %in% c("thirty", "minute")])) ||
    any(gk_win$be_pp[gk_win$key %in% c("thirty", "minute")] <= gk_dq_max))
  warning("Section 5.1 reports the 15--30 s and 30--60 s break-evens as far ",
          "above the like-for-like gap under either coding. They are not. ",
          "Rewrite the passage.")
# (2) At fifteen seconds the break-even and the measured gap are of the same
#     order---which is what makes the decision close rather than obvious. A
#     factor of two either way is the loosest reading of "same order" the prose
#     can carry.
if (is.na(gk15$be_pp) ||
    gk15$be_pp > 2 * 100 * gk_dq || gk15$be_pp < 0.5 * 100 * gk_dq)
  warning("Section 5.1 reports the 15 s break-even and the like-for-like gap ",
          "as being of the same order. They differ by more than a factor of ",
          "two. Rewrite the passage.")

f1 <- function(x) formatC(x, format = "f", digits = 1)
f3 <- function(x) formatC(x, format = "f", digits = 3)
save_scalar("gkN",         format(gk15$n, big.mark = ","))
save_scalar("gkQ",         f3(gk15$q))
save_scalar("gkR",         f3(gk15$r))
save_scalar("gkReply",     formatC(100 * gk15$reply, format = "f", digits = 0))
save_scalar("gkBreakeven", f1(gk15$be_pp))
save_scalar("gkDeltaQ",    f1(100 * gk_dq))
save_scalar("gkDeltaQStrict", f1(100 * gk_dq_strict))
save_scalar("gkEmptyNet", formatC(100 * gk_e, format = "f", digits = 0))
gk_e_min <- gk_e_set |> filter(time_prev >= 3540, time_prev < 3600)
save_scalar("gkEmptyNetMinute",
            formatC(100 * mean(gk_e_min$net_empty_next), format = "f", digits = 0))
# The two windows one and two steps further from the whistle, so the manuscript
# can quote the rise in the break-even without hard-coding it.
gk30 <- gk_win[gk_win$key == "thirty", ]
gk60 <- gk_win[gk_win$key == "minute", ]
save_scalar("gkReplyThirty",     formatC(100 * gk30$reply, format = "f", digits = 0))
save_scalar("gkBreakevenThirty", f1(gk30$be_pp))
save_scalar("gkBreakevenMinute", f1(gk60$be_pp))
# What equalizing is worth a minute from the whistle. Quoted in the text as the
# reason the break-even rises: away from the buzzer the match continues, so an
# equalizer no longer buys the half a point it buys at the whistle.
save_scalar("gkAMinute", f3(gk60$A))

gk_tex <- gk_win |>
  transmute(Window = window,
            n = format(n, big.mark = ","),
            `$q$` = f3(q),
            `reply` = paste0(formatC(100 * reply, format = "f", digits = 0), "\\%"),
            `$r$` = f3(r),
            `$A$` = f3(A),
            `break-even $\\Delta q$` = ifelse(is.na(be_pp), "--",
                                              paste0(f1(be_pp), " pp")))
save_latex_table(
  gk_tex, "gk_canonical",
  caption = paste0("The goalkeeper decision in the closing seconds, from ",
                   "counted frequencies and the fitted value function. Each ",
                   "row covers possessions ",
                   "that BEGIN in that window at full strength with the keeper ",
                   "in goal: $q$ is how often they score, ``reply'' how often ",
                   "the opponent gets the ball at all afterwards, and $r$ how ",
                   "often the opponent then scores. Because an exchange this ",
                   "late ends at the whistle, the resulting states are settled ",
                   "by their lead and Equation~\\eqref{eq:gk_decomp} collapses ",
                   "to $V(a) = \\tfrac{1}{2} q_a (1 - r_a)$. Charging the pull ",
                   "the worst case the data allow---every reply that happens ",
                   "becomes a goal because the net is empty---gives the ",
                   "break-even in the last column: the smallest gain in $q$ ",
                   "that still makes pulling worthwhile. Charging instead only ",
                   "the share of replies that met an unguarded goal would be ",
                   "milder, but that share is a function of the clock rather ",
                   "than of the tactic---$\\gkEmptyNet$ per cent league-wide ",
                   "against $\\gkEmptyNetMinute$ in the last minute---and it ",
                   "is unmeasurable in the first row, where such possessions ",
                   "almost never draw a reply at all. It is to be compared ",
                   "with the like-for-like gap between pulled and kept set ",
                   "attacks, $\\gkDeltaQ$ percentage points under the ",
                   "ten-second keeper coding used throughout and ",
                   "$\\gkDeltaQStrict$ under the strict one. $A$ is what ",
                   "equalizing is worth at the clock the exchange ends on, read ",
                   "off the win-probability model; it is what the break-even is ",
                   "measured against, and it is $\\tfrac{1}{2}$ exactly only ",
                   "when the exchange runs past the whistle. In the last row ",
                   "the opponent is certain to reply, the bound is vacuous, and ",
                   "the decision turns on estimating the empty-net cost rather ",
                   "than bounding it."),
  label = "tab:gk_canonical",
  align = "lcccccc",
  raw_cols = c("Window", "n", "$q$", "reply", "$r$", "break-even $\\Delta q$"),
  escape_header = FALSE)

# ---- 7C-1b. THE CASE STUDY: level scores, thirty seconds to play -------------
# One named situation, worked through end to end. Two evenly matched teams, the
# score level, our possession about to start with thirty seconds on the clock.
# Pull the keeper for a seventh attacker, or not?
#
# WHY THIS SITUATION AND NOT A DEFICIT. One goal down, a further goal against
# costs nothing that is not already lost, so the downside of the empty net is
# close to zero and pulling is nearly free by construction---only the size of
# the gain is in question. Level, a point is already in hand: scoring turns one
# league point into two, conceding turns it into none. Both sides of the
# decision are worth exactly half a point, which is what makes it a decision.
#
# WHY THE RATES COME FROM ALL POSSESSIONS. Earlier drafts read them off the
# closing seconds alone, which is where the decision is taken but also where
# there are a few dozen pulls to read them from. Measured there the pulled arm
# appeared to concede LESS than the kept one---impossible for an empty net, and
# a sign the contrast was picking up who pulls and how a leading opponent
# responds rather than the keeper. Fitted across all set attacks the sign is the
# one theory requires, and the samples are in the tens of thousands.
GKX_TIME <- 3570                 # our possession begins with thirty seconds left
GKX_LEAD <- 0                    # level
GKX_BWP  <- 0.5                  # two evenly matched teams

# (a) WHAT THE SITUATION IS WORTH BEFORE THE DECISION, from the paper's model.
gkx_base <- neutral
gkx_base$bet_win_prob <- GKX_BWP
# PERSPECTIVE. Under the post-possession convention a row describes the team
# that has just finished attacking and is now about to defend, so state_probs at
# this state returns ITS outcome distribution. We are the team about to attack,
# which is that row's opponent: our win is its loss, and the draw is shared.
gkx_p0 <- state_probs(gkx_base, GKX_TIME, GKX_LEAD)
gkx_v0 <- gkx_p0$loss + VALUE_DRAW_WEIGHT * gkx_p0$draw

# (b) THREE SIMPLE MODELS for what the exchange does. Deliberately small: the
# pull indicator plus the state the possession is played FROM (its start clock,
# the score difference before it, and the matchup strength), so that the
# contrast is adjusted for the fact that it is mostly weaker, trailing teams
# that pull. Not a causal estimate---see the limitations printed below.
set_all <- bind_rows(set_keep |> mutate(pulled = 0L),
                     set_pull |> mutate(pulled = 1L)) |>
  mutate(gls_diff_prev = gls_diff_pos_opp - goal,   # goal is numeric 0/1
         reply         = as.integer(!is.na(lead_pos_no)))

m_goal  <- glm(goal ~ pulled + bet_win_prob + gls_diff_prev + time_prev,
               data = set_all, family = binomial)
# NOT a model. The chance that a reply still follows holds near one until the
# final minute and then collapses inside it, which no logit linear in time can
# follow---fitted over the whole match it put the chance at thirty seconds near
# 0.19, and refitted on the closing five minutes it still did, because the flat
# stretch dominates either way. Both errors ran in the direction that spares the
# pull. It is a counted frequency instead, over set attacks beginning within
# fifteen seconds either side of the decision.
m_conc  <- glm(opp_scores_next ~ pulled + bet_win_prob + gls_diff_prev + time_prev,
               data = set_all |> filter(reply == 1), family = binomial)

nd <- function(a) data.frame(pulled = a, bet_win_prob = GKX_BWP,
                             gls_diff_prev = GKX_LEAD, time_prev = GKX_TIME)
gkx_q      <- sapply(0:1, function(a) predict(m_goal, nd(a), type = "response"))
gkx_r      <- sapply(0:1, function(a) predict(m_conc, nd(a), type = "response"))
gkx_band   <- set_all |> filter(abs(time_prev - GKX_TIME) <= 15)
gkx_preply <- mean(gkx_band$reply)

# (c) THE TREE. Our possession, then the opponent's reply if there is time for
# one. Thirty seconds is one exchange, so every path ends at the whistle and the
# final states are settled by their lead---two points for a win, one for a
# draw, none for a defeat---which is the boundary condition imposed above.
gkx_value <- function(q, r, p_reply) {
  vW <- state_value(gkx_base, 3600,  1)      # 1.0
  vD <- state_value(gkx_base, 3600,  0)      # 0.5
  vL <- state_value(gkx_base, 3600, -1)      # 0.0
  q        * ((1 - p_reply) * vW + p_reply * ((1 - r) * vW + r * vD)) +
  (1 - q)  * ((1 - p_reply) * vD + p_reply * ((1 - r) * vD + r * vL))
}
gkx_v <- sapply(1:2, function(i) gkx_value(gkx_q[i], gkx_r[i], gkx_preply))
gkx_dv <- gkx_v[2] - gkx_v[1]
# The same tree read as outcome probabilities, which is the form a reader who
# thinks in win probability rather than in league points will want.
gkx_pwin <- function(q, r, p) q * ((1 - p) + p * (1 - r))
gkx_w <- sapply(1:2, function(i) gkx_pwin(gkx_q[i], gkx_r[i], gkx_preply))

cat("\n=== CASE STUDY: level, thirty seconds left, evenly matched ===\n")
cat(sprintf("  the situation before the decision:  V = %.3f  (P(win) = %.3f, P(draw) = %.3f)\n",
            gkx_v0, gkx_p0$loss, gkx_p0$draw))
cat(sprintf("  a reply still follows with probability %.2f (counted, n=%d)\n",
            gkx_preply, nrow(gkx_band)))
cat(sprintf("  KEEP:  we score %.3f | they then score %.3f  ->  V = %.4f\n",
            gkx_q[1], gkx_r[1], gkx_v[1]))
cat(sprintf("  PULL:  we score %.3f | they then score %.3f  ->  V = %.4f\n",
            gkx_q[2], gkx_r[2], gkx_v[2]))
cat(sprintf("  DIFFERENCE: %+.4f expected league points  =  %+.1f per 100 decisions\n",
            gkx_dv, 100 * gkx_dv))
cat(sprintf("  in win probability: %.3f keeping vs %.3f pulling (%+.1f pp)\n",
            gkx_w[1], gkx_w[2], 100 * (gkx_w[2] - gkx_w[1])))
cat(sprintf("  fitted pull effects: scoring %+.3f pp | conceding %+.3f pp\n",
            100 * (gkx_q[2] - gkx_q[1]), 100 * (gkx_r[2] - gkx_r[1])))
cat(sprintf("  model n: goal %s | concede %s\n",
            format(nobs(m_goal), big.mark = ","), format(nobs(m_conc), big.mark = ",")))

# DOES THE DECOMPOSITION REPRODUCE THE MODEL? The keep arm is what teams almost
# always do, so its value should come out close to what the win-probability
# model independently says the position is worth. The two are built from
# different things---three small outcome models against the blended GAM---so
# agreement is a real check rather than an identity, and it is the one place
# this case study can be validated at all.
cat(sprintf("  CHECK: keeping is worth %.3f, the model says the position is worth %.3f (gap %+.3f)\n",
            gkx_v[1], gkx_v0, gkx_v[1] - gkx_v0))
gkx_pfit <- uniroot(function(x) gkx_value(gkx_q[1], gkx_r[1], x) - gkx_v0,
                    c(0, 1))$root
cat(sprintf("         (the two agree exactly at a reply rate of %.2f)\n", gkx_pfit))
if (abs(gkx_v[1] - gkx_v0) > 0.05)
  warning("The decomposition values keeping at ", round(gkx_v[1], 3),
          " where the win-probability model values the position at ", round(gkx_v0, 3),
          ". They should agree closely; check the reply probability and the ",
          "perspective of the baseline before quoting either.")

# Removing the goalkeeper cannot lower the rate at which the opponent scores.
if (gkx_r[2] <= gkx_r[1])
  warning("The fitted pull effect on conceding is not positive (", round(gkx_r[2], 3),
          " vs ", round(gkx_r[1], 3), "). An empty net cannot do that, so the ",
          "contrast is confounded and Section 5.1 must not quote the difference.")
# How far the conceding effect would have to move to flip the decision.
gkx_rflip <- uniroot(function(x) gkx_value(gkx_q[2], x, gkx_preply) - gkx_v[1],
                     c(0, 0.999))$root
cat(sprintf("  the decision flips if pulling made them score %.3f instead of %.3f\n",
            gkx_rflip, gkx_r[2]))

f3 <- function(x) formatC(x, format = "f", digits = 3)
save_scalar("gkxVBase",   f3(gkx_v0))
save_scalar("gkxWinBase", f3(gkx_p0$loss))
save_scalar("gkxDrawBase",f3(gkx_p0$draw))
save_scalar("gkxReply",   formatC(100 * gkx_preply, format = "f", digits = 0))
save_scalar("gkxQKeep",   f3(gkx_q[1]))
save_scalar("gkxQPull",   f3(gkx_q[2]))
save_scalar("gkxRKeep",   f3(gkx_r[1]))
save_scalar("gkxRPull",   f3(gkx_r[2]))
save_scalar("gkxVKeep",   f3(gkx_v[1]))
save_scalar("gkxVPull",   f3(gkx_v[2]))
save_scalar("gkxDV",      formatC(100 * gkx_dv, format = "f", digits = 1))
save_scalar("gkxRFlip",   f3(gkx_rflip))
save_scalar("gkxNGoal",   format(nobs(m_goal), big.mark = ","))
save_scalar("gkxNConc",   format(nobs(m_conc), big.mark = ","))
save_scalar("gkxWinKeep", f3(gkx_w[1]))
save_scalar("gkxWinPull", f3(gkx_w[2]))


# ---- 7C-2. The one figure this use case carries ------------------------------
# It shows the WIN-PROBABILITY MODEL, not the auxiliary outcome rates. The rates
# are the immature part of the decomposition (see the manuscript's discussion of
# what a full treatment would need), so a figure resting on them would put the
# weakest link on display. What this shows instead is the conceptual point the
# section turns on and which does not depend on the rates at all: a draw is
# worth a league point, so valuing states by P(win) alone makes the entire
# upside of a late pull invisible. The shaded gap between the two curves IS that
# upside.
value_grid <- tibble::as_tibble(expand.grid(minute = seq(45, 60, by = 0.5),
                                            lead   = c(-2, -1, 0)))
vg <- do.call(rbind, lapply(seq_len(nrow(value_grid)), function(i) {
  p <- state_probs(neutral, min(value_grid$minute[i] * 60, 3600),
                   value_grid$lead[i])
  data.frame(win = p$win, value = p$win + VALUE_DRAW_WEIGHT * p$draw)
}))
lead_lab <- c("-2" = "two goals down", "-1" = "one goal down", "0" = "level")
value_grid <- bind_cols(value_grid, vg) |>
  mutate(panel = factor(lead_lab[as.character(lead)], levels = unname(lead_lab)))

L_V <- "expected points  V(x)"
L_W <- "win probability  P(win | x)"
p_gk_value <- ggplot(value_grid, aes(minute)) +
  geom_ribbon(aes(ymin = win, ymax = value), fill = "#1a9850", alpha = 0.15) +
  geom_line(aes(y = value, colour = L_V), linewidth = 0.9) +
  geom_line(aes(y = win,   colour = L_W), linewidth = 0.9) +
  facet_wrap(~panel) +
  scale_colour_manual(values = setNames(c("#1a9850", "#4575b4"), c(L_V, L_W)),
                      name = NULL) +
  labs(title = "Why the goalkeeper decision has to be valued in expected points",
       subtitle = paste0("Closing minutes, at the strength typical of teams that pull. The shaded gap is the half-point a\n",
                         "draw is worth---much of the upside of a late pull, which a win-probability objective scores as zero."),
       x = "game minute", y = "value of the state") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p_gk_value)
save_plot(p_gk_value, "gk_value_function", width = 7.2, height = 3.4)

cat("\nWhat this case study does and does not claim:\n",
    " - Value: states are valued by expected league points, P(win) + 0.5*P(draw),\n",
    "   NOT by win probability alone. A draw is worth a point, and turning a\n",
    "   certain loss into a draw is much of the reason to pull late; a win-only\n",
    "   value function scores that upside as zero and can never favor pulling.\n",
    " - Scope: the break-even argument works only where the opponent's reply is\n",
    "   rare enough to be bounded by counting. Away from the whistle it is\n",
    "   vacuous, and the decision needs an empty-net cost this data cannot pin\n",
    "   down---roughly seventy late pulls per arm.\n",
    " - Downside: 'the opponent scores next' is not 'scores into an empty net';\n",
    "   teams often re-insert the keeper before the shot, which we cannot see.\n",
    "   The worst-case bound charges the pull for all of it regardless.\n",
    " - Timing: the exchange is assumed to run to the whistle, which is why the\n",
    "   terminal values are exact. That assumption is what confines the argument\n",
    "   to the closing seconds; a full treatment needs the state-space model of\n",
    "   the companion paper.\n")


cat("\nDone. Trajectory, goal-value, and goalkeeper plots printed; tables above.\n")
