library(dplyr)
library(here)
# Read in Data ------------------------------------------------------------

pre_process_data <- function(daten) {
  # daten = daten[1:170385,]
  # Calc pre-game win probs
  daten <- daten |>
    mutate(final = ifelse(winner == team, 1,
      ifelse(winner == opp, -1, 0)
    )) |>
    group_by(matchid) |>
    mutate(gls_pos = as.numeric(gls_pos)) |>
    mutate(gls_opp = as.numeric(gls_opp)) |>
    mutate(bet_win_prob = as.numeric(bet_win_prob)) |>
    mutate(bet_loss_prob = as.numeric(bet_loss_prob)) |>
    mutate(bet_draw_prob = as.numeric(bet_draw_prob)) |>
    mutate(goal = as.numeric(goal)) |>
    mutate(gls_pos_prev = gls_pos - goal) |>
    mutate(gls_opp_prev = gls_opp) |>
    mutate(gls_diff_pos_opp_prev = gls_pos_prev - gls_opp_prev) |>
    mutate(gls_diff_opp_pos_prev = gls_opp_prev - gls_pos_prev) |>
    mutate(gls_diff_pos_opp = gls_pos - gls_opp) |>
    mutate(gls_diff_opp_pos = gls_opp - gls_pos) |>
    # EVERY column of a row describes the state AFTER that possession, and the
    # pipeline keeps it that way rather than reconstructing a "before".
    #
    # The ticker records a possession by its terminating event: the clock is the
    # stamp of the shot or the turnover, the score already contains the goal if
    # one was scored, and the on-court player counts already reflect a suspension
    # awarded during the attack---a two-minute penalty conceded on the way to
    # the shot is in the data as 6 against 5, not as the 6 against 6 the attack
    # was played at. Three of those quantities used to be pushed back to the
    # start of the possession (the score by subtracting the goal, the momentum
    # statistics by a one-possession shift, the clock by subtracting the
    # possession's duration) while the player counts, which cannot be pushed back
    # at all, were left as recorded. The state was a mixture of two instants.
    #
    # Reading every row as the state AFTER its possession removes the mixture at
    # no cost, because the state after possession i is exactly the state from
    # which possession i+1 is played: the same sequence of states, labeled by
    # the possession that produced them instead of the one that follows. It also
    # brings the last action of a match into the data. With pre-possession states
    # the final row sits before the deciding attack and nothing follows it, so a
    # win-probability path never reaches its terminal value and a match-winning
    # goal in the closing seconds produces no swing at all.
    #
    # THE 10 s VARIANTS ARE THE ONES THE ANALYSIS READS. gk_pos and
    # no_field_players_pos are the court exactly at the terminating event, and a
    # team that pulled its keeper usually restores them in the closing seconds of
    # the attack---so read strictly, an attack played seven against six for all
    # but its last moments is recorded as even strength. gk_pos_10s and
    # no_field_players_pos_10s apply a ten-second tolerance and are what the
    # manuscript's data section describes. The difference is not marginal: 9,518
    # seven-against-six attacks against 6,977 read strictly, and 22,356
    # possessions with the keeper out against 18,517.
    #
    # The *_prev columns survive because the auxiliary outcome-rate models of the
    # goalkeeper use case predict the outcome OF a possession and therefore have
    # to condition on the state that possession started from---which is the
    # post-state of the row before it. Everything those models need is
    # reconstructible exactly: the opponent cannot score while we attack, so
    # gls_opp is the same at both ends of the possession, and time_pos is the
    # recorded duration. The quantities that are not reconstructible---the
    # player counts and the goalkeeper configuration---are precisely the ones
    # those models want at their in-attack value, since that is the decision.
    mutate(time = 60 * as.numeric(time_min) + as.numeric(time_sec)) |>
    mutate(time_prev = time - as.numeric(time_pos)) |>
    # Drop the ticker's boundary markers. Roughly one row per match sits at
    # exactly 30:00 and one at exactly 60:00 carrying no goal, no missed shot,
    # no turnover and zero duration---they record the half-time and full-time
    # whistle, not an attacking sequence. The separation is exact: 3,211 such
    # rows against 167,274 that have both a positive duration and an event.
    #
    # They have to go for three reasons. They are not decision points, so the
    # goalkeeper analysis should not see them; they carry the final score at
    # t = 3600, which makes them trivially predictable and flatters every
    # accuracy and calibration figure computed over them; and they duplicate the
    # state of the possession they follow, so the last genuine attack of a half
    # is immediately shadowed by a twin row that adds no information and takes
    # the closing win-probability swing away from the possession that earned it.
    filter(!(as.numeric(time_pos) == 0 &
             as.numeric(goal) == 0 &
             as.numeric(missed_shot) == 0 &
             as.numeric(turnover) == 0)) |>
    mutate(home = ifelse(team == team_home, 1, 0)) |>
    mutate(HZ = as.factor(HZ)) |>
    mutate(
      gk_pos               = as.factor(gk_pos),
      gk_pos_prev          = as.factor(gk_pos_prev),
      no_players_pos       = as.factor(no_players_pos),
      no_players_opp       = as.factor(no_players_opp),
      no_field_players_pos = as.factor(no_field_players_pos),
      no_field_players_opp = as.factor(no_field_players_opp),
      goal_prev            = as.factor(goal_prev),
      goal_next            = as.factor(goal_next),
      gk_pos_10s               = as.factor(gk_pos_10s),
      gk_pos_10s_prev          = as.factor(gk_pos_10s_prev),
      no_field_players_pos_10s = as.factor(no_field_players_pos_10s),
      final                = as.factor(final),
      home                 = as.factor(home),
      team                 = as.factor(team),
      opp                  = as.factor(opp)
    ) |>
    filter(!is.na(opp))

  # Schritt 1: letzte Zeile pro Spiel extrahieren (Endstand)
  final_scores <- daten %>%
    group_by(matchid) %>%
    slice_tail(n = 1) %>% # letzte Zeile des Spiels
    dplyr::select(
      matchid, team, opp,
      gls_pos, gls_opp
    ) %>%
    rename(
      final_team = team,
      final_opp = opp,
      final_gls_pos = gls_pos,
      final_gls_opp = gls_opp
    )

  # Schritt 2: Merge zurück auf den Hauptdatensatz
  daten <- daten %>%
    left_join(final_scores, by = "matchid") %>%
    mutate(
      # Wenn das angreifende Team das Heimteam ist:
      final_goals_team = ifelse(team == final_team,
        final_gls_pos,
        final_gls_opp
      ),
      final_goals_opp = ifelse(opp == final_team,
        final_gls_pos,
        final_gls_opp
      ),
      # Goals still to come FROM the state this row records, i.e. after the
      # possession has been played out.
      rem_goals_pos = final_goals_team - gls_pos,
      rem_goals_opp = final_goals_opp - gls_opp
    ) |>
    dplyr::select(
      -final_team, -final_opp,
      -final_gls_pos, -final_gls_opp
    )

  daten[which(is.na(daten$gk_pos_prev)), "gk_pos_prev"] <- as.factor(1)
  daten[which(is.na(daten$goal_prev)), "goal_prev"] <- as.factor(0)
  daten[which(is.na(daten$gk_pos_10s_prev)), "gk_pos_10s_prev"] <- as.factor(1)
  daten[which(is.na(daten$goal_next)), "goal_next"] <- as.factor(0)

  daten$gls_diff_pos_opp      <- as.numeric(daten$gls_diff_pos_opp)
  daten$gls_diff_opp_pos      <- as.numeric(daten$gls_diff_opp_pos)
  daten$gls_diff_pos_opp_prev <- as.numeric(daten$gls_diff_pos_opp_prev)
  daten$gls_diff_opp_pos_prev <- as.numeric(daten$gls_diff_opp_pos_prev)
  daten$final_ord <- factor(daten$final, levels = c(-1, 0, 1))

  return(daten)
}

# write.csv(daten, here("raw_data", "preprocessed_data.csv"), row.names = FALSE)
