# 06_rps_by_time.R -------------------------------------------------------------
# Draws generated/cv_rps_by_time.pdf: mean ranked probability score by five-
# minute game-time bin, one line per model.
#
# It reads generated/rps_by_time.csv, which 01_model_comparison.R writes. The
# numbers need every model's out-of-fold predictions and so cost a full run of
# that script; the figure is separated from them because its appearance is the
# part one wants to change, and doing that should not cost hours.
#
# Averaged over a whole match the advantage of the in-game models is diluted.
# This view shows where it is earned: the pre-match baseline cannot react and
# stays flat, while the fitted models fall away as the outcome settles.
# -----------------------------------------------------------------------------

rm(list = ls())
library(here)
library(dplyr)
library(ggplot2)

source(here("functions", "output_helpers.R"))

csv <- file.path(OUTPUT_DIR, "rps_by_time.csv")
if (!file.exists(csv))
  stop("generated/rps_by_time.csv is missing. Run 01_model_comparison.R first; ",
       "it writes the table this figure is drawn from.")
rps_by_time <- read.csv(csv, stringsAsFactors = FALSE)

# Draw the match-side weighted mean, not the possession-weighted one. A match
# contributes a different number of possessions to each window, so a
# possession-weighted curve moves with the changing mix of matches as well as
# with the models -- most visibly on the pre-match baseline, whose forecast is
# constant within a match side and whose curve therefore has to be flat.
# functions/comparison_table.R sets out both weightings. Older CSVs carry only
# the possession-weighted column.
if ("rps_eq" %in% names(rps_by_time)) {
  rps_by_time$rps <- rps_by_time$rps_eq
} else {
  warning("rps_by_time.csv predates the match-side weighting and carries only ",
          "the possession-weighted column, so the reference lines will not be ",
          "exactly flat. Rerun 01_model_comparison.R, or rebuild the file with ",
          "07_rebuild_comparison_table.R from a finished session.")
}

# Short labels, because they sit at the end of each line rather than in a
# legend, and colours that let one model carry the eye. The selected model is
# the only saturated one; the two references are grey and red.
#
# The four tree ensembles share a label. They are drawn individually, but over
# the whole match they lie within a thousandth of one another, and six separate
# labels on a band that thin would suggest a ranking the figure cannot support
# -- Table "model_comparison" is where the ensembles are told apart. What this
# figure is about is the gap between the three kinds of forecast: one that
# knows the teams and not the match, one that knows the match and not the
# teams, and the fitted models that know both.
style <- tibble::tribble(
  ~model,               ~label,           ~colour,   ~width,
  "gam",                "GAM",            "#E8720C", 1.15,
  "baseline_xgb_multi", "Tree ensembles", "#7F9BBF", 0.5,
  "baseline_xgb",       "Tree ensembles", "#7F9BBF", 0.5,
  "baseline_rf",        "Tree ensembles", "#7F9BBF", 0.5,
  "baseline_rf_ord",    "Tree ensembles", "#7F9BBF", 0.5,
  "baseline",           "Pre-match",      "#C44E52", 0.7,
  "odds_leader",        "Odds + lead",    "#8172B2", 0.7,
  "leader",             "Leading team",   "#9A9A9A", 0.7
)
d <- rps_by_time |> inner_join(style, by = "model")
missing <- setdiff(unique(rps_by_time$model), style$model)
if (length(missing))
  warning("no style defined for: ", paste(missing, collapse = ", "),
          " -- these models are left out of the figure.")

# Labels go at the right-hand end of each line. Where two lines finish close
# together the text would overlap, so the labels are pushed apart from the top
# down by a minimum gap. Only the label moves; the line does not, and a label
# that has been moved is joined to its line by a short leader.
#
# The gap has to be in the units of the y axis, and large enough for a line of
# text: at this figure height the panel spans the data range over about three
# and a half inches, so a twentieth of the range clears the type comfortably.
# Pushing down alone is not enough: the best models finish bunched just above
# zero, so a stack that only grows downwards runs off the bottom of the panel
# and the last label disappears. The second pass catches that and restacks from
# the floor upwards.
spread <- function(y, gap, lo) {
  o <- order(y, decreasing = TRUE)
  v <- y[o]
  for (i in seq_along(v)[-1]) v[i] <- min(v[i], v[i - 1] - gap)
  if (v[length(v)] < lo) {
    v[length(v)] <- lo
    for (i in rev(seq_along(v))[-1]) v[i] <- max(v[i], v[i + 1] + gap)
  }
  out <- numeric(length(y)); out[o] <- v; out
}
x_end <- max(d$time_bin)
y_top <- max(d$rps)
ends <- d |>
  filter(time_bin == x_end) |>
  # one label per group; the ensembles' leader points at the middle of their band
  group_by(label, colour) |>
  summarize(rps = mean(rps), .groups = "drop") |>
  arrange(desc(rps)) |>
  mutate(y_lab = spread(rps, gap = y_top / 22, lo = y_top / 40))

pal <- setNames(style$colour, style$label)
p <- ggplot(d, aes(time_bin, rps, colour = label)) +
  # grouped by model, not by label, so the four ensembles stay four lines
  geom_line(aes(linewidth = I(width), group = model)) +
  geom_segment(data = ends, aes(x = x_end, xend = x_end + 1.6,
                                y = rps, yend = y_lab),
               linewidth = 0.25, show.legend = FALSE) +
  geom_text(data = ends, aes(x = x_end + 2.2, y = y_lab, label = label),
            hjust = 0, size = 3.1, show.legend = FALSE) +
  scale_colour_manual(values = pal, guide = "none") +
  scale_x_continuous(limits = c(0, x_end + 17), breaks = seq(0, x_end, 10),
                     expand = expansion(mult = c(0.01, 0))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.04))) +
  labs(x = "Game time [min]", y = "RPS") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.major = element_line(colour = "grey88", linetype = "dashed",
                                    linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(angle = 0, hjust = 0, vjust = 1.02,
                                face = "italic", colour = "grey35"),
    axis.title.x = element_text(hjust = 1, face = "italic", colour = "grey35"),
    axis.text    = element_text(colour = "grey35"),
    axis.ticks   = element_blank(),
    # ggplot2:: because randomForest, attached by functions/general_functions.R,
    # exports a margin() generic of its own. Run in the same session as 01 it
    # sits ahead of ggplot2 on the search path -- library(ggplot2) here does
    # not move it, since the package is already attached -- and the call fails
    # with "6 is not a factor".
    plot.margin  = ggplot2::margin(10, 6, 6, 6)
  )

# Zoom on the last ten minutes. At full scale the GAM and the ensembles are a
# single band, and the panel above the curves is empty once the scoreboard
# baseline has dropped away. The inset enlarges the last two five-minute bins
# for the fitted models alone, each ensemble under its own name, so that the
# GAM can be seen to lie below every ensemble there; the stretch is marked on
# the main panel. Random forests are dashed, XGBoost solid.
ZOOM_X <- c(max(d$time_bin) - 5, max(d$time_bin))
zoom_names <- c(gam                = "GAM",
                baseline_xgb_multi = "XGBoost (multiclass)",
                baseline_xgb       = "XGBoost (ordinal)",
                baseline_rf        = "Random forest (multiclass)",
                baseline_rf_ord    = "Random forest (ordinal)")
zoom_d <- d |>
  filter(time_bin >= ZOOM_X[1], model %in% names(zoom_names)) |>
  mutate(zlabel = zoom_names[model])
zoom_y <- range(zoom_d$rps) + c(-1, 1) * 0.04 * diff(range(zoom_d$rps))
zoom_ends <- zoom_d |>
  filter(time_bin == ZOOM_X[2]) |>
  mutate(y_lab = spread(rps, gap = diff(zoom_y) / 11, lo = zoom_y[1]))

inset <- ggplot(zoom_d, aes(time_bin, rps, colour = label)) +
  geom_line(aes(linewidth = I(width * 0.9), group = model,
                linetype = grepl("_rf", model))) +
  geom_segment(data = zoom_ends,
               aes(x = ZOOM_X[2], xend = ZOOM_X[2] + 0.25, y = rps, yend = y_lab),
               linewidth = 0.2, show.legend = FALSE) +
  geom_text(data = zoom_ends,
            aes(x = ZOOM_X[2] + 0.35, y = y_lab, label = zlabel),
            hjust = 0, size = 2.3, show.legend = FALSE) +
  scale_colour_manual(values = pal, guide = "none") +
  scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "22"), guide = "none") +
  scale_x_continuous(breaks = ZOOM_X,
                     labels = paste0(ZOOM_X, "-", ZOOM_X + 5),
                     limits = c(ZOOM_X[1], ZOOM_X[2] + 5.4),
                     expand = expansion(mult = c(0.03, 0))) +
  coord_cartesian(ylim = zoom_y) +
  labs(x = NULL, y = NULL, title = "Last ten minutes") +
  theme_minimal(base_size = 7.5) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90", linewidth = 0.25),
        panel.background = element_rect(fill = "white", colour = "grey60",
                                        linewidth = 0.3),
        plot.background  = element_rect(fill = "white", colour = NA),
        plot.title = element_text(size = 7.5, colour = "grey35"),
        axis.text = element_text(colour = "grey35"),
        plot.margin = ggplot2::margin(2, 2, 2, 2))

p <- p +
  annotate("rect", xmin = ZOOM_X[1], xmax = ZOOM_X[2],
           ymin = zoom_y[1], ymax = zoom_y[2],
           fill = NA, colour = "grey45", linewidth = 0.3, linetype = "dashed") +
  annotation_custom(ggplotGrob(inset),
                    xmin = 31, xmax = x_end + 17,
                    ymin = y_top * 0.59, ymax = y_top * 1.03)

print(p)
save_plot(p, "cv_rps_by_time", width = 5.6, height = 4.2)
cat("\nFinal-bin values (where the labels sit):\n")
print(as.data.frame(ends |> transmute(label, rps = round(rps, 4))), row.names = FALSE)
