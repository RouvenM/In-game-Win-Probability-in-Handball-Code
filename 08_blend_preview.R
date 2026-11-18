# 08_blend_preview.R -----------------------------------------------------------
# Writes generated/blend_preview.tex: the best configuration of each segment
# count, as a short stand-in for the full tuning grid in the appendix.
#
# It reads generated/blend_tuning.tex, which 02_tune_blend_parameters.R writes,
# rather than recomputing anything. That table is already sorted by the
# selection criterion -- the worst-phase calibration error among competitive
# states -- so the best configuration of a given segment count is simply the
# first row carrying that count. Deriving the preview from the table the
# appendix prints is also what keeps the two from ever disagreeing.
#
# Separated from 02 for the same reason 06 is separated from 01: the numbers
# cost a full tuning run, the presentation of them costs nothing, and one
# should not have to pay for the first to change the second.
# -----------------------------------------------------------------------------

rm(list = ls())
library(here)
library(dplyr)

source(here("functions", "output_helpers.R"))

tex <- file.path(OUTPUT_DIR, "blend_tuning.tex")
if (!file.exists(tex))
  stop("generated/blend_tuning.tex is missing. Run 02_tune_blend_parameters.R ",
       "first; it writes the grid this preview is drawn from.")
ln <- readLines(tex, warn = FALSE)

# Data rows end in \\ and are neither the header nor a rule.
body <- ln[grepl("\\\\\\\\\\s*$", ln) &
           !grepl("^Configuration|hline|multicolumn|cline", ln)]
split_row <- function(l) {
  cells <- trimws(strsplit(sub("\\s*\\\\\\\\\\s*$", "", l), "&", fixed = TRUE)[[1]])
  stopifnot(length(cells) == 6)
  cells
}
rows <- lapply(body, split_row)
cat(sprintf("read %d configurations from %s\n", length(rows), basename(tex)))
stopifnot(length(rows) >= 4)

grid <- data.frame(
  Configuration = vapply(rows, `[`, "", 1),
  `Seg.`        = vapply(rows, `[`, "", 2),
  RPS           = vapply(rows, `[`, "", 3),
  ECE           = vapply(rows, `[`, "", 4),
  `max ECE (all)`   = vapply(rows, `[`, "", 5),
  `max ECE (comp.)` = vapply(rows, `[`, "", 6),
  check.names = FALSE, stringsAsFactors = FALSE
)
# Strip the bold off the segment count only; the cells keep whatever emphasis
# the full table gave them, so the selected row stays marked here too.
seg_n <- as.integer(gsub("[^0-9]", "", grid$`Seg.`))
stopifnot(!anyNA(seg_n))

# First row per segment count = that count's best, the table being sorted.
keep <- !duplicated(seg_n)
preview <- grid[keep, ]
cat("best of each segment count, in the order the grid ranks them:\n")
for (i in seq_len(nrow(preview)))
  cat(sprintf("  %-22s seg %s   worst competitive phase %s\n",
              gsub("\\\\textbf\\{|\\}", "", preview$Configuration[i]),
              gsub("[^0-9]", "", preview$`Seg.`[i]),
              gsub("\\\\textbf\\{|\\}", "", preview$`max ECE (comp.)`[i])))

save_latex_table(
  preview, "blend_preview",
  caption = paste0("The best configuration of each segment count, taken from ",
                   "the full grid of Table~\\ref{tab:blend_tuning} in ",
                   "\\ref{app:blend_tuning} and ordered as that grid orders ",
                   "them, by the worst-phase calibration error among ",
                   "competitive states. Bold marks the selected ",
                   "configuration. The columns are those of the full grid: ",
                   "RPS and ECE are overall, on the pooled out-of-fold ",
                   "predictions, and the two max-ECE columns are the worst ",
                   "five-minute game-time bin, over all states and over ",
                   "competitive states only."),
  label = "tab:blend_preview",
  align = "lccccc",
  raw_cols = names(preview))
