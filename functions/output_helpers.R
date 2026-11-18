# output_helpers.R ------------------------------------------------------------
# Save plots and tables into paper/generated/ so the manuscript can \input the
# tables and \includegraphics the figures. Run all R scripts once, then compile
# paper/manuscript.tex---no numbers or figures are copied by hand.
#
#   save_plot(p, "name")            -> paper/generated/name.pdf
#   save_latex_table(df, "name",    -> paper/generated/name.tex   (a tabular,
#                    caption, label, ...)                          \input-ready)
#   save_scalar("name", value)      -> paper/generated/name.tex   (a \newcommand
#                                                                   or raw value)
# -----------------------------------------------------------------------------
library(here)

OUTPUT_DIR <- here("paper", "generated")
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Save a ggplot (or any grid object) as a vector PDF.
save_plot <- function(plot, name, width = 6.2, height = 3.6) {
  path <- file.path(OUTPUT_DIR, paste0(name, ".pdf"))
  ggplot2::ggsave(path, plot, width = width, height = height, units = "in")
  cat(sprintf("  saved figure -> %s\n", path))
  invisible(path)
}

# Escape the handful of LaTeX-special characters that occur in our labels.
.latex_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  # NOTE the replacement here is a LITERAL string, because fixed = TRUE turns off
  # backslash processing in `replacement`. It must therefore contain exactly the
  # one backslash we want in the output: paste0("\\", ch) is the two-character
  # string \& , \# , ... Using paste0("\\\\", ch) would emit a doubled backslash
  # (\\#), which LaTeX reads as a line break followed by a raw #---a fatal
  # "You can't use macro parameter character #" error.
  for (ch in c("&", "%", "$", "#", "_", "{", "}")) {
    x <- gsub(ch, paste0("\\", ch), x, fixed = TRUE)
  }
  x <- gsub("~", "\\\\textasciitilde{}", x)
  x
}

# Format a data frame as a booktabs-free \hline tabular (matching the
# manuscript's existing style) and write it to a .tex file the manuscript can
# \input. `align` defaults to left for the first column, center for the rest.
# `digits` may be a single value or a per-numeric-column vector; character
# columns are escaped, numeric columns are formatted and NOT escaped so math
# stays intact. Set `math_cols` to wrap those columns' cells in $...$. Columns
# named in `raw_cols` are passed through verbatim---neither escaped nor
# reformatted---so pre-built LaTeX (e.g. \textbf{...} on the best cell) survives.
save_latex_table <- function(df, name, caption = NULL, label = NULL,
                             align = NULL, digits = 3, header = names(df),
                             math_cols = character(0), raw_cols = character(0),
                             escape_header = TRUE, prelude = NULL,
                             small = TRUE, size = NULL, placement = "!htbp") {
  df <- as.data.frame(df)
  ncol_ <- ncol(df)
  if (is.null(align)) align <- paste0("l", strrep("c", ncol_ - 1))
  if (length(digits) == 1) digits <- rep(digits, ncol_)

  fmt_cell <- function(v, j) {
    if (names(df)[j] %in% raw_cols) return(as.character(v))
    out <- if (is.numeric(v)) formatC(v, format = "f", digits = digits[j]) else .latex_escape(v)
    if (names(df)[j] %in% math_cols) out <- paste0("$", out, "$")
    out
  }

  rows <- character(nrow(df))
  for (i in seq_len(nrow(df))) {
    cells <- vapply(seq_len(ncol_), function(j) fmt_cell(df[[j]][i], j), character(1))
    rows[i] <- paste(cells, collapse = " & ")
  }

  # raw_cols exempts CELLS from escaping but not the header, so a column named
  # "$q$ keep" would arrive as "\$q\$ keep". escape_header = FALSE passes the
  # header through verbatim for tables whose column names are themselves LaTeX.
  head_line <- paste(if (escape_header) .latex_escape(header) else header,
                     collapse = " & ")
  lines <- c(
    sprintf("\\begin{table}[%s]", placement),
    "\\centering",
    if (!is.null(size)) paste0("\\", size) else if (small) "\\small" else NULL,
    # Verbatim LaTeX between the size and the caption, for the odd table that
    # needs a column-separation tweak to fit the text block.
    prelude,
    if (!is.null(caption)) sprintf("\\caption{%s}", caption) else NULL,
    if (!is.null(label))   sprintf("\\label{%s}", label)     else NULL,
    sprintf("\\begin{tabular}{%s}", align),
    "\\hline",
    paste0(head_line, " \\\\"),
    "\\hline",
    paste0(rows, " \\\\"),
    "\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )
  path <- file.path(OUTPUT_DIR, paste0(name, ".tex"))
  writeLines(lines, path)
  cat(sprintf("  saved table  -> %s\n", path))
  invisible(path)
}

# Save a single number/string as a reusable LaTeX macro:
#   save_scalar("gamRPS", 0.118)  ->  \newcommand{\gamRPS}{0.118}
# Use in the manuscript after \input{generated/scalars.tex} as \gamRPS.
#
# Idempotent by design: before appending, any existing definition of the SAME
# macro is removed. Re-running a script, or running 01/02/04 in any order into
# the shared scalars.tex, therefore never writes a duplicate \newcommand---a
# duplicate would be a fatal LaTeX "command already defined" error.
save_scalar <- function(name, value, file = "scalars") {
  # LaTeX rejects some macro names outright, and the failure surfaces only at
  # compile time as a confusing "already defined" error. \newcommand runs the
  # name through \@ifdefinable, which refuses anything whose first three letters
  # are "end"---the guard that stops you redefining an environment's closing
  # command---and likewise refuses "relax". Letters only, too: digits and
  # underscores are not valid in a control sequence built this way. Catching it
  # here points at the offending save_scalar call instead of at line 8 of a
  # generated file.
  if (!grepl("^[A-Za-z]+$", name))
    stop("save_scalar: '", name, "' must consist of letters only; ",
         "LaTeX control sequences cannot contain digits or underscores.")
  if (grepl("^end", name) || name == "relax")
    stop("save_scalar: '", name, "' is not a definable LaTeX macro name. ",
         "\\newcommand refuses names beginning with \"end\" (and \\relax). ",
         "Rename it, e.g. '", sub("^end(game)?", "late", name), "'.")

  # A thousands separator has to be braced. Scalars are quoted in math mode as
  # often as not, and there a bare comma is punctuation: $93,679$ sets as
  # "93, 679" with a gap after the comma. Braced, {,} is an ordinary symbol and
  # sets tight. Only commas BETWEEN DIGITS are touched, so text-valued scalars
  # such as a club or season name keep their prose commas.
  value <- gsub("(?<=[0-9]),(?=[0-9])", "{,}", as.character(value), perl = TRUE)

  path <- file.path(OUTPUT_DIR, paste0(file, ".tex"))
  line <- sprintf("\\newcommand{\\%s}{%s}", name, value)
  existing <- if (file.exists(path)) readLines(path, warn = FALSE) else character(0)
  # Drop a prior definition of exactly this macro (trailing \} anchors the name
  # so a shorter name cannot match a longer one, e.g. blendSelected vs
  # blendSelectedRPS), plus any blank lines left by earlier writers.
  pat <- sprintf("\\\\newcommand\\{\\\\%s\\}", name)
  existing <- existing[nzchar(trimws(existing)) & !grepl(pat, existing)]
  writeLines(c(existing, line), path)
  invisible(path)
}
