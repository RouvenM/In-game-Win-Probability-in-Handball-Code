# blend_functions.R -----------------------------------------------------------
# Shared K-segment blend machinery, used by
#   01_model_comparison.R      (the GAM entries in the model comparison),
#   02_tune_blend_parameters.R (the low-level weight functions for the grid),
#   04_use_cases.R             (the deployed model behind the applications).
#
# A K-segment blend fits K copies of a model on overlapping game-time windows
# and combines their predictions with a smooth partition of unity built from
# cosine ramps (each weight has zero derivative at its ends, so the blended
# curve has no kink where two segments meet). See the "blend" section of the
# manuscript.
#
# Requires realign_factor_levels() from general_functions.R (resolved at call
# time, so source order does not matter as long as both are sourced).
# -----------------------------------------------------------------------------

# Deployed configuration, selected by 02_tune_blend_parameters.R: three segments
# with break points at 50 and 57 minutes, and a ramp half-width of 5 minutes at
# the first and 1 minute at the second. 01 and 04 use these defaults; 02
# overrides them while searching the grid.
#
# The second ramp has to be narrow, and not as a tuning nicety: a 5-minute ramp
# centered at 57 would still be running when the match ends, so its segment could
# never reach full weight and could not own the closing seconds. Those seconds
# are the point of it---the last segment spans three minutes rather than ten,
# so its time smooth resolves roughly one basis function per 18 seconds instead
# of one per minute, where the result stops being uncertain.
BLEND_HALF_WIDTH  <- c(300, 60)        # seconds (ramp half-width per break point)
BLEND_BREAKPOINTS <- c(50, 57) * 60    # seconds (K-1 transition centres)

# Smooth 0 -> 1 S-ramp centered at `center`, reaching 0 at center-hw and 1 at
# center+hw, with zero derivative at both ends.
ramp_up <- function(t, center, hw) {
  lo <- center - hw; hi <- center + hw
  ifelse(t <= lo, 0,
         ifelse(t >= hi, 1,
                0.5 * (1 - cos(pi * (t - lo) / (2 * hw)))))
}

# THE RAMP HALF-WIDTH MAY DIFFER PER BREAK POINT. `hw` is recycled to the length
# of `bp`, so a scalar keeps the original behavior, while a vector lets a late
# transition be narrow where an early one is wide. That is not a refinement for
# its own sake: with a 5-minute ramp the latest usable break point is 55 min,
# since its ramp then completes exactly at the final whistle. Anything later
# never reaches full weight before the match ends, so its segment cannot own the
# closing seconds---which is precisely the stretch a late segment would be for.
# Isolating the last minutes therefore requires a narrower ramp, not just a
# later center. `blend_is_valid()` enforces both conditions.
.recycle_hw <- function(bp, hw) rep_len(hw, length(bp))

# Segment support bounds: segment k is fit on rows with time in [lo[k], hi[k]].
seg_bounds <- function(bp, hw) {
  hw <- .recycle_hw(bp, hw)
  list(lo = c(-Inf, bp - hw), hi = c(bp + hw, Inf), K = length(bp) + 1L)
}

# Is this a usable partition of unity over [0, t_end]?
#   - break points strictly increasing;
#   - ramps disjoint, so every weight stays non-negative and each segment owns a
#     contiguous stretch of game time;
#   - the first ramp starts at or after 0 and the last one COMPLETES at or before
#     t_end, so the final segment actually reaches weight 1 inside the match.
blend_is_valid <- function(bp, hw, t_end = 3600) {
  if (length(bp) == 0L) return(TRUE)
  hw <- .recycle_hw(bp, hw)
  if (any(hw <= 0)) return(FALSE)
  if (length(bp) > 1L && any(diff(bp) <= 0)) return(FALSE)
  if (bp[1] - hw[1] < 0) return(FALSE)
  if (bp[length(bp)] + hw[length(hw)] > t_end) return(FALSE)
  if (length(bp) > 1L &&
      any(bp[-length(bp)] + hw[-length(hw)] > bp[-1] - hw[-1])) return(FALSE)
  TRUE
}

# Partition-of-unity segment weights: an n x K matrix whose rows sum to 1. With
# non-overlapping ramps (see blend_is_valid) every weight is non-negative and
# each segment owns a contiguous stretch of game time.
segment_weights <- function(time, bp, hw) {
  K <- length(bp) + 1L
  if (K == 1L) return(matrix(1, nrow = length(time), ncol = 1L))
  hw <- .recycle_hw(bp, hw)
  U <- matrix(sapply(seq_along(bp), function(k) ramp_up(time, bp[k], hw[k])),
              nrow = length(time))
  W <- matrix(0, nrow = length(time), ncol = K)
  W[, 1] <- 1 - U[, 1]
  if (K >= 3L) for (k in 2:(K - 1L)) W[, k] <- U[, k - 1L] - U[, k]
  W[, K] <- U[, K - 1L]
  W
}

# Fit one model per segment. `fit_one(data)` fits a single model on a data
# subset; each subset is the rows falling in that segment's time window, with
# factor levels realigned to the full training frame.
fit_kblend <- function(train, fit_one, bp = BLEND_BREAKPOINTS, hw = BLEND_HALF_WIDTH) {
  sb <- seg_bounds(bp, hw)
  fits <- lapply(seq_len(sb$K), function(k) {
    keep <- !is.na(train$time) & train$time >= sb$lo[k] & train$time <= sb$hi[k]
    fit_one(realign_factor_levels(train[keep, , drop = FALSE], train))
  })
  list(fits = fits, bp = bp, hw = hw)
}

# Predict: the smooth convex combination of the K segment models' predictions.
# `predict_one(fit, newdata)` returns an n x 3 matrix of class probabilities;
# the caller sets the column names.
predict_kblend <- function(blend, newdata, predict_one) {
  n   <- nrow(newdata)
  W   <- segment_weights(newdata$time, blend$bp, blend$hw)
  acc <- matrix(0, nrow = n, ncol = 3L)
  for (k in seq_along(blend$fits)) {
    # Coerce to a fixed n x 3 matrix so a single-row prediction that a predict
    # function returned as a length-3 vector cannot break the accumulation.
    pk  <- matrix(as.numeric(as.matrix(predict_one(blend$fits[[k]], newdata))),
                  nrow = n, ncol = 3L)
    acc <- acc + W[, k] * pk
  }
  acc
}
