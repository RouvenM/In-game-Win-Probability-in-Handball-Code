library(arrow)
library(here)
library(purrr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(MLmetrics)
library(lubridate)
library(magrittr)
library(nnet)
library(data.table)
library(caret)
library(randomForest)
library(ranger)
library(rpart)
library(VGAM)
library(rpart.plot)
library(glmnet)
library(xgboost)
library(party)
library(locfit)
library(mgcv)
library(skellam)

#' Realign factor levels in a subset to match those in a reference data frame.
#' Prevents design matrix dimension mismatches when a model is fit on a
#' time-filtered subset that may be missing rare factor levels present in the
#' full data (e.g. unusual goalkeeper positions or player counts).
#' @param data    Data frame whose factor levels should be realigned
#' @param reference Data frame supplying the authoritative level sets
realign_factor_levels <- function(data, reference) {
  for (col in names(data)) {
    if (is.factor(data[[col]]) && col %in% names(reference)) {
      data[[col]] <- factor(data[[col]], levels = levels(reference[[col]]))
    }
  }
  data
}

#' Create train and test datasets based on a split season
create_traintest <- function(data, split_season, covariates) {
  train_data <- data %>%
    filter(season != split_season) %>%
    dplyr::select(all_of(covariates))
  test_data <- data %>%
    filter(season == split_season) %>%
    dplyr::select(all_of(covariates))
  return(list(train_data = train_data, test_data = test_data))
}

#' Helper to get predictions from a model
get_preds <- function(fit, newdata, final_order) {
  predict(fit, newdata, type = "probs")[, final_order, drop = FALSE]
}
