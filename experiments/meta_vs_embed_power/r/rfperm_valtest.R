#!/usr/bin/env Rscript
# PermValTest-style RFPerm via official RFPerm::permTest_ + ranger (binary/regression).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: rfperm_valtest.R <exist.npz> <new.npz> <n_perm> <seed> [tr_prop]")
}
suppressPackageStartupMessages(library(RFPerm))
suppressPackageStartupMessages(library(ranger))

load_npz <- function(path) {
  # Minimal npz reader: expect .csv fallback from Python
  stop("use csv")
}

exist_path <- args[1]
new_path <- args[2]
B <- as.integer(args[3])
seed <- as.integer(args[4])
tr_prop <- if (length(args) >= 5) as.numeric(args[5]) else 0.7

set.seed(seed)
df_exist <- read.csv(exist_path, check.names = FALSE)
df_new <- read.csv(new_path, check.names = FALSE)
if (!"Y" %in% names(df_exist)) colnames(df_exist)[ncol(df_exist)] <- "Y"
if (!"Y" %in% names(df_new)) colnames(df_new)[ncol(df_new)] <- "Y"

n_col <- ncol(df_exist)
n_exist <- nrow(df_exist)
min_node <- max(1L, round(sqrt(n_exist) / 2))
p <- n_col - 1L
y <- df_exist$Y
n_cls <- length(unique(y))

if (n_cls <= 6) {
  mtry_fit <- max(1L, round(sqrt(p)))
  tr_ind <- sample(seq_len(n_exist), round(tr_prop * n_exist), replace = FALSE)
  df_tr <- df_exist[tr_ind, , drop = FALSE]
  df_val <- df_exist[-tr_ind, , drop = FALSE]
  model1 <- ranger::ranger(
    as.factor(Y) ~ ., data = df_tr, num.trees = 150,
    min.node.size = min_node, mtry = mtry_fit, sample.fraction = 0.8,
    probability = TRUE, seed = seed
  )
  prob_val <- predict(model1, df_val[, 1:p, drop = FALSE])$predictions
  prob_te <- predict(model1, df_new[, 1:p, drop = FALSE])$predictions
  ynum <- as.numeric(df_val$Y)
  cn <- colnames(prob_val)
  col <- if ("1" %in% cn) which(cn == "1") else if (ncol(prob_val) >= 2) 2 else 1
  val_err <- (ynum - prob_val[, col])^2
  test_err <- (as.numeric(df_new$Y) - prob_te[, col])^2
} else {
  mtry_fit <- max(1L, round(n_col / 3))
  tr_ind <- sample(seq_len(n_exist), round(tr_prop * n_exist), replace = FALSE)
  df_tr <- df_exist[tr_ind, , drop = FALSE]
  df_val <- df_exist[-tr_ind, , drop = FALSE]
  model1 <- ranger::ranger(
    Y ~ ., data = df_tr, num.trees = 150,
    min.node.size = min_node, mtry = mtry_fit, sample.fraction = 0.8,
    seed = seed
  )
  val_err <- (df_val$Y - predict(model1, df_val[, 1:p, drop = FALSE])$predictions)^2
  test_err <- (df_new$Y - predict(model1, df_new[, 1:p, drop = FALSE])$predictions)^2
}

result <- RFPerm:::permTest_(val_err, test_err, B = B)
d0 <- result$d0
d_list <- result$d_list
p_upper <- (sum(d_list >= d0) + 1) / (B + 1)
p_lower <- (sum(d0 > d_list) + 1) / (B + 1)
p_abs <- (sum(abs(d_list) >= abs(d0)) + 1) / (B + 1)
cat(sprintf(
  "p_value=%s\np_value_upper=%s\np_value_lower=%s\np_value_abs=%s\np_value_two_sided=%s\nstatistic=%s\n",
  p_upper, p_upper, p_lower, p_abs, p_abs, d0
))