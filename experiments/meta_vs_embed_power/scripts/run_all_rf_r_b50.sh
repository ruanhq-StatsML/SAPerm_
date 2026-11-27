#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"
export RF_PERM_BACKEND=rfperm_r RF_ONLY_RUN=1 POWER_BACKEND=loky POWER_N_JOBS=4
LOG="$ROOT/experiments/meta_vs_embed_power/results/rf_r_b50_all.log"
for ds in gender bias train_bias_finder; do
  echo "=== $ds $(date -Is) ===" | tee -a "$LOG"
  python3 -m experiments.meta_vs_embed_power.run_rf_only_b50 "$ds" 2>&1 | tee -a "$LOG"
done