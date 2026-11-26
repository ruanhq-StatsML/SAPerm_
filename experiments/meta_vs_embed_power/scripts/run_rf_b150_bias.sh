#!/bin/bash
cd /tmp/rfwork
export PYTHONPATH=/tmp/rfwork RF_PERM_BACKEND=python RF_ONLY_RUN=1 POWER_BACKEND=loky POWER_N_JOBS=4
python3 -m experiments.meta_vs_embed_power.run_power --B 150 --n-perm 150 --n-jobs 4 --skip-anisotropy \
  --datasets bias --scenarios sentiment_meta syntax_meta train_vs_heldout \
  --feature-sets without_meta with_meta meta_only --out-suffix rf_b150_bias 2>&1 | tee experiments/meta_vs_embed_power/results/power_rf_b150_bias.log
echo "[ALL DONE $(date -Is)]" >> experiments/meta_vs_embed_power/results/power_rf_b150_bias.log
