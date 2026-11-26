#!/bin/bash
cd /tmp/rfwork
export PYTHONPATH=/tmp/rfwork RF_PERM_BACKEND=python RF_ONLY_RUN=1 POWER_BACKEND=loky POWER_N_JOBS=4
L=experiments/meta_vs_embed_power/results
for sp in random ordered; do
  RF_SPLIT=$sp python3 -m experiments.meta_vs_embed_power.run_power --B 150 --n-perm 150 --n-jobs 4 --skip-anisotropy \
    --datasets bias --scenarios sentiment_meta syntax_meta train_vs_heldout \
    --feature-sets without_meta with_meta meta_only --out-suffix rf_b150_bias_upper_$sp > $L/power_rf_b150_bias_upper_$sp.log 2>&1
done
echo ALLDONE > $L/rf_upper_both.done
