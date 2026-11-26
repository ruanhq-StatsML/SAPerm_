#!/usr/bin/env bash
# 2+2 core split: without/with_meta continuation + fixed meta_only (d2f3e00).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POWER_BACKEND=loky
R=experiments/meta_vs_embed_power/results
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s gender-embed -c "$ROOT" -- bash -lc \
 "POWER_ARMS=without_meta,with_meta POWER_N_JOBS=2 until python3 -m experiments.meta_vs_embed_power.run_gender_b50 train_vs_heldout null_shift syntax_meta 2>&1 | tee -a $R/gender_b50.log; do sleep 10; done"
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s gender-metafix -c "$ROOT" -- bash -lc \
 "POWER_ARMS=meta_only POWER_OUT=power_gender_B50_meta_only_fix.json POWER_CKPT=gender_b50_fix POWER_N_JOBS=2 until python3 -m experiments.meta_vs_embed_power.run_gender_b50 train_vs_heldout syntax_meta sentiment_meta null_shift 2>&1 | tee -a $R/gender_b50_fix.log; do sleep 10; done"
