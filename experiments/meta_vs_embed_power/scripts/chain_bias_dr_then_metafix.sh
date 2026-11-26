#!/usr/bin/env bash
# 1) wait for run_bias_b50.sh  2) archive old (invalid-DR) numbers  3) DR residual rerun, 3 arms x 3 scenarios  4) meta_only block-CF rerun
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export POWER_BACKEND=loky PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
RES="$ROOT/experiments/meta_vs_embed_power/results"
until grep -q 'ALL DONE' "$RES/power_bias_B50.log" 2>/dev/null; do sleep 30; done
mkdir -p "$RES/prefix_fix"
cp -n "$RES"/power_bias_B50.json "$RES"/power_bias_B50_table_*.md "$RES"/checkpoint_bias_B50_bias.npz "$RES"/power_bias_B50.log "$RES/prefix_fix/"
export POWER_N_JOBS=4 POWER_B=50
python3 -m experiments.meta_vs_embed_power.run_dr_rerun bias without_meta,with_meta,meta_only sentiment_meta,syntax_meta,train_vs_heldout >>"$RES/power_bias_B50_dr_residual.log" 2>&1
echo "[DR RESIDUAL DONE $(date -Is)]" >>"$RES/power_bias_B50_dr_residual.log"
./experiments/meta_vs_embed_power/scripts/run_bias_b50_metafix.sh
