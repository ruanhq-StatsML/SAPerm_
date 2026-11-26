#!/usr/bin/env bash
# Reruns bias meta_only (B=50) with block CFPerm + residual DR (power.py already merged). Called by chain_bias_dr_then_metafix.sh.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export POWER_BACKEND=loky PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
RES="$ROOT/experiments/meta_vs_embed_power/results"
LOG="$RES/power_bias_B50_metafix.log"
for sc in sentiment_meta syntax_meta train_vs_heldout; do
  until python3 -m experiments.meta_vs_embed_power \
      --B 50 --n-perm 150 --n-estimators 150 --n-jobs "${POWER_N_JOBS:-4}" \
      --datasets bias --scenarios "$sc" --feature-sets meta_only \
      --out-suffix bias_B50_mfix --skip-anisotropy >>"$LOG" 2>&1; do
    echo "[retry $sc $(date -Is)]" >>"$LOG"; sleep 10
  done
  python3 -m experiments.meta_vs_embed_power.scripts.merge_bias_b50_metafix >"$RES/power_bias_B50_table_metafix_${sc}.md" 2>>"$LOG"
  echo "[METAFIX SCENARIO DONE $sc $(date -Is)]" >>"$LOG"
done
echo "[METAFIX ALL DONE $(date -Is)]" >>"$LOG"
