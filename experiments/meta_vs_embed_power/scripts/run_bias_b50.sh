#!/usr/bin/env bash
# bias only, 3 arms, B=50 n_perm=150, scenarios sequential, all cores, resume-safe.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export POWER_BACKEND=loky PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
LOG="$ROOT/experiments/meta_vs_embed_power/results/power_bias_B50.log"
NJ="${POWER_N_JOBS:-4}"
for sc in sentiment_meta syntax_meta train_vs_heldout; do
  until python3 -m experiments.meta_vs_embed_power \
      --B 50 --n-perm 150 --n-estimators 150 --n-jobs "$NJ" \
      --datasets bias --scenarios "$sc" \
      --feature-sets without_meta with_meta meta_only \
      --out-suffix bias_B50 --skip-anisotropy >>"$LOG" 2>&1; do
    echo "[retry $sc $(date -Is)]" >>"$LOG"; sleep 10
  done
  python3 -m experiments.meta_vs_embed_power.scripts.bias_b50_table >"${LOG%.log}_table_${sc}.md" 2>>"$LOG"
  echo "[SCENARIO DONE $sc $(date -Is)]" >>"$LOG"
done
echo "[ALL DONE $(date -Is)]" >>"$LOG"
