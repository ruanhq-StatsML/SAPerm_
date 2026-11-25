#!/usr/bin/env bash
# This VM: train_bias_finder embed arms only (forks handle gender/bias/meta_only).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
NPROC="${POWER_N_JOBS:-4}"
LOG="$ROOT/experiments/meta_vs_embed_power/results/bf_embed_b50.log"
PRIORITY_SCENARIOS=(sentiment_meta syntax_meta train_vs_heldout)
TAG=bf_embed_b50

run_scenarios() {
  local -n arr=$1
  python3 -m experiments.meta_vs_embed_power \
    --B 50 --n-perm 150 --n-estimators 150 --n-jobs "${NPROC}" \
    --datasets train_bias_finder \
    --scenarios "${arr[@]}" \
    --feature-sets without_meta with_meta \
    --out-suffix "${TAG}" --skip-anisotropy \
    2>&1 | tee -a "$LOG"
}

echo "[bf_embed_b50] start $(date -Is)" | tee -a "$LOG"
run_scenarios PRIORITY_SCENARIOS
# Optional null (Type-I) last:
# python3 -m ... --scenarios null_shift ...
echo "[bf_embed_b50] done priority scenarios $(date -Is)" | tee -a "$LOG"