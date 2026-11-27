#!/usr/bin/env bash
# Wall-clock first: one heavy job at a time, n_jobs = all CPUs per dataset.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
LOG_DIR="$ROOT/experiments/meta_vs_embed_power/results"
mkdir -p "$LOG_DIR"
NPROC="${POWER_N_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
if [[ "${NPROC}" -lt 2 ]]; then NPROC=4; fi

DATASETS=(train_bias_finder gender bias)

run_embed_ds() {
  local ds="$1"
  echo "[orchestrate] embed ${ds} n_jobs=${NPROC} $(date -Is)" | tee -a "$LOG_DIR/orchestrate_b150.log"
  python3 -m experiments.meta_vs_embed_power \
    --B 150 --n-perm 150 --n-estimators 150 --n-jobs "${NPROC}" \
    --datasets "$ds" \
    --feature-sets without_meta with_meta \
    --out-suffix b150_embed --skip-anisotropy \
    2>&1 | tee -a "$LOG_DIR/power_b150_run.log"
}

run_meta_ds() {
  local ds="$1"
  echo "[orchestrate] meta_only ${ds} n_jobs=${NPROC} $(date -Is)" | tee -a "$LOG_DIR/orchestrate_b150.log"
  python3 -m experiments.meta_vs_embed_power \
    --B 150 --n-perm 150 --n-estimators 150 --n-jobs "${NPROC}" \
    --datasets "$ds" \
    --feature-sets meta_only \
    --out-suffix meta_only --skip-anisotropy \
    2>&1 | tee -a "$LOG_DIR/power_meta_only_b150.log"
}

while true; do
  for ds in "${DATASETS[@]}"; do
  # Phase A: embed arms (without + with), exclusive 4 cores
    run_embed_ds "$ds" || true
  # Phase B: meta_only arm, exclusive 4 cores
    run_meta_ds "$ds" || true
    echo "[orchestrate] finished both phases for ${ds} $(date -Is)" | tee -a "$LOG_DIR/orchestrate_b150.log"
  done
  echo "[orchestrate] full grid done $(date -Is)" | tee -a "$LOG_DIR/orchestrate_b150.log"
  break
done