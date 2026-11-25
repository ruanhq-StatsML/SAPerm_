#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
MODE="${1:-embed}"
NPROC="${POWER_N_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)}"
# Cloud pods sometimes report nproc=1; prefer online CPUs.
if [[ "${NPROC}" -lt 2 ]]; then
  NPROC="$(getconf _NPROCESSORS_ONLN)"
fi
LOG_DIR="$ROOT/experiments/meta_vs_embed_power/results"
mkdir -p "$LOG_DIR"

run_embed() {
  echo "[keepalive] embed n_jobs=${NPROC} nproc=${NPROC} $(date -Is)" | tee -a "$LOG_DIR/power_b150_run.log"
  python3 -m experiments.meta_vs_embed_power \
    --B 150 --n-perm 150 --n-estimators 150 --n-jobs "${NPROC}" \
    --datasets train_bias_finder gender bias \
    --feature-sets without_meta with_meta \
  --out-suffix b150_embed --skip-anisotropy \
    2>&1 | tee -a "$LOG_DIR/power_b150_run.log"
}

run_meta_only() {
  echo "[keepalive] meta_only n_jobs=${NPROC} nproc=${NPROC} $(date -Is)" | tee -a "$LOG_DIR/power_meta_only_b150.log"
  python3 -m experiments.meta_vs_embed_power \
    --B 150 --n-perm 150 --n-estimators 150 --n-jobs "${NPROC}" \
    --datasets train_bias_finder gender bias \
    --feature-sets meta_only \
    --out-suffix meta_only --skip-anisotropy \
    2>&1 | tee -a "$LOG_DIR/power_meta_only_b150.log"
}

while true; do
  if [[ "$MODE" == "meta_only" ]]; then
    run_meta_only && break
  else
    run_embed && break
  fi
  echo "[keepalive] restart in 10s ($(date -Is))" | tee -a "$LOG_DIR/keepalive_${MODE}.log"
  sleep 10
done