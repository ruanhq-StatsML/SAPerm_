#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export POWER_BACKEND=loky POWER_N_JOBS="${POWER_N_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
cd "$ROOT"
LOG="$ROOT/experiments/meta_vs_embed_power/results/gender_b50.log"
while true; do
  python3 -m experiments.meta_vs_embed_power.run_gender_b50 2>&1 | tee -a "$LOG" && break
  echo "[restart $(date -Is)]" | tee -a "$LOG"; sleep 10
done
