#!/usr/bin/env bash
# train_bias_finder x meta_only only; scenarios run one at a time on all cores.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
R="$ROOT/experiments/meta_vs_embed_power/results"
NJ="${POWER_N_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
B="${B:-50}"
SCENARIOS=(${SCENARIOS:-sentiment_meta syntax_meta train_vs_heldout})
for sc in "${SCENARIOS[@]}"; do
  echo "[tbf-meta] start $sc B=$B n_jobs=$NJ $(date -Is)" | tee -a "$R/tbf_meta_only.log"
  python3 -m experiments.meta_vs_embed_power \
    --B "$B" --n-perm 150 --n-estimators 150 --n-jobs "$NJ" \
    --datasets train_bias_finder --scenarios "$sc" \
    --feature-sets meta_only --out-suffix "tbfmeta_$sc" --skip-anisotropy \
    2>&1 | grep --line-buffered -v -E 'UserWarning|warnings.warn' | tee -a "$R/tbf_meta_only.log"
  python3 "$ROOT/experiments/meta_vs_embed_power/scripts/merge_tbf_meta.py" "$B"
done
echo "[tbf-meta] all done $(date -Is)" | tee -a "$R/tbf_meta_only.log"
