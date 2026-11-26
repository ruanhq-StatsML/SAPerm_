#!/usr/bin/env bash
# Waits for run_bias_b50.sh to finish, archives old numbers, pulls power.py from d2f3e00, reruns bias meta_only (B=50).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export POWER_BACKEND=loky PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
RES="$ROOT/experiments/meta_vs_embed_power/results"
LOG="$RES/power_bias_B50_metafix.log"
until grep -q 'ALL DONE' "$RES/power_bias_B50.log" 2>/dev/null; do sleep 30; done
mkdir -p "$RES/prefix_fix"
cp -n "$RES"/power_bias_B50.json "$RES"/power_bias_B50_table_*.md "$RES"/checkpoint_bias_B50_bias.npz "$RES"/power_bias_B50.log "$RES/prefix_fix/"
git checkout d2f3e00 -- experiments/meta_vs_embed_power/power.py
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
