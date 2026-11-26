#!/usr/bin/env bash
# Amazon full-grid runner (no-PCA k=full focus + priority queue).
# Writes strictly under results/amazon/fullgrid/. Uses all CPU cores (POWER_N_JOBS=4).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND=python POWER_DR_NULL=residual POWER_BACKEND=loky
export POWER_N_JOBS="${POWER_N_JOBS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

M="python3 -m experiments.meta_vs_embed_power.run_amazon_fullgrid"
A="python3 -m experiments.meta_vs_embed_power.run_amazon_ablation"
STAGE="${STAGE:-vader_then_full}"
B50_ARMS="without_meta with_meta with_spacy with_vader4 with_meta_decorr"

run_rf_k() {
  local k="$1"
  POWER_CHUNK_MULT=16 $M --run --tests RFPerm --k "$k" --split random  --tag "fg_rf_k${k}" --b 150
  POWER_CHUNK_MULT=16 $M --run --tests RFPerm --k "$k" --split ordered --tag "fg_rf_k${k}_ordered" --b 150
  $M --summarize
  echo "[fullgrid rf done k=$k]"
}

run_vader_full() {
  $A --vader-diag --k-vals full
  $A --vader-rf --k-vals full --encoders MiniLM_L6 qwen3-0.6b
  $A --vader-summarize
  $M --summarize
  echo "[fullgrid vader_full done]"
}

run_b50_batch() {
  local tests="$1" tag="$2" enc="$3" phase="$4"
  if [ "$phase" = sent_syn ]; then
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec gift --encoders "$enc" --scenarios sentiment_meta syntax_meta --arms $B50_ARMS
  else
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec gift --encoders "$enc" --scenarios train_vs_heldout null_shift --arms $B50_ARMS
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec cross --encoders "$enc" --scenarios domain_shift null_shift --arms $B50_ARMS
  fi
  $M --summarize
  echo "[fullgrid stage done $tests $enc $phase]"
}

case "$STAGE" in
  diag)
    $M --diag --diag-r 50
    $M --summarize
    echo "[fullgrid diag done]"
    ;;
  rf_priority)
    run_rf_k full
    run_rf_k 128
    echo "[fullgrid rf_priority done]"
    ;;
  rf_k64)
    run_rf_k 64
    echo "[fullgrid rf_k64 done]"
    ;;
  vader_full)
    run_vader_full
    ;;
  full_drrr_cf)
    for enc in MiniLM_L6 qwen3-0.6b; do
      run_b50_batch DRPerm,RRPerm fg_drrr_kfull "$enc" sent_syn
      run_b50_batch DRPerm,RRPerm fg_drrr_kfull "$enc" rest
    done
    for enc in MiniLM_L6 qwen3-0.6b; do
      run_b50_batch CFPerm fg_cf_kfull "$enc" sent_syn
      run_b50_batch CFPerm fg_cf_kfull "$enc" rest
    done
    echo "[fullgrid all full_drrr_cf done]"
    ;;
  vader_then_full)
    run_vader_full
    for enc in MiniLM_L6 qwen3-0.6b; do
      run_b50_batch DRPerm,RRPerm fg_drrr_kfull "$enc" sent_syn
      run_b50_batch DRPerm,RRPerm fg_drrr_kfull "$enc" rest
    done
    for enc in MiniLM_L6 qwen3-0.6b; do
      run_b50_batch CFPerm fg_cf_kfull "$enc" sent_syn
      run_b50_batch CFPerm fg_cf_kfull "$enc" rest
    done
    echo "[fullgrid all vader_then_full done]"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 1
    ;;
esac
