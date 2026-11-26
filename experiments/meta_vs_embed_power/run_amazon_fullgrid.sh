#!/usr/bin/env bash
# Amazon full-grid runner. Writes only under results/amazon/fullgrid/.
#   STAGE=diag|rf_priority|rf_k64|vader_full|b50|all   (default: all)
#   SHARD=i/N   optional, only affects b50: job (dataset|scenario|arm|b) runs on shard crc32(key) % N,
#               rows append to fullgrid_shard{i}.jsonl and finished keys are skipped on restart.
# Merge shards: python3 -m experiments.meta_vs_embed_power.run_amazon_fullgrid --merge DIR... (then it summarizes).
# Same seeds, tags and arms as run_amazon_fullgrid_shard.sh, which this script supersedes.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND=python POWER_DR_NULL=residual POWER_BACKEND=loky
export POWER_N_JOBS="${POWER_N_JOBS:-$(nproc --all)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -n "${SHARD:-}" ] && export POWER_SHARD="$SHARD"

M="python3 -m experiments.meta_vs_embed_power.run_amazon_fullgrid"
A="python3 -m experiments.meta_vs_embed_power.run_amazon_ablation"
STAGE="${STAGE:-all}"
ENCODERS="MiniLM_L6 qwen3-0.6b"
CORE_ARMS="without_meta with_meta with_spacy with_vader4 with_meta_decorr"

summarize() { [ -n "${SHARD:-}" ] || $M --summarize; }

rf_k() {
  POWER_CHUNK_MULT=16 $M --run --tests RFPerm --k "$1" --split random  --tag "fg_rf_k$1" --b 150
  POWER_CHUNK_MULT=16 $M --run --tests RFPerm --k "$1" --split ordered --tag "fg_rf_k$1_ordered" --b 150
  summarize
}

vader_full() {
  $A --vader-diag --k-vals full
  $A --vader-rf --k-vals full --encoders $ENCODERS
  $A --vader-summarize
  summarize
}

b50_batch() {
  local tests="$1" tag="$2" enc="$3" phase="$4"
  local common=(--run --tests "$tests" --k full --tag "$tag" --b 50 --encoders "$enc" --arms $CORE_ARMS)
  if [ "$phase" = sent_syn ]; then
    POWER_CHUNK_MULT=2 $M "${common[@]}" --spec gift --scenarios sentiment_meta syntax_meta
  else
    POWER_CHUNK_MULT=2 $M "${common[@]}" --spec gift --scenarios train_vs_heldout null_shift
    POWER_CHUNK_MULT=2 $M "${common[@]}" --spec cross --scenarios domain_shift null_shift
  fi
  summarize
  echo "[fullgrid ${SHARD:+shard $SHARD }done $tests $enc $phase]"
}

b50() {
  for enc in $ENCODERS; do
    for phase in sent_syn rest; do b50_batch DRPerm,RRPerm fg_drrr_kfull "$enc" "$phase"; done
  done
  for enc in $ENCODERS; do
    for phase in sent_syn rest; do b50_batch CFPerm fg_cf_kfull "$enc" "$phase"; done
  done
}

case "$STAGE" in
  diag)        $M --diag --diag-r 50; summarize ;;
  rf_priority) rf_k full; rf_k 128 ;;
  rf_k64)      rf_k 64 ;;
  vader_full)  vader_full ;;
  b50)         b50 ;;
  all)         vader_full; b50 ;;
  *)           echo "Unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
echo "[fullgrid STAGE=$STAGE ${SHARD:+shard $SHARD }all done]"
