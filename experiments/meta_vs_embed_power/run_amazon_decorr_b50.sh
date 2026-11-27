#!/usr/bin/env bash
# Amazon DR / RR / CF at B=50 for the decorrelated arms. Same paired rep seeds as the main chain.
# DR+RR use tag "decorr", CF uses tag "decorr_cf" (checkpoints are keyed per job, so a test run
# later under the same tag would be skipped).
#   STAGE=falsify : gift MiniLM sentiment/syntax, tau=0.2 and tau=0.3 arms; pauses the main chain.
#   STAGE=grid    : tau=0.3 arms on every scenario, both encoders, plus without_meta on null_shift.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export POWER_DR_NULL=residual POWER_BACKEND=loky POWER_N_JOBS="${POWER_N_JOBS:-4}" POWER_CHUNK_MULT="${POWER_CHUNK_MULT:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
M="python3 -m experiments.meta_vs_embed_power.run_amazon_b50"
DRRR="$M --tests DRPerm,RRPerm --tag decorr"
CF="$M --tests CFPerm --tag decorr_cf"
STAGE="${STAGE:-grid}"

if [ "${PAUSE_CHAIN:-0}" = 1 ]; then
  chain_pids() { pgrep -f "run_amazon_b50 --tests RFPerm,CFPerm" | while read -r p; do echo "$p"; pgrep -P "$p"; done; }
  PAUSED="$(chain_pids | tr '\n' ' ')"
  [ -n "$PAUSED" ] && kill -STOP $PAUSED && echo "[decorr] paused B=50 chain: $PAUSED"
  trap '[ -n "$PAUSED" ] && kill -CONT $PAUSED 2>/dev/null && echo "[decorr] resumed B=50 chain"' EXIT
fi

both() { $DRRR "$@"; $CF "$@"; }

if [ "$STAGE" = falsify ]; then
  # t02nv: tau=0.2 drop plus verb_ratio / noun_ratio, the components of the syntax selection variable.
  G="--spec gift --encoders MiniLM_L6"
  KEY="with_meta_decorr with_meta_decorr_t02 with_meta_decorr_t02nv"
  ARMS="$KEY with_spacy_decorr with_spacy_decorr_t02"
  $DRRR $G --scenarios syntax_meta --arms $KEY
  echo "[decorr falsify syntax M-arms DR/RR done]"
  $DRRR $G --scenarios syntax_meta sentiment_meta --arms $ARMS
  echo "[decorr falsify DR/RR done]"
  $CF $G --scenarios syntax_meta sentiment_meta --arms $ARMS
  echo "[decorr falsify CF done]"
else
  for enc in ${ENCODERS:-MiniLM_L6 qwen3-0.6b}; do
    both --spec gift --encoders "$enc" --scenarios sentiment_meta syntax_meta train_vs_heldout null_shift \
      --arms with_meta_decorr with_spacy_decorr
    both --spec cross --encoders "$enc" --scenarios domain_shift null_shift --arms with_meta_decorr with_spacy_decorr
    both --spec gift --encoders "$enc" --scenarios null_shift --arms without_meta
    both --spec cross --encoders "$enc" --scenarios null_shift --arms without_meta
    echo "[decorr grid encoder done $enc]"
  done
  echo "[decorr grid done]"
fi
