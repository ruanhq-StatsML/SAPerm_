#!/usr/bin/env bash
# Amazon B=50: with_meta vs without_meta first (MiniLM, then Qwen), meta_only last.
# Shift scenarios only (gift: sentiment/syntax/time; cross: domain). Resumes from checkpoints;
# refuses to start unless both alignment reports passed.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND="${RF_PERM_BACKEND:-python}"
export POWER_DR_NULL="${POWER_DR_NULL:-residual}"
export POWER_BACKEND=loky POWER_N_JOBS="${POWER_N_JOBS:-4}" POWER_CHUNK_MULT="${POWER_CHUNK_MULT:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TESTS="${TESTS:-RFPerm,CFPerm,RRPerm,DRPerm}"
TAG="${TAG:-py}"
R="python3 -m experiments.meta_vs_embed_power.run_amazon_b50 --tests $TESTS --tag $TAG"
A=experiments/meta_vs_embed_power/results/amazon

for f in alignment_report.json alignment_report_cross.json; do
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/$f'))['passed'] else 1)"
done

run_enc() {  # $1 encoder, rest = arms
  local enc="$1"; shift
  $R --spec gift --encoders "$enc" --scenarios sentiment_meta syntax_meta train_vs_heldout --arms "$@"
  $R --spec cross --encoders "$enc" --scenarios domain_shift --arms "$@"
  python3 -m experiments.meta_vs_embed_power.run_amazon_b50 --table > /dev/null
  echo "[chain stage done $enc $*]"
}

run_enc MiniLM_L6 without_meta with_meta
run_enc qwen3-0.6b without_meta with_meta
run_enc MiniLM_L6 meta_only
run_enc qwen3-0.6b meta_only
echo "[chain done $TAG]"
