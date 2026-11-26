#!/usr/bin/env bash
# Amazon B=50 chain: shift scenarios first (MiniLM, then Qwen), null_shift last.
# Resumes from checkpoints; cross runs only if its alignment report passed.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND="${RF_PERM_BACKEND:-python}"
export POWER_BACKEND=loky POWER_N_JOBS="${POWER_N_JOBS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TESTS="${TESTS:-RFPerm,CFPerm,RRPerm}"
TAG="${TAG:-py}"
R="python3 -m experiments.meta_vs_embed_power.run_amazon_b50 --tests $TESTS --tag $TAG"
REPORT=experiments/meta_vs_embed_power/results/amazon/alignment_report_cross.json

python3 -c "import json,sys; sys.exit(0 if json.load(open('$REPORT'))['passed'] else 1)"

for enc in MiniLM_L6 qwen3-0.6b; do
  $R --spec gift --encoders "$enc" --scenarios sentiment_meta syntax_meta train_vs_heldout
  $R --spec cross --encoders "$enc" --scenarios domain_shift
done
for enc in MiniLM_L6 qwen3-0.6b; do
  $R --spec all --encoders "$enc" --scenarios null_shift
done
python3 -m experiments.meta_vs_embed_power.run_amazon_b50 --table
echo "[chain done $TAG]"
