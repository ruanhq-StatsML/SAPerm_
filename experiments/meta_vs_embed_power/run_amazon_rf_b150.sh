#!/usr/bin/env bash
# Amazon RFPerm-only B=150, n_perm=150 (Python backend), all scenarios incl. null_shift, four arms,
# RF_SPLIT=random (main) and RF_SPLIT=ordered (windows in frame order, i.e. chronological for gift).
# Same paired rep seeds as the B=50 chain. Pauses the B=50 chain while it runs (RF has priority),
# resumes it on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND=python RF_ONLY_RUN=1 POWER_B=150 POWER_N_PERM=150
export POWER_BACKEND=loky POWER_N_JOBS="${POWER_N_JOBS:-4}" POWER_CHUNK_MULT="${POWER_CHUNK_MULT:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
A=experiments/meta_vs_embed_power/results/amazon
R="python3 -m experiments.meta_vs_embed_power.run_amazon_b50 --tests RFPerm"
ARMS="without_meta with_meta with_spacy meta_only"

for f in alignment_report.json alignment_report_cross.json alignment_report_spacy_gift.json alignment_report_spacy_cross.json; do
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/$f'))['passed'] else 1)"
done

chain_pids() { pgrep -f "run_amazon_b50 --tests RFPerm,CFPerm" | while read -r p; do echo "$p"; pgrep -P "$p"; done; }
PAUSED="$(chain_pids | tr '\n' ' ')"
[ -n "$PAUSED" ] && kill -STOP $PAUSED && echo "[rf150] paused B=50 chain: $PAUSED"
trap '[ -n "$PAUSED" ] && kill -CONT $PAUSED 2>/dev/null && echo "[rf150] resumed B=50 chain"' EXIT

for enc in ${ENCODERS:-MiniLM_L6 qwen3-0.6b}; do
  for split in ${SPLITS:-random ordered}; do
    tag=rf_b150; [ "$split" = ordered ] && tag=rf_b150_ordered
    RF_SPLIT=$split $R --tag "$tag" --spec gift --encoders "$enc" --arms $ARMS
    RF_SPLIT=$split $R --tag "$tag" --spec cross --encoders "$enc" --arms $ARMS
  done
  python3 -m experiments.meta_vs_embed_power.rf_b150_table > /dev/null
  echo "[rf150 encoder done $enc]"
done
echo "[rf150 done]"
