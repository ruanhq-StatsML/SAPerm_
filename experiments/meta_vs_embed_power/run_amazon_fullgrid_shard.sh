#!/usr/bin/env bash
# Sharded k=full DR/RR/CF queue (core arms only). Usage: SHARD=i/N bash run_amazon_fullgrid_shard.sh
# Each job (dataset|scenario|arm|b) goes to shard crc32(key) % N; results append to
# results/amazon/fullgrid/fullgrid_shard{i}.jsonl and finished keys are skipped on restart.
# Seeds/settings are identical to run_amazon_fullgrid.sh. Merge: copy all shard jsonl files into
# one results/amazon/fullgrid/ and run `python3 -m experiments.meta_vs_embed_power.run_amazon_fullgrid --summarize`.
set -euo pipefail
: "${SHARD:?set SHARD=i/N, e.g. SHARD=1/4}"
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export RF_PERM_BACKEND=python POWER_DR_NULL=residual POWER_BACKEND=loky
export POWER_N_JOBS="${POWER_N_JOBS:-$(nproc --all)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export POWER_SHARD="$SHARD"

M="python3 -m experiments.meta_vs_embed_power.run_amazon_fullgrid"
ARMS="without_meta with_meta with_spacy with_vader4 with_meta_decorr"

batch() {
  local tests="$1" tag="$2" enc="$3" phase="$4"
  if [ "$phase" = sent_syn ]; then
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec gift --encoders "$enc" --scenarios sentiment_meta syntax_meta --arms $ARMS
  else
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec gift --encoders "$enc" --scenarios train_vs_heldout null_shift --arms $ARMS
    POWER_CHUNK_MULT=2 $M --run --tests "$tests" --k full --tag "$tag" --b 50 \
      --spec cross --encoders "$enc" --scenarios domain_shift null_shift --arms $ARMS
  fi
  echo "[shard $SHARD stage done $tests $enc $phase]"
}

for enc in MiniLM_L6 qwen3-0.6b; do
  batch DRPerm,RRPerm fg_drrr_kfull "$enc" sent_syn
  batch DRPerm,RRPerm fg_drrr_kfull "$enc" rest
done
for enc in MiniLM_L6 qwen3-0.6b; do
  batch CFPerm fg_cf_kfull "$enc" sent_syn
  batch CFPerm fg_cf_kfull "$enc" rest
done
echo "[shard $SHARD all done]"
