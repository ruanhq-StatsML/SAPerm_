#!/usr/bin/env bash
# Sharded gender DR/RR/CF at k=full (five arms, B=50). Usage: run_dim_scan_gender_shard.sh i N [EMB]
# Job j goes to shard crc32(key) % N. Writes results/dim_scan_gender_shard{i}.jsonl; keys already in any
# results/dim_scan_gender*.jsonl are skipped. Resume-safe. Seeds and settings are identical to the unsharded run.
# Merge by copying the shard jsonl files next to each other; summarize_dim_scan dedups by key.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
I=${1:?shard index}; N=${2:?shard count}; EMB=${3:-MiniLM_L6}
python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4} --emb "$EMB" --shard "$I/$N" \
  --tests drc --b0 0 --b1 50 --ks full --kinds without with spacy vader with_decorr
echo "[shard $I/$N] done"
