#!/usr/bin/env bash
# Split one slow shard across M machines. Usage: run_dim_scan_gender_subshard.sh i N j M [EMB]
# Shard i of N keeps crc32(key) % N == i; sub-shard j of M keeps (crc32(key) // N) % M == j. The sub-shards of a
# shard are disjoint by construction, so no key runs on two machines as long as every machine of that shard uses
# a distinct j with the same M. The original owner of shard i must stop and restart as sub-shard j=0 (use this
# script); its finished rows are skipped by key. Writes results/dim_scan_gender_shard{i}_{j}of{M}.jsonl.
# Remaining work per shard: python3 -m experiments.meta_vs_embed_power.run_dim_scan --tests drc --ks full \
#   --kinds without with spacy vader with_decorr --b1 50 --status   (after git pull of everyone's jsonl).
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
I=${1:?shard index}; N=${2:?shard count}; J=${3:?sub-shard index}; M=${4:?sub-shard count}; EMB=${5:-MiniLM_L6}
python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4} --emb "$EMB" --shard "$I/$N" --subshard "$J/$M" \
  --tests drc --b0 0 --b1 50 --ks full --kinds without with spacy vader with_decorr
echo "[shard $I/$N sub $J/$M] done"
