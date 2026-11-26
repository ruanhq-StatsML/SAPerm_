#!/usr/bin/env bash
# gender full-embedding (no PCA), main VM: DR/RR/CF at k=full for the five core arms (MiniLM, B=50).
# RF (all arms, both embeddings) and diagnostics are already done. Per-column VADER arms, mtry controls and
# the k = 24/64/128 DR/RR/CF batches are cancelled (existing rows are kept). Resume-safe.
# Unsharded: bash run_dim_scan_gender.sh        Sharded: bash run_dim_scan_gender_shard.sh i N
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4} \
  --tests drc --b0 0 --b1 50 --ks full --kinds without with spacy vader with_decorr
echo "[chain] main VM done"
