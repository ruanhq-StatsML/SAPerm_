#!/usr/bin/env bash
# gender full-embedding (no PCA) scan, second VM: Qwen DR/RR/CF at k=full, then optional k = 128 / 64 / 24.
# Resume-safe. Writes results/dim_scan_gender_qwen_vm.jsonl; summarize_dim_scan merges every
# dim_scan_gender*.jsonl by key. Set OPTIONAL=0 to stop after k=full.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=experiments/meta_vs_embed_power/results
Q=qwen3-0.6b
S="python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4} --emb $Q --log $R/dim_scan_gender_qwen_vm.jsonl"
DRC="--tests drc --b0 0 --b1 50"

if [ ! -f "$R/gender_${Q}_embedding.npy" ]; then
  python3 -m experiments.meta_vs_embed_power.extract_gender_embeddings $Q
fi
$S $DRC --ks full --kinds without with spacy vader with_decorr
echo "[chain] qwen full DR/RR/CF done"

if [ "${OPTIONAL:-1}" = "1" ]; then
  $S $DRC --ks 128 --kinds without with spacy
  $S $DRC --ks 24 64 --kinds without with spacy
  $S $DRC --ks 24 --kinds vader with_decorr
fi
echo "[chain] qwen VM done"
