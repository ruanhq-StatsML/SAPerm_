#!/usr/bin/env bash
# gender dimension scan, second VM: DR/RR/CF batches 2 and 5 (Qwen). Resume-safe: rerun to continue.
# Writes results/dim_scan_gender_qwen_vm.jsonl; summarize_dim_scan merges every dim_scan_gender*.jsonl by key.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=experiments/meta_vs_embed_power/results
Q=qwen3-0.6b
S="python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4} --emb $Q --log $R/dim_scan_gender_qwen_vm.jsonl"
DRC="--tests drc --b0 0 --b1 50"

if [ ! -f "$R/gender_${Q}_embedding.npy" ]; then
  echo "missing $R/gender_${Q}_embedding.npy: merge it from the extraction machine, do not re-extract" >&2
  exit 1
fi
# batch 2: Qwen full + 128
$S $DRC --ks full 128
# batch 5: Qwen decorr at 24, then 24 / 64
$S $DRC --ks 24 --kinds with_decorr
$S $DRC --ks 24 64
echo "[chain] qwen VM done"
