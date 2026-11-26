#!/usr/bin/env bash
# gender dimension scan, main VM. Resume-safe: rerun to continue.
# Order: all RFPerm (B=150, MiniLM + Qwen), diagnostics, then DR/RR/CF batches 1, 3, 4.
# Batches 2 and 5 (Qwen DR/RR/CF) run on a second VM: scripts/run_dim_scan_gender_qwen_drc.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=experiments/meta_vs_embed_power/results
S="python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4}"
Q=qwen3-0.6b
RF="--tests rf --b0 0 --b1 150"
DRC="--tests drc --b0 0 --b1 50"

$S $RF --ks full 128
$S $RF --ks full 128 --mtry 7
$S $RF --ks 24 64
$S $RF --ks 24 full --kinds with_decorr

while [ ! -f "$R/gender_${Q}_embedding.npy" ]; do sleep 30; done
$S $RF --emb $Q --ks full 128
$S $RF --emb $Q --ks 24 64
$S $RF --emb $Q --ks 24 full --kinds with_decorr
$S $RF --emb $Q --ks full 128 --mtry 7

python3 -m experiments.meta_vs_embed_power.diag_dim_scan MiniLM_L6 $Q

# batch 1: MiniLM full + 128
$S $DRC --ks full 128
# batch 3: decorr arm at full, both embeddings
$S $DRC --ks full --kinds with_decorr
$S $DRC --emb $Q --ks full --kinds with_decorr
# batch 4: MiniLM 24 / 64 + decorr at 24
$S $DRC --ks 24 64
$S $DRC --ks 24 --kinds with_decorr
echo "[chain] main VM done"
