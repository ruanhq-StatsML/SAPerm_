#!/usr/bin/env bash
# gender dimension scan (MiniLM + Qwen). Resume-safe: rerun to continue.
# Order: all RFPerm (B=150) first, then diagnostics, then DR/RR/CF (B=50) by priority batch.
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

$S $DRC --ks full 128
$S $DRC --emb $Q --ks full 128
$S $DRC --ks full --kinds with_decorr
$S $DRC --emb $Q --ks full --kinds with_decorr
$S $DRC --ks 24 64
$S $DRC --ks 24 --kinds with_decorr
$S $DRC --emb $Q --ks 24 64
$S $DRC --emb $Q --ks 24 --kinds with_decorr
echo "[chain] all done"
