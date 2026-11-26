#!/usr/bin/env bash
# gender dimension scan, main VM. Resume-safe: rerun to continue.
# Order: RF base grid, VADER-focused RF grid (MiniLM, Qwen), per-feature RF ablation (MiniLM),
# remaining RF, diagnostics, then DR/RR/CF: batch 0 (VADER vs M at k 24/full), DR/RR on the
# top/bottom-5 ablation columns at k=24, then batches 1, 3, 4.
# Batches 2 and 5 (Qwen DR/RR/CF) run on a second VM: scripts/run_dim_scan_gender_qwen_drc.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=experiments/meta_vs_embed_power/results
S="python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4}"
ABL="--log $R/dim_scan_gender_ablation.jsonl"
Q=qwen3-0.6b
RF="--tests rf --b0 0 --b1 150"
DRC="--tests drc --b0 0 --b1 50"
VK="vader v_neg v_pos v_compound v_polar_gap vader_spacy"

$S $RF --ks full 128
$S $RF --ks full 128 --mtry 7
$S $RF --ks 24 64
$S $RF --ks 24 full --kinds with_decorr

# VADER-focused grid (M minus VADER == with_decorr)
$S $RF --ks 24 128 full --kinds $VK
$S $RF --ks 128 --kinds with_decorr

while [ ! -f "$R/gender_${Q}_embedding.npy" ]; do sleep 30; done
$S $RF --emb $Q --ks full 128
$S $RF --emb $Q --ks 24 64
$S $RF --emb $Q --ks 24 128 full --kinds $VK

# per-feature ablation: Z+column and Z+[M,S] minus column, every column of M and S
$S $RF --ks 24 full --kinds ablation $ABL

$S $RF --emb $Q --ks 24 128 full --kinds with_decorr
$S $RF --emb $Q --ks full 128 --mtry 7

python3 -m experiments.meta_vs_embed_power.diag_dim_scan MiniLM_L6 $Q

# batch 0: Z+VADER and Z+M at k 24 / full (MiniLM)
$S $DRC --ks 24 full --kinds vader with
# DR/RR on the 5 largest and 5 smallest single-column gains (k=24)
PICKS=$(python3 -m experiments.meta_vs_embed_power.summarize_ablation --select)
$S --tests dr --b0 0 --b1 50 --ks 24 --kinds without $PICKS \
   --scenarios null_shift sentiment_meta syntax_meta $ABL
# batch 1: MiniLM full + 128
$S $DRC --ks full 128
# batch 3: decorr arm at full, both embeddings
$S $DRC --ks full --kinds with_decorr
$S $DRC --emb $Q --ks full --kinds with_decorr
# batch 4: MiniLM 24 / 64 + decorr at 24
$S $DRC --ks 24 64
$S $DRC --ks 24 --kinds with_decorr
echo "[chain] main VM done"
