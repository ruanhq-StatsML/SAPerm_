#!/usr/bin/env bash
# gender full-embedding (no PCA) scan, main VM. Resume-safe: rerun to continue.
# Order: remaining RF at k=full (Qwen VADER, decorr, mtry=7), diagnostics, DR/RR/CF at k=full (MiniLM),
# then optional DR/RR/CF at k = 128 / 64 / 24 (set OPTIONAL=0 to stop before them).
# Qwen DR/RR/CF runs on a second VM: scripts/run_dim_scan_gender_qwen_drc.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S="python3 -m experiments.meta_vs_embed_power.run_dim_scan --n-jobs ${N_JOBS:-4}"
Q=qwen3-0.6b
RF="--tests rf --b0 0 --b1 150"
DRC="--tests drc --b0 0 --b1 50"
VK="vader v_neg v_pos v_compound v_polar_gap vader_spacy"
FULL_ARMS="without with spacy vader with_decorr"

$S $RF --emb $Q --ks full --kinds $VK
$S $RF --emb $Q --ks full --kinds with_decorr
$S $RF --emb $Q --ks full --mtry 7

python3 -m experiments.meta_vs_embed_power.diag_dim_scan MiniLM_L6 $Q

$S $DRC --ks full --kinds $FULL_ARMS
echo "[chain] full DR/RR/CF done"

if [ "${OPTIONAL:-1}" = "1" ]; then
  $S $DRC --ks 128 --kinds without with spacy
  $S $DRC --ks 24 64 --kinds without with spacy
  $S $DRC --ks 24 --kinds vader with_decorr
fi
echo "[chain] main VM done"
