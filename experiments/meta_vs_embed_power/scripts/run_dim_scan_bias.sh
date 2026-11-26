#!/bin/bash
cd /tmp/rfwork
export PYTHONPATH=/tmp/rfwork RF_PERM_BACKEND=python POWER_BACKEND=loky OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m experiments.meta_vs_embed_power.run_dim_scan_bias --ks full 24 --b1 150 --n-jobs 4 >> experiments/meta_vs_embed_power/results/dim_scan_bias.log 2>&1
echo "[ALL DONE $(date -Is)]" >> experiments/meta_vs_embed_power/results/dim_scan_bias.log
