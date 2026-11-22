#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "[watch] waiting for power_cross_modality_full.md ..."
while [[ ! -f results/power_cross_modality_full.md ]]; do
  if ! pgrep -f "power_cross_modality.py" >/dev/null; then
    echo "[watch] process gone without md"; tail -n 50 results/power_cross_modality_full.log || true; exit 1
  fi
  if [[ -f results/power_cross_modality.ckpt.npz ]]; then
    python3 -c "import numpy as np; d=np.load('results/power_cross_modality.ckpt.npz',allow_pickle=True); print('[watch]', len(d['keys']), '/600')"
  fi
  sleep 300
done
echo "[watch] cross-modality done; refreshing interaction writeup"
python3 refresh_interaction_writeup.py
cd /agent/repos/SAPerm
git add agents/shift-detect/python/results/power_cross_modality_full.json \
        agents/shift-detect/python/results/power_cross_modality_full.md \
        agents/shift-detect/python/results/interaction_writeup_v2.tex \
        agents/shift-detect/python/results/interaction_signatures.json \
        agents/shift-detect/python/results/unified_meta_power.tex 2>/dev/null || true
git commit -m "Add cross-modality results and interaction-evidence writeup v2." || true
git push origin cursor/embed-meta-power-0595 || true
echo "[watch] done"
