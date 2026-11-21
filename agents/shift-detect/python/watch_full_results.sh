#!/usr/bin/env bash
# Wait for overnight power_embed_meta full results, then commit + push.
set -euo pipefail
cd "$(dirname "$0")"
echo "[watch] waiting for results/power_embed_meta_full.md ..."
while [[ ! -f results/power_embed_meta_full.md ]]; do
  if ! pgrep -f "power_embed_meta.py" >/dev/null; then
    echo "[watch] power process gone without full.md"
    tail -n 100 results/power_embed_meta_full.log || true
    ls -la results/
    exit 1
  fi
  if [[ -f results/power_embed_meta_full.ckpt.npz ]]; then
    python3 -c "import numpy as np; from pathlib import Path; p=Path('results/power_embed_meta_full.ckpt.npz'); d=np.load(p, allow_pickle=True); print('[watch] ckpt', len(d['keys']), '/900')"
  fi
  tail -n 2 results/power_embed_meta_full.log 2>/dev/null || true
  sleep 300
done
echo "[watch] full results ready; committing"
cd /agent/repos/SAPerm
git add agents/shift-detect/python/results/power_embed_meta_full.json \
        agents/shift-detect/python/results/power_embed_meta_full.md
git status -sb
git commit -m "Add full embed±meta power results (B=150, N_perm=120)." || echo "nothing to commit"
git push origin cursor/embed-meta-power-0595
echo "[watch] done"
cat agents/shift-detect/python/results/power_embed_meta_full.md
