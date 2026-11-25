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

echo "[watch] launching ModaSense multimodal thesis run"
python3 power_mm_modasense.py --B 80 --n-perm 100 --n-jobs 4 --n-doc 60 --n-estimators 40 2>&1 | tee results/power_mm_modasense_full.log
# merge mm table into mm_thesis tex (simple append of md)
python3 - <<'PY'
from pathlib import Path
md = Path("results/power_mm_modasense_full.md")
tex = Path("results/mm_thesis_modasense.tex")
if md.exists() and tex.exists():
    t = tex.read_text()
    if "Filled results" not in t:
        body = md.read_text().replace("_", r"\_")
        t = t.replace(
            r"\section{What we are \emph{not} claiming}",
            "\\subsection{Filled results}\n\\begin{verbatim}\n" + md.read_text() + "\n\\end{verbatim}\n\n\\section{What we are \\emph{not} claiming}",
        )
        tex.write_text(t)
        print("updated mm thesis tex")
PY
cd /agent/repos/SAPerm
git add agents/shift-detect/python/results/power_mm_modasense_full.json agents/shift-detect/python/results/power_mm_modasense_full.md agents/shift-detect/python/results/mm_thesis_modasense.tex 2>/dev/null || true
git commit -m "Add ModaSense multimodal thesis power results." || true
git push origin cursor/embed-meta-power-0595 || true
