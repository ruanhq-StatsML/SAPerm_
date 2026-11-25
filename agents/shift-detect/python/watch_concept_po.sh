#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "[watch] waiting power_concept_po_full.md"
while [[ ! -f results/power_concept_po_full.md ]]; do
  if ! pgrep -f "power_concept_po_interaction.py" >/dev/null; then
    echo "[watch] process gone"; tail -n 40 results/power_concept_po_full.log || true; exit 1
  fi
  tail -n 2 results/power_concept_po_full.log || true
  sleep 120
done
# inject md table into tex
python3 - <<'PY'
from pathlib import Path
md = Path("results/power_concept_po_full.md").read_text()
tex_path = Path("results/concept_po_crosscheck.tex")
tex = tex_path.read_text()
if "Filled results" not in tex:
    tex = tex.replace(
        r"\textit{(table merged on completion)}",
        "\\textbf{Filled results.}\n\\begin{verbatim}\n" + md + "\n\\end{verbatim}",
    )
    tex_path.write_text(tex)
    print("tex updated")
else:
    print("already filled")
PY
cd /agent/repos/SAPerm
git add agents/shift-detect/python/results/power_concept_po_full.json \
        agents/shift-detect/python/results/power_concept_po_full.md \
        agents/shift-detect/python/results/concept_po_crosscheck.tex \
        agents/shift-detect/python/power_concept_po_interaction.py
git commit -m "Add concept-drift synthetic PO-risk × power cross-check results." || true
git push origin cursor/embed-meta-power-0595 || true
echo "[watch] done"
cat agents/shift-detect/python/results/power_concept_po_full.md
