"""
Real-world knowledge representation → LoRA FT → embedding CSV
=============================================================
Drop-in companion to fine_tuning_data_extraction.py.

Flow:
  datasets/dataset.zip
      → extract typed KG / packed facts (fin_kg_posttrain.realworld)
      → ft_text_*.csv  (text + labels)
      → fine_tuning_data_wrapper_bert / gpt   [optional --run-ft]
      → Resulted_data/X_emb_kgrep_*.csv
      → pipeline_example.py / benchmark_all_method

Example (representation only, no GPU train):
  python fine_tuning_kg_representation.py --corpus equity_evaluation

Example (full FT hook, needs deps):
  python fine_tuning_kg_representation.py --corpus biasfinder --run-ft --ft-backend bert
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow `python fine_tuning_kg_representation.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fin_kg_posttrain.bridge_to_ft import main  # noqa: E402

if __name__ == "__main__":
    main()
