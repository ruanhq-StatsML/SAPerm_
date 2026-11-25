# Meta vs embedding power (single pipeline)

Integrates the Colab notebook flow (`Collections_of_Embeddings_6dec.ipynb`),
`utils.text_meta`, and cached `get_embedding` / BiasFinder BERT CSV.

## Pipeline

```
extract (LLM or frozen BERT, batch=32, ≤1000 rows)
  → meta (utils.text_meta)
  → features: without_meta = PCA(Z)  |  with_meta = hstack(Z_pca, M)
  → power: RFPerm, DRPerm, RRPerm, CFPerm (RF outcome, B=150, n_perm=150)
  → geometry: anisotropy + effective_rank (n_pc = linspace(30,330,25))
```

## Run

```bash
export PYTHONPATH=/agent/repos/SAPerm
python -m experiments.meta_vs_embed_power --smoke
python -m experiments.meta_vs_embed_power --B 150 --n-perm 150 --n-jobs 4
python -m experiments.meta_vs_embed_power --anisotropy-only
```

**Datasets:** `train_bias_finder` (frozen BERT + GEN text), `gender`, `bias` (`train_bias2`).

**Outputs:** `experiments/meta_vs_embed_power/results/`

| File | Content |
|------|---------|
| `power_meta_embed_{smoke,full}.json` | Rejection rates by dataset / scenario / feature set |
| `anisotropy_effective_rank.json` | PCA sweep metrics |
| `*_embedding.npy`, `meta_*.npz` | Embeddings + meta caches |

Core RFPerm / CFPerm / DRPerm / RRPerm logic lives in `perm_lib.py` (experiment-local RF ports; upstream libs unchanged).