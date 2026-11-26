# RFPerm-only B=150 parallel workers

**Backend (2026-11-26): `python`** — `rfperm_official.PermValTest` + fixed `permTest` pool shuffle.  
B=10 vs R: power **0.25 vs 0.20**, p corr **~0.52**; gaps mostly **train/val split** (sklearn vs `sample`) and subprocess overhead, not a broken permutation. Use R only if you require bit-identical `RFPerm::permrf` (`RF_PERM_BACKEND=rfperm_r`).

## Env (every worker)

```bash
export PYTHONPATH=/agent/repos/SAPerm   # repo root
export RF_PERM_BACKEND=python         # or rfperm_r
export RF_ONLY_RUN=1                  # skip DR/RR/CF
export POWER_BACKEND=loky
export POWER_N_JOBS=4                 # one dataset per VM; do not oversubscribe
```

**Seed / windows:** `worker.py` — `rep_seed = 2026 + 10007*crc32(scenario)%997 + 17*b` (paired across arms; unchanged).

## Cache paths

| Dataset | Embeddings / meta |
|---------|-------------------|
| gender | `results/gender_MiniLM_L6_embedding.npy`, `results/meta_gender_n1000.npz`, `results/spacy_gender_n1000.npz` (for `with_spacy` if enabled) |
| bias | `results/bias_MiniLM_L6_embedding.npy`, `results/meta_bias_n1000.npz`, `results/spacy_bias_n1000.npz` |
| train_bias_finder | BERT CSV via `data.load_biasfinder`; meta `results/meta_train_bias_finder_n1000.npz` |
| Amazon gift/cross | `results/amazon/emb_{encoder}_*.npz`, `meta_*.npz`, `spacy_*.npz` (must exist; see `amazon_extract`) |

Default `--pca-k 24`, `--n-doc 120`, `--n-estimators 150`, `--n-max 1000`.

## One dataset / worker (gender example)

```bash
python3 -m experiments.meta_vs_embed_power.run_power \
  --B 150 --n-perm 150 --n-jobs 4 \
  --skip-anisotropy \
  --datasets gender \
  --scenarios null_shift sentiment_meta syntax_meta train_vs_heldout \
  --feature-sets without_meta with_meta meta_only \
  --out-suffix rf_b150_gender
```

**Outputs** (`experiments/meta_vs_embed_power/results/`):

- Checkpoint: `checkpoint_rf_b150_gender_<dataset>.npz` (resume-safe)
- Per-scenario: `power_scenario_rf_b150_gender_<dataset>_<scenario>_B150.json`
- Aggregate: `power_meta_embed_full_rf_b150_gender.json`

## Amazon (per spec × encoder)

Requires aligned `amazon_extract` artifacts. RF-only:

```bash
export RF_ONLY_RUN=1 RF_PERM_BACKEND=python POWER_B=150
python3 -m experiments.meta_vs_embed_power.run_amazon_b50 \
  --spec gift --tests RFPerm --tag rf_b150 \
  --B 150    # if CLI added; else set via env below
```

Until `--B` is wired, patch `run_power` directly or use:

```bash
POWER_B=150 python3 -c "
import os; os.environ['RF_ONLY_RUN']='1'
from experiments.meta_vs_embed_power.pipeline import run_power, OUT
from experiments.meta_vs_embed_power import config as C
run_power(B=150, n_perm=150, n_doc=C.N_DOC, pca_k=24, n_estimators=150, n_max=1000,
  seed=2026, datasets=['amazon_MiniLM_L6'], scenarios=['null_shift','sentiment_meta','syntax_meta','train_vs_heldout'],
  feature_sets=['without_meta','with_meta','meta_only'], n_jobs=4, out_dir=OUT, embed_model='MiniLM_L6',
  checkpoint_tag='rf_b150_amazon_gift')
"
```

Dataset names: `amazon_{encoder}` (gift), `amazon-cross_{encoder}` (cross).  
Cross scenarios: `null_shift`, `domain_shift` (add `sentiment_meta` / `syntax_meta` only for gift).

## Py vs R spot-check

```bash
python3 experiments/meta_vs_embed_power/compare_rfperm_py_r.py
# -> results/rfperm_py_r_compare_b10.json
```