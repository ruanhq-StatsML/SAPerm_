# gender dimension scan (k = 24 / 64 / 128 / full, MiniLM + Qwen)

Driver: `run_dim_scan.py`. Summary: `summarize_dim_scan.py` → `results/dim_scan_gender_summary.{json,md}`.
Diagnostics (y-AUC, AUC_W, Var(W−ê)): `diag_dim_scan.py` → `results/dim_scan_gender_diag.json`.

## Inputs (all in `results/`, committed)

| file | content |
|---|---|
| `meta_gender_n1000.npz` | M (1000×32) and `meta_names` |
| `spacy_gender_n1000.npz` | S (1000×36) and `spacy_names` |
| `gender_MiniLM_L6_embedding.npy` | MiniLM Z (1000×384) |
| `gender_qwen3-0.6b_embedding.npy` | Qwen Z (1000×1024); rebuild with `python3 -m experiments.meta_vs_embed_power.extract_gender_embeddings qwen3-0.6b` |

Row order is `data.load_gender(1000, 2026)`.

## Arms

`kind` → features. Z_k is `pca_embed(Z, k, seed=2026)` (standardize, then PCA); `full` is raw Z.

| kind | arm label | X | CF test block |
|---|---|---|---|
| `without` | `without_meta_k{k}` | Z_k | last 8 columns of Z_k |
| `with` | `with_meta_k{k}` | [Z_k, M] | M |
| `spacy` | `with_spacy_k{k}` | [Z_k, S] | S |
| `with_decorr` | `with_meta_decorr_k{k}` | [Z_k, M_kept] | M_kept |

Decorr rule (τ = 0.3, same as `decorr_features.py` on the amazon branch): drop any column with |spearman(column, y)| ≥ 0.3 on the 1000-row gender frame. On gender this drops `vader_neg`, `vader_pos`, `vader_compound`, `polar_gap` from M (28 kept). No spaCy column reaches 0.3, so `with_spacy_decorr` equals `with_spacy` and is not run. Windows are always selected on the full M.

VADER-focused kinds: `vader` = [Z_k, vader_neg, vader_pos, vader_compound, polar_gap] (label `with_vader`), single columns `v_neg` / `v_pos` / `v_compound` / `v_polar_gap`, and `vader_spacy` = [Z_k, VADER4, S]. "M minus VADER" is `with_decorr`.

Per-feature ablation kinds (MiniLM, k = 24 and full; rows in `results/dim_scan_gender_ablation.jsonl`): `all` = [Z_k, M, S] (label `with_all_meta`), `add:<col>` = [Z_k, col] (label `add[<col>]`), `loo:<col>` = [Z_k, M, S] without col (label `loo[<col>]`); spaCy columns are named `spacy:<name>`. `--kinds ablation` expands to all 137. Summary: `summarize_ablation.py` → `results/ablation_gender.{md,json}`; `--select` prints the 5 largest and 5 smallest single-column gains (mean over sentiment_meta and syntax_meta, k=24, random split) for the DR/RR check (`--tests dr` = DRPerm + RRPerm only).

`--mtry 7` adds RF-only control arms (`..._mtry7`) with the RFPerm classifier mtry fixed at round(√(24+32)) = 7.

## Seeds and pairing

`rep_seed = 2026 + 10007 * (crc32(scenario) % 997) + 17 * b`, identical to `worker.run_job`. Windows depend only on (scenario, b) and M, so every arm, k and embedding sees the same window pair; at k=24 the RF p-values reproduce `run_power` exactly.

Per-test seeds inside a rep: RF `seed`, DR `seed+3` (residual null), RR `seed+11`, CF `seed+5`. n_perm = 150, RF trees = 150, n_doc = 120.

## jsonl key convention

One JSON object per (arm, scenario, b, tests):

```
key = "{emb_prefix}{arm}|{scenario}|{b}|{tests}"
emb_prefix = ""              for MiniLM_L6
             "qwen3-0.6b:"   for Qwen
tests = "rf"   -> fields RF_random, RF_ordered   (p_value = upper tail; also p_value_abs, p_value_lower)
        "drc"  -> fields DR, RR, CF              (p_value, statistic, null_mean, null_sd; DR also var_w_resid)
```

Examples: `with_meta_kfull|syntax_meta|3|rf`, `qwen3-0.6b:without_meta_k128|null_shift|0|drc`, `with_meta_k128_mtry7|sentiment_meta|10|rf`.

Each machine appends to its own file (`--log`); the driver skips keys already in that file. `summarize_dim_scan.py` merges every `results/dim_scan_gender*.jsonl`, deduplicating by key.

## Batches (2026-11-26 07:00: core arms only, sharded)

Core arms: without_meta, with_meta, with_spacy, Z+VADER4 (`vader`), with_meta_decorr. Cancelled: per-feature ablation, per-column VADER arms (beyond the finished RF rows), mtry controls (beyond the finished RF rows), DR/RR/CF at k = 24 / 64 / 128. Finished rows are kept.

Remaining: DR/RR/CF at k = full, B = 50, five core arms, 4 scenarios, for MiniLM (sharded) and Qwen (separate VM, `run_dim_scan_gender_qwen_drc.sh`).

Sharding: `--shard i/N` keeps jobs with `crc32(key) % N == i` (the key is `arm|scenario|b|tests`), writes `results/dim_scan_gender_shard{i}.jsonl`, and skips keys already present in any `results/dim_scan_gender*.jsonl`. Seeds and settings are unchanged; a sharded run reproduces the unsharded p-values exactly (checked on 8 jobs, 0 mismatches). N = 4 for MiniLM:

```
bash experiments/meta_vs_embed_power/scripts/run_dim_scan_gender_shard.sh 0 4   # this VM
bash experiments/meta_vs_embed_power/scripts/run_dim_scan_gender_shard.sh 1 4
bash experiments/meta_vs_embed_power/scripts/run_dim_scan_gender_shard.sh 2 4
bash experiments/meta_vs_embed_power/scripts/run_dim_scan_gender_shard.sh 3 4
```

The optional third argument is the embedding (default `MiniLM_L6`; the same script works for `qwen3-0.6b`). Merge by putting all shard jsonl files in `results/`; `summarize_dim_scan.py` dedups by key.

Cost (4 cores): one DR+RR+CF rep is about 70 core-seconds at k = 24, 95 at MiniLM full, about 130 at Qwen full. Five arms at k = full are about 26 core-hours for MiniLM (about 6.5 h on 4 cores) and about 36 core-hours for Qwen (about 9 h).
