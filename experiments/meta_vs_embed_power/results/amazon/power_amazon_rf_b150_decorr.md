**amazon-cross_MiniLM_L6**

| 场景 | w/o | with_meta | meta_decorr | with_spacy | spacy_decorr | Δmeta | Δmeta_decorr | Δspacy | Δspacy_decorr | τ=0.2 Δmeta_decorr / Δspacy_decorr | ordered Δmeta_decorr / Δspacy_decorr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| null_shift | 0.04 | 0.07 | 0.07 | 0.04 | 0.05 | +0.03 | +0.03 | +0.00 | +0.01 | +0.00 / +0.00 | +0.00 / -0.03 |
| domain_shift | 0.33 | 0.36 | 0.41 | 0.35 | 0.35 | +0.03 | +0.09 | +0.03 | +0.03 | +0.07 / +0.05 | -0.01 / -0.02 |

**amazon-cross_qwen3-0.6b**

| 场景 | w/o | with_meta | meta_decorr | with_spacy | spacy_decorr | Δmeta | Δmeta_decorr | Δspacy | Δspacy_decorr | τ=0.2 Δmeta_decorr / Δspacy_decorr | ordered Δmeta_decorr / Δspacy_decorr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| null_shift | 0.05 | 0.06 | 0.10 | 0.09 | 0.07 | +0.01 | +0.05 | +0.03 | +0.02 | +0.01 / +0.03 | -0.01 / +0.00 |
| domain_shift | 0.69 | 0.37 | 0.51 | 0.43 | 0.51 | -0.33 | -0.18 | -0.26 | -0.19 | -0.07 / -0.13 | -0.15 / -0.15 |

**amazon_MiniLM_L6**

| 场景 | w/o | with_meta | meta_decorr | with_spacy | spacy_decorr | Δmeta | Δmeta_decorr | Δspacy | Δspacy_decorr | τ=0.2 Δmeta_decorr / Δspacy_decorr | ordered Δmeta_decorr / Δspacy_decorr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| null_shift | 0.03 | 0.07 | 0.03 | 0.03 | 0.05 | +0.03 | +0.00 | +0.00 | +0.01 | +0.01 / +0.00 | -0.02 / -0.01 |
| sentiment_meta | 0.82 | 0.64 | 0.94 | 0.94 | 0.90 | -0.18 | +0.12 | +0.12 | +0.08 | +0.05 / +0.09 | +0.03 / +0.06 |
| syntax_meta | 0.36 | 0.19 | 0.37 | 0.21 | 0.35 | -0.17 | +0.01 | -0.15 | -0.01 | +0.01 / +0.05 | -0.12 / -0.08 |
| train_vs_heldout | 0.16 | 0.17 | 0.23 | 0.24 | 0.22 | +0.01 | +0.07 | +0.08 | +0.06 | +0.04 / +0.05 | +0.03 / -0.03 |

**amazon_qwen3-0.6b**

| 场景 | w/o | with_meta | meta_decorr | with_spacy | spacy_decorr | Δmeta | Δmeta_decorr | Δspacy | Δspacy_decorr | τ=0.2 Δmeta_decorr / Δspacy_decorr | ordered Δmeta_decorr / Δspacy_decorr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| null_shift | 0.07 | 0.06 | 0.05 | 0.03 | 0.05 | -0.01 | -0.02 | -0.04 | -0.02 | -0.03 / -0.01 | +0.00 / +0.01 |
| sentiment_meta | 0.89 | 0.66 | 0.94 | 0.93 | 0.93 | -0.23 | +0.05 | +0.05 | +0.04 | +0.03 / +0.04 | +0.01 / +0.01 |
| syntax_meta | 0.29 | 0.17 | 0.33 | 0.20 | 0.33 | -0.12 | +0.05 | -0.09 | +0.05 | +0.07 / +0.12 | -0.03 / -0.01 |
| train_vs_heldout | 0.23 | 0.17 | 0.22 | 0.27 | 0.25 | -0.06 | -0.01 | +0.04 | +0.03 | +0.01 / +0.01 | +0.02 / -0.01 |
