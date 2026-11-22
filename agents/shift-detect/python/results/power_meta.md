# Meta-feature power (text DGPs)

Meta raises power on structure-aligned covariates (esp. hard syntax_shift); Type-I controlled.

Config: n_rep=20, n_doc=20, n_perm=40, α=0.05.

| scenario | baseline | meta | both | metric |
| --- | ---: | ---: | ---: | --- |
| null_shift | 0.15 | 0.05 | 0.00 | typeI |
| sentiment_shift | 0.40 | 1.00 | 1.00 | power |
| entity_shift | 1.00 | 1.00 | 1.00 | power |
| style_shift | 1.00 | 1.00 | 1.00 | power |
| syntax_shift **HARD** | 0.10 | 1.00 | 1.00 | power |

## Δ (meta − baseline)

- `sentiment_shift`: 0.40 → 1.00 (Δ=+0.60) ↑
- `entity_shift`: 1.00 → 1.00 (Δ=+0.00) ·
- `style_shift`: 1.00 → 1.00 (Δ=+0.00) ·
- `syntax_shift`: 0.10 → 1.00 (Δ=+0.90) ↑
