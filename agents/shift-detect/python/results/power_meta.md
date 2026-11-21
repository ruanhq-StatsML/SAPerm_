# Meta-feature power argument (spaCy + VADER)

On alternatives whose signal sits in spaCy/VADER structure (sentiment, NER, style), meta-features raise rejection power versus surface baseline, while null Type-I stays controlled.

Config: n_rep=20, n_doc=20, n_perm=40, α=0.05.

| scenario | baseline | meta | both | metric |
| --- | ---: | ---: | ---: | --- |
| null_shift | 0.05 | 0.05 | 0.05 | typeI |
| sentiment_shift | 0.55 | 1.00 | 1.00 | power |
| entity_shift | 1.00 | 1.00 | 1.00 | power |
| style_shift | 1.00 | 1.00 | 1.00 | power |
| syntax_shift | 0.05 | 1.00 | 1.00 | power |

## Power deltas (meta − baseline)

- `sentiment_shift`: 0.55 → 1.00 (Δ=+0.45) ↑
- `entity_shift`: 1.00 → 1.00 (Δ=+0.00) ·
- `style_shift`: 1.00 → 1.00 (Δ=+0.00) ·
- `syntax_shift`: 0.05 → 1.00 (Δ=+0.95) ↑

Null Type-I: baseline=0.05, meta=0.05, both=0.05
