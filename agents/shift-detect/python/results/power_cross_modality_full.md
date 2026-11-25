# Cross-modality power (embed ↔ meta blocks)

B=150, N_perm=150, α=0.05. X=[Z_PCA, M]; CFPerm LOCO per named block via feature_schema indices.

## Detection (RFPerm / DRPerm)

| scenario | set | RFPerm | DRPerm |
| --- | --- | ---: | ---: |
| null_shift | embed | 0.03 | 0.07 |
| null_shift | embed_meta | 0.05 | 0.05 |
| sentiment_meta | embed | 0.05 | 0.00 |
| sentiment_meta | embed_meta | 0.05 | 0.00 |
| syntax_meta | embed | 0.16 | 0.00 |
| syntax_meta | embed_meta | 0.07 | 0.00 |
| train_vs_heldout | embed | 1.00 | 1.00 |
| train_vs_heldout | embed_meta | 1.00 | 1.00 |

## Cross-modality CFPerm (block power on embed_meta X)

| scenario | embed_pca | meta | vader_sent | spacy_syntax | spacy_style |
| --- | ---: | ---: | ---: | ---: | ---: |
| null_shift | 0.05 | 0.07 | 0.04 | 0.09 | 0.04 |
| sentiment_meta | 0.01 | 0.03 | 0.02 | 0.04 | 0.06 |
| syntax_meta | 0.16 | 0.31 | 0.18 | 0.32 | 0.20 |
| train_vs_heldout | 0.96 | 0.99 | 0.62 | 0.67 | 0.73 |

## Reading

- High CFPerm on `embed_pca` vs low on meta ⇒ shift lives in embedding modality.
- High on `vader_sent` / `spacy_syntax` ⇒ interpretable meta modality localizes it.
- `meta` = all meta dims; sub-blocks are the portable spaCy/VADER slices.
