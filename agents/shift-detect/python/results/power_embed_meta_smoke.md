# Embedding ± Meta-feature power (RFPerm / DRPerm / CFPerm)

Question: do spaCy+VADER meta-features (M) give **incremental value** on top of BiasFinder BERT embeddings (Z) for distribution-shift tests?

Config: B=4, N_perm=40, n_doc=80, PCA(Z)=16, RF n_estimators≈30, α=0.05.

Feature sets: `embed` = Z_PCA only; `embed_meta` = [Z_PCA, M].

## Power / Type-I

| scenario | set | RFPerm | DRPerm | CFPerm |
| --- | --- | ---: | ---: | ---: |
| null_shift (typeI) | embed | 0.25 | 0.25 | 0.00 |
| null_shift (typeI) | embed_meta | 0.25 | 0.00 | 0.00 |
| sentiment_meta (power) | embed | 0.00 | 0.00 | 0.00 |
| sentiment_meta (power) | embed_meta | 0.00 | 0.00 | 0.00 |
| train_vs_heldout (power) | embed | 1.00 | 1.00 | 0.00 |
| train_vs_heldout (power) | embed_meta | 1.00 | 1.00 | 0.00 |

## Δ power (embed_meta − embed)

- `null_shift`: RFPerm +0.00, DRPerm -0.25, CFPerm +0.00
- `sentiment_meta`: RFPerm +0.00, DRPerm +0.00, CFPerm +0.00
- `train_vs_heldout`: RFPerm +0.00, DRPerm +0.00, CFPerm +0.00

## PO-risk interpretation (DRPerm)

DRPerm PO-risk = E[τ(X)^2], τ≈(Y−μ(X))(W−e(X)). If adding M shrinks residual batch–outcome dependence, DRPerm power may drop (better nuisance adjustment). If M exposes concept signal that Z missed, power rises. Both are scientifically informative for incremental value.

## CFPerm block note

CFPerm here tests the **meta block** under `embed_meta` (LOCO R-risk drop). Under `embed` it tests a trailing PCA placebo block of similar size. A positive Δ on CFPerm is direct evidence that M carries shift-localizing signal beyond Z.
