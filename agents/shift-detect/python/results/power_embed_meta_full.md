# Embedding ± Meta-feature power (RFPerm / DRPerm / CFPerm)

Question: do spaCy+VADER meta-features (M) give **incremental value** on top of BiasFinder BERT embeddings (Z) for distribution-shift tests?

Config: B=150, N_perm=120, n_doc=120, PCA(Z)=24, RF n_estimators≈40, α=0.05.

Feature sets: `embed` = Z_PCA only; `embed_meta` = [Z_PCA, M].

## Power / Type-I

| scenario | set | RFPerm | DRPerm | CFPerm |
| --- | --- | ---: | ---: | ---: |
| null_shift (typeI) | embed | 0.03 | 0.07 | 0.03 |
| null_shift (typeI) | embed_meta | 0.03 | 0.07 | 0.06 |
| sentiment_meta (power) | embed | 0.02 | 0.00 | 0.02 |
| sentiment_meta (power) | embed_meta | 0.08 | 0.00 | 0.03 |
| train_vs_heldout (power) | embed | 1.00 | 1.00 | 0.77 |
| train_vs_heldout (power) | embed_meta | 1.00 | 1.00 | 0.99 |

## Δ power (embed_meta − embed)

- `null_shift`: RFPerm +0.00, DRPerm +0.00, CFPerm +0.03
- `sentiment_meta`: RFPerm +0.06, DRPerm +0.00, CFPerm +0.01
- `train_vs_heldout`: RFPerm +0.00, DRPerm +0.00, CFPerm +0.23

## PO-risk interpretation (DRPerm)

DRPerm PO-risk = E[τ(X)^2], τ≈(Y−μ(X))(W−e(X)). If adding M shrinks residual batch–outcome dependence, DRPerm power may drop (better nuisance adjustment). If M exposes concept signal that Z missed, power rises. Both are scientifically informative for incremental value.

## CFPerm block note

CFPerm here tests the **meta block** under `embed_meta` (LOCO R-risk drop). Under `embed` it tests a trailing PCA placebo block of similar size. A positive Δ on CFPerm is direct evidence that M carries shift-localizing signal beyond Z.
