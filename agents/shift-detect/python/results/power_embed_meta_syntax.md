# Embedding ± Meta-feature power (RFPerm / DRPerm / CFPerm)

Question: do spaCy+VADER meta-features (M) give **incremental value** on top of BiasFinder BERT embeddings (Z) for distribution-shift tests?

Config: B=150, N_perm=120, n_doc=120, PCA(Z)=24, RF n_estimators≈40, α=0.05.

Feature sets: `embed` = Z_PCA only; `embed_meta` = [Z_PCA, M].

## Power / Type-I

| scenario | set | RFPerm | DRPerm | CFPerm |
| --- | --- | ---: | ---: | ---: |
| syntax_meta (power) | embed | 0.11 | 0.00 | 0.21 |
| syntax_meta (power) | embed_meta | 0.07 | 0.00 | 0.31 |

## Δ power (embed_meta − embed)

- `syntax_meta`: RFPerm -0.04, DRPerm +0.00, CFPerm +0.09

## PO-risk interpretation (DRPerm)

DRPerm PO-risk = E[τ(X)^2], τ≈(Y−μ(X))(W−e(X)). If adding M shrinks residual batch–outcome dependence, DRPerm power may drop (better nuisance adjustment). If M exposes concept signal that Z missed, power rises. Both are scientifically informative for incremental value.

Reading guide for ±meta deltas:

1. **Covariate shift absorbed by M** (e.g. `sentiment_meta`): adding M should *lower* DRPerm power if μ(X), e(X) become better nuisances — residual (Y−μ)(W−e) collapses. That is incremental value as *adjustment*, not as a louder alarm.
2. **Concept / label-channel shift**: if batches differ in P(Y|X) even after M, DRPerm power stays high on both feature sets; CFPerm then localizes which block of X mediates the residual batch effect.
3. **RFPerm** responds to predictive-risk gaps (OOB vs new). Meta can raise power when M makes the risk gap sharper, or lower it when M stabilizes extrapolation across batches.
4. **CFPerm LOCO** on the meta block: importance = R(−M) − R(full) with R = E[(μ₁−μ₀)²]. Positive Δ power vs the embed placebo block ⇒ M carries shift-localizing signal beyond Z.

## CFPerm block note

CFPerm here tests the **meta block** under `embed_meta` (LOCO R-risk). Under `embed` it tests a trailing PCA placebo block of similar size. A positive Δ on CFPerm is direct evidence that M carries shift-localizing signal beyond Z.
