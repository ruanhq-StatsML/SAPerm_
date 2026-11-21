# Embedding ± Meta-feature power (RFPerm / DRPerm / CFPerm)

Question: do spaCy+VADER meta-features (M) give **incremental value** on top of BiasFinder BERT embeddings (Z) for distribution-shift tests?

Config: B=150, N_perm=120, n_doc=120, PCA(Z)=24, RF n_estimators≈40, α=0.05, meta_dim=37, elapsed≈1.44h.

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

## Findings (what the deltas mean)

1. **Type-I is controlled.** Null rejection stays near α for all three methods on both feature sets (DRPerm ≈0.07 is slightly liberal but usable). Adding M does not break calibration.

2. **`sentiment_meta` is pure covariate shift in VADER polarity.**  
   - DRPerm power stays **0.00** on both sets — expected: once X is rich enough (or even with Z alone here), residual batch→Y dependence is gone, so PO-risk does not alarm.  
   - Mean PO-risk statistic still **shrinks** when M is added: 0.0054 → 0.0014, and mean p moves 0.95 → 1.00. That is incremental value as *nuisance adjustment*, not as a louder detector.  
   - RFPerm picks up a small bump (+0.06: 0.02→0.08): M makes the predictive-risk gap slightly sharper under sentiment routing.

3. **`train_vs_heldout` is a strong real shift.**  
   - RFPerm and DRPerm are already at the ceiling (power 1.00) with Z alone — meta cannot raise detection power further.  
   - **CFPerm is where meta earns its keep:** placebo trailing-PCA block power 0.77 → meta-block power 0.99 (**Δ +0.23**). Mean LOCO importance 0.029 → 0.060. M carries shift-localizing signal beyond the embedding PCA subspace.

## PO-risk interpretation (DRPerm)

PO-risk = E[τ(X)²] with τ ≈ (Y−μ(X))(W−e(X)).

- Large PO-risk ⇒ batch still predicts residual outcome after X-adjustment ⇒ concept / label-channel shift.  
- Small PO-risk after adding M ⇒ M absorbed the covariate channel (better μ, e) ⇒ detection power can *fall* even though M is useful.  
- In this BiasFinder run: sentiment routing is absorbed (PO-risk↓, power flat at 0); heldout split remains a concept-level alarm (power 1.00 either way). Use CFPerm, not DRPerm Δ-power alone, to argue incremental *localization* value of M.

## CFPerm block note

CFPerm tests the **meta block** under `embed_meta` (LOCO: R(−M)−R(full) with R=E[(μ₁−μ₀)²]). Under `embed` it tests a trailing PCA placebo block of similar size. The +0.23 Δ on `train_vs_heldout` is the cleanest evidence that M adds localizing value on top of Z.
