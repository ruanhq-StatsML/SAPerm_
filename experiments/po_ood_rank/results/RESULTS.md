# Does the PO-risk variance recover a planted OOD ranking?

The observation-level score is `r ** mag_power * Var(r)`, where `r = (phi - tau)^2` is the doubly robust pseudo-outcome residual and `Var(r) = 4 * r * Var(tau)` is the delta-method variance. `Var(tau)` comes from the infinitesimal jackknife of a pseudo-outcome forest (`PO`), the squared standard error of a causal forest (`CF`), or an orthogonal forest (`orf`). Each model uses its own `tau`, so the residual and the variance refer to the same fit.

The second half of each sample is a shifted batch whose perturbation is multiplied by a ramp from 0 to the shift level. Rank agreement is Spearman correlation on that half only. Two targets are reported:

- the designed ramp, which is the within-batch row order
- the realized magnitude, `|intervention|` for concept drift and the Euclidean displacement of `X` for covariate shift

A shift of 0 is a negative control: the ramp is identically zero, and any correlation with row order is spurious. The grid is shift levels `{0,1,2,3,4,5}`, magnitude powers `linspace(0, 2, 9)`, two seeds, `n = 1000`, `p = 20`. The semi-synthetic rows use a seeded subsample of the BiasFinder BERT embeddings, whitened to 20 principal components, with the same planted mechanisms.

## The ranking aligns

On every shifted batch the score orders observations in the same direction as the planted OOD magnitude. The zero-shift cells sit near zero (Spearman with row order between about `-0.09` and `0.02`), so the correlation is not an artifact of the index. Agreement then rises with the shift level and levels off by shift 3 or 4.

The realized magnitude is the cleaner target. The ramp is only a multiplier; the actual intervention still depends on `X`. Causal-forest and orthogonal-forest scores track that realized covariate shift very closely. Concept drift is weaker, and positive.

Numbers below are means over shifts 1 through 5 and both seeds, at magnitude power 1. "Residual" is Spearman of `r` alone, with no variance weight.

| Data | Shift | Model | Score vs ramp | Residual vs ramp | Score vs realized | Residual vs realized | AUROC, upper half of realized |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | Concept | PO | 0.33 | 0.34 | 0.55 | 0.55 | 0.76 |
| Synthetic | Concept | CF | 0.28 | 0.29 | 0.48 | 0.49 | 0.72 |
| Synthetic | Concept | orf | 0.22 | 0.23 | 0.48 | 0.44 | 0.73 |
| Synthetic | Covariate | PO | 0.40 | 0.33 | 0.64 | 0.52 | 0.79 |
| Synthetic | Covariate | CF | 0.56 | 0.56 | 0.94 | 0.94 | 0.97 |
| Synthetic | Covariate | orf | 0.55 | 0.56 | 0.93 | 0.95 | 0.96 |
| BiasFinder PCs | Concept | PO | 0.25 | 0.29 | 0.63 | 0.61 | 0.81 |
| BiasFinder PCs | Concept | CF | 0.21 | 0.23 | 0.58 | 0.56 | 0.77 |
| BiasFinder PCs | Concept | orf | 0.17 | 0.19 | 0.55 | 0.50 | 0.76 |
| BiasFinder PCs | Covariate | PO | 0.25 | 0.20 | 0.55 | 0.45 | 0.75 |
| BiasFinder PCs | Covariate | CF | 0.46 | 0.47 | 0.90 | 0.90 | 0.95 |
| BiasFinder PCs | Covariate | orf | 0.45 | 0.47 | 0.88 | 0.90 | 0.94 |

Seed-to-seed standard deviations of these shift-averages are typically 0.01–0.04. The exception is the concept-drift ramp correlation on the embeddings, where one causal-forest seed is 0.30 and the other is 0.12. The realized-magnitude correlation on that same split stays at 0.60 and 0.55. The magnitude ranking is the stable one.

## The variance weight helps in one place

Changing the exponent on `r` from 0 to 2 barely moves the rank correlation. The curves in `rank_ood_spearman.png` and `rank_ood_spearman_realized.png` are flat. Up-weighting large residuals is not what aligns the ranking.

`Var(tau)` itself is uncorrelated with the concept-drift ramp (Spearman about 0, and slightly negative for the causal forest). The covariates do not move, so a forest's pointwise variance has nothing to order. The residual carries that ranking, and multiplying by the variance does not improve it.

Under covariate shift the pseudo-outcome forest is different. Its `Var(tau)` correlates with the ramp (0.53 on Gaussian covariates, 0.35 on the embeddings), because the shifted points leave the bulk of the sample. Ranking by the delta-method variance, which is magnitude power 0, beats the residual:

| Data | Score vs ramp | Residual vs ramp | Lift | Score vs realized | Residual vs realized | Lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 0.44 | 0.33 | +0.11 | 0.72 | 0.52 | +0.21 |
| BiasFinder PCs | 0.28 | 0.20 | +0.08 | 0.61 | 0.45 | +0.17 |

Higher powers pull this score back toward the residual and give back part of the lift. For the causal forest and the orthogonal forest the residual already matches the realized covariate shift (Spearman 0.89–0.94), and their `Var(tau)` is only weakly ordered by the ramp (0.09–0.19). The product does not add agreement those residuals do not already have.

## Semi-synthetic

The embedding experiment repeats the Gaussian pattern rather than breaking it. Concept drift on the principal components is a bit noisier on the nominal ramp, especially at shift 1, where the ramp correlation is about 0. Realized-magnitude correlation is if anything stronger than on the Gaussian concept drift (PO 0.63 against 0.55). Covariate shift on the embeddings is slightly harder for the pseudo-outcome forest and still easy for the two forests: at shift 2 and above, causal-forest Spearman with the realized displacement is 0.94–0.97 and the upper-half AUROC is about 0.98. OOD magnitude does characterize the ranking on this table.

## Setup notes

The pasted loop indexed a scalar mean risk, correlated `argsort` of the score with the row index (a second ranking, with the sign flipped), and applied the covariate ramp on the first half while reading the second. The covariate generator also referenced `Y` before it was drawn. The runner fixes those and evaluates both batches on the half that actually moves.

Fits are in-sample. Nuisance outcome models are random forests and the propensity is logistic, with probabilities clipped to `[0.01, 0.99]`. Tree counts are 120 (pseudo-outcome), 80 (causal forest), and 40 (orthogonal forest), with minimum leaf size 20. That is smaller than the sketched `n = 2000`, `p = 30`, 150 trees; the orthogonal-forest bootstrap standard errors dominate the runtime. The sign and the shape of the curves are consistent across both seeds and both data sources.

## Nonlinear concept drift

The linear concept drift above moves four coefficients by a ramp. The harder design redraws the second domain's sine, square, and interaction term and multiplies that term by `linspace(0, 5)`. `delta_beta` only rescales the nonlinear function, from `0.5` to `0.5 + delta_beta`. Covariates stay on the same AR(1) law, so a forest cannot see the ramp in `X`. The realized target is `|drift × ramp|`. The planted target is the row-order ramp itself.

This grid is synthetic only: `n = 2000` (1000 per domain), `p = 20`, shifts `{0,1,2,3,4,5}`, two seeds. The residual `r = (phi - tau)^2` is one five-fold held-out PO fit, shared by every curve, so the dashed magnitude line is the same for PO, CF, and orf. `Var(tau)` is that fit's out-of-fold jackknife for PO, and one full-sample standard-error squared for CF and orf. The score is still `r^power * Var(r)` with `Var(r) = 4 r Var(tau)`. Powers run from 0 to 2. Numbers below average shifts 1–5.

| Model | Power 0 vs realized | Power 2 vs realized | vs planted ramp | `Var(tau)` vs ramp |
| --- | ---: | ---: | ---: | ---: |
| PO | 0.30 | 0.22 | −0.02 | 0.03 |
| CF | 0.34 | 0.24 | −0.02 | 0.03 |
| orf | 0.31 | 0.23 | 0.00 | 0.09 |
| magnitude only | 0.17 | 0.17 | −0.01 |  |

The planted row order is not recovered. Spearman of the score against the ramp sits on zero at every power, and `Var(tau)` against that ramp is 0.03 for PO and CF, 0.09 for orf. The rows are iid; the ramp is an index, and the nonlinear coefficients are redrawn, so the index is not a function of `X`.

The realized size of the nonlinear term is only weakly ordered. Power 0 is the top of a low curve: CF 0.34, orf 0.31, PO 0.30, against a residual of 0.17. Raising the power pulls every curve back toward that residual, down to 0.22–0.24 at power 2. The seed band on the shift average is about 0.02 at power 0 and about 0.02–0.05 at power 2. Raising `delta_beta` from 1 to 5 moves the power-0 CF correlation from 0.32 to 0.36.

On the linear concept drift the same realized Spearman sits near 0.55 (PO) and 0.47 (CF, orf), flat on the residual. The nonlinear term keeps a small variance lift at power 0 and loses it as the exponent grows. The figure is `nonlinear_concept_power.png`.
