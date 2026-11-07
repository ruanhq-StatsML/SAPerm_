# Cross-fit: magnitude × Var(tau) against magnitude alone

Five-fold cross-fit, stratified on the batch indicator. Outcome, propensity, and the CATE model are fit on the training fold. The held-out fold supplies the residual magnitude `r = (phi - tau)^2` and `Var(tau)` (jackknife for the pseudo-outcome forest, squared standard error for the causal forest and the orthogonal forest). The product is `r * Var(tau)`. There is no exponent on `r`.

`spearman` is positive when larger scores sit on larger OOD. `formula` is `spearman(argsort(-score), arange)`. With no ties that number is the negation, so a successful ranking is negative.

Grid: `n = 800`, `p = 20`, shifts `{0,1,2,3,4,5}`, two seeds, synthetic Gaussian draws and whitened BiasFinder components. Tree counts 80 / 60 / 24.

## What the comparison says

Means over shifts 1–5 and both seeds. The target is the realized perturbation.

| Data | Shift | Model | Magnitude | mag × var | Var(tau) alone | formula(mag × var) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Synthetic | Concept | PO | 0.55 | 0.55 | 0.17 | −0.35 |
| Synthetic | Concept | CF | 0.51 | 0.45 | −0.10 | −0.32 |
| Synthetic | Concept | orf | 0.42 | 0.48 | 0.36 | −0.16 |
| Synthetic | Covariate | PO | 0.50 | 0.71 | 0.89 | −0.40 |
| Synthetic | Covariate | CF | 0.93 | 0.91 | 0.12 | −0.51 |
| Synthetic | Covariate | orf | 0.93 | 0.86 | 0.20 | −0.49 |
| BiasFinder PCs | Concept | PO | 0.64 | 0.66 | 0.50 | −0.20 |
| BiasFinder PCs | Concept | CF | 0.62 | 0.63 | 0.39 | −0.19 |
| BiasFinder PCs | Concept | orf | 0.55 | 0.59 | 0.44 | −0.13 |
| BiasFinder PCs | Covariate | PO | 0.43 | 0.60 | 0.74 | −0.33 |
| BiasFinder PCs | Covariate | CF | 0.87 | 0.84 | 0.15 | −0.51 |
| BiasFinder PCs | Covariate | orf | 0.87 | 0.80 | 0.31 | −0.45 |

Shift 0, where nothing is planted, gives `formula` between about −0.05 and 0.03.

The product beats the magnitude in one place: the pseudo-outcome forest under covariate shift (+0.21 synthetic, +0.16 on the embeddings, realized Spearman). There the pure variance is stronger than the product (0.89 and 0.74). Cross-fitting is what makes that variance usable. The in-sample jackknife on the same design only reached about 0.53 against the ramp.

Everywhere else the magnitude is the ranking, and multiplying by `Var(tau)` leaves it alone or pulls it down. Causal-forest and orthogonal-forest residuals already track realized covariate shift at 0.87–0.93. Their `Var(tau)` does not (0.12–0.31), so the product is a diluted residual. Under concept drift the covariates do not move, `Var(tau)` stays weak, and the product sits on top of the magnitude.

`formula(mag × var)` for the pseudo-outcome forest on synthetic covariate shift is about −0.40. That is the aligned ranking, not a reversal. The same cell's ordinary Spearman against the ramp index is +0.40, and against the realized displacement is +0.71.
