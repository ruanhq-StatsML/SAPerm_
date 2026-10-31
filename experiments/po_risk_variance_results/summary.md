# PO-risk variance on the two shift generators

Each batch has 1000 rows when `n = 2000`. Concept drift keeps P(X) fixed and adds `X @ beta` on the second batch, with `beta[:4] = (0, delta, 2 delta, 3 delta)`. Covariate shift keeps the regression fixed and moves the first batch's first four columns. The population CATE is that linear drift, and it is identically zero under covariate shift.

Holdout PO-risk is `mean((phi - tau)^2)` on a treatment-stratified half. `mean_var_tau` is the average estimated variance of the CATE. `mean_delta_var` is the average of `4 (phi - tau)^2 Var(tau)`. `plugin_se` is the ordinary standard error of the mean squared residual. `se_indep` folds the delta-method variance into a standard error of the mean under an independence approximation. BLB's `se_indep` is instead the little-bootstrap standard error of the scalar training risk.

## Means across seeds

| kind | strength | split | method | po_risk | mean_var_tau | mean_delta_var | plugin_se | se_indep | tau_mse | tau_corr |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| concept | 0 | holdout | CF | 60.931 | 0.012 | 2.777 | 3.092 | 0.053 | 0.009 |  |
| concept | 0 | holdout | RF_IJ | 61.553 | 1.917 | 487.552 | 3.101 | 0.697 | 0.704 |  |
| concept | 0 | holdout | RF_delta | 61.553 | 0.071 | 18.466 | 3.101 | 0.136 | 0.704 |  |
| concept | 0 | insample | BLB | 40.884 |  |  |  | 0.565 |  |  |
| concept | 0 | insample | CF | 48.043 | 0.012 | 2.268 | 2.415 | 0.048 | 0.009 |  |
| concept | 0 | insample | RF_IJ | 37.158 | 2.140 | 488.717 | 1.909 | 0.696 | 1.371 |  |
| concept | 0 | insample | RF_delta | 37.158 | 0.066 | 10.843 | 1.909 | 0.104 | 1.371 |  |
| concept | 1 | holdout | CF | 80.275 | 0.260 | 81.092 | 4.317 | 0.283 | 3.106 | 0.911 |
| concept | 1 | holdout | RF_IJ | 80.169 | 2.961 | 1047.664 | 4.293 | 1.022 | 3.288 | 0.896 |
| concept | 1 | holdout | RF_delta | 80.169 | 0.106 | 36.113 | 4.293 | 0.190 | 3.288 | 0.896 |
| concept | 1 | insample | BLB | 53.703 |  |  |  | 0.857 |  |  |
| concept | 1 | insample | CF | 62.520 | 0.260 | 63.625 | 3.248 | 0.252 | 2.857 | 0.915 |
| concept | 1 | insample | RF_IJ | 48.872 | 3.090 | 866.592 | 2.605 | 0.928 | 3.476 | 0.870 |
| concept | 1 | insample | RF_delta | 48.872 | 0.096 | 20.837 | 2.605 | 0.144 | 3.476 | 0.870 |
| covariate | 0 | holdout | CF | 4.178 | 0.011 | 0.190 | 0.195 | 0.014 | 0.009 |  |
| covariate | 0 | holdout | RF_IJ | 4.237 | 0.137 | 2.309 | 0.198 | 0.048 | 0.059 |  |
| covariate | 0 | holdout | RF_delta | 4.237 | 0.005 | 0.089 | 0.198 | 0.009 | 0.059 |  |
| covariate | 0 | insample | BLB | 3.016 |  |  |  | 0.039 |  |  |
| covariate | 0 | insample | CF | 3.489 | 0.012 | 0.171 | 0.164 | 0.013 | 0.009 |  |
| covariate | 0 | insample | RF_IJ | 2.739 | 0.153 | 2.301 | 0.132 | 0.048 | 0.105 |  |
| covariate | 0 | insample | RF_delta | 2.739 | 0.005 | 0.056 | 0.132 | 0.008 | 0.105 |  |
| covariate | 1 | holdout | CF | 60.580 | 0.012 | 2.844 | 6.254 | 0.053 | 0.010 |  |
| covariate | 1 | holdout | RF_IJ | 14.493 | 1.534 | 230.810 | 2.111 | 0.448 | 32.461 |  |
| covariate | 1 | holdout | RF_delta | 14.493 | 0.051 | 4.903 | 2.111 | 0.069 | 32.461 |  |
| covariate | 1 | insample | BLB | 13.335 |  |  |  | 0.947 |  |  |
| covariate | 1 | insample | CF | 53.735 | 0.012 | 2.606 | 7.939 | 0.051 | 0.010 |  |
| covariate | 1 | insample | RF_IJ | 11.997 | 1.343 | 515.390 | 3.918 | 0.518 | 30.979 |  |
| covariate | 1 | insample | RF_delta | 11.997 | 0.043 | 5.789 | 3.918 | 0.068 | 30.979 |  |

## Paired shift minus null

The null is strength 0 drawn with the same seed. A positive PO-risk gap means the squared pseudo-outcome residual got larger. A positive `mean_var_tau` gap means the CATE estimate got less stable.

| kind | split | method | d_po_risk | seeds po up | d_var_tau | seeds var up | d_plugin_se | null po | alt po |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| concept | holdout | CF | 19.344 | 5/5 | 0.249 | 5/5 | 1.225 | 60.931 | 80.275 |
| concept | holdout | RF_IJ | 18.617 | 5/5 | 1.044 | 5/5 | 1.192 | 61.553 | 80.169 |
| concept | holdout | RF_delta | 18.617 | 5/5 | 0.035 | 5/5 | 1.192 | 61.553 | 80.169 |
| concept | insample | BLB | 12.819 | 5/5 |  | 0/0 |  | 40.884 | 53.703 |
| concept | insample | CF | 14.477 | 5/5 | 0.248 | 5/5 | 0.833 | 48.043 | 62.520 |
| concept | insample | RF_IJ | 11.713 | 5/5 | 0.951 | 5/5 | 0.695 | 37.158 | 48.872 |
| concept | insample | RF_delta | 11.713 | 5/5 | 0.030 | 5/5 | 0.695 | 37.158 | 48.872 |
| covariate | holdout | CF | 56.403 | 5/5 | 0.000 | 4/5 | 6.060 | 4.178 | 60.580 |
| covariate | holdout | RF_IJ | 10.256 | 5/5 | 1.397 | 5/5 | 1.913 | 4.237 | 14.493 |
| covariate | holdout | RF_delta | 10.256 | 5/5 | 0.046 | 5/5 | 1.913 | 4.237 | 14.493 |
| covariate | insample | BLB | 10.319 | 5/5 |  | 0/0 |  | 3.016 | 13.335 |
| covariate | insample | CF | 50.246 | 5/5 | 0.000 | 3/5 | 7.775 | 3.489 | 53.735 |
| covariate | insample | RF_IJ | 9.258 | 5/5 | 1.190 | 5/5 | 3.787 | 2.739 | 11.997 |
| covariate | insample | RF_delta | 9.258 | 5/5 | 0.039 | 5/5 | 3.787 | 2.739 | 11.997 |

## Propensity weight on the holdout

Under a perfect overlap of one half, `mean((T - pi) / (pi (1 - pi)))^2` equals 4. Covariate shift pushes mass toward the clip, and the pseudo-outcome gets heavier even when the CATE is zero.

| kind | strength | mean weight^2 | clip rate |
| --- | ---: | ---: | ---: |
| concept | 0 | 4.219 | 0.000 |
| concept | 1 | 4.219 | 0.000 |
| covariate | 0 | 4.219 | 0.000 |
| covariate | 1 | 4.231 | 0.000 |

## Feature permutation of the holdout PO-risk

One column of the holdout covariates is permuted and `tau` is recomputed. `phi` is left as fitted. Concept drift's nonzero CATE coordinates are 1, 2 and 3 (coordinate 0 has coefficient 0). Covariate shift moves columns 0 through 3, but the population CATE is zero, so those columns should not be required to predict `tau`.

| kind | strength | vimp planted | vimp rest | top 4 by seed |
| --- | ---: | ---: | ---: | --- |
| concept | 0 | -0.033 | 0.017 | s0: 5 11 4 17; s1: 15 13 8 16; s2: 15 4 12 13; s3: 0 13 8 10; s4: 18 15 12 5 |
| concept | 1 | 6.034 | 0.009 | s0: 3 2 1 11; s1: 3 2 1 4; s2: 3 2 1 9; s3: 3 2 1 13; s4: 3 2 1 18 |
| covariate | 0 | -0.002 | -0.001 | s0: 13 16 9 18; s1: 14 17 1 19; s2: 5 13 10 11; s3: 5 13 0 19; s4: 10 7 19 15 |
| covariate | 1 | 8.550 | 0.001 | s0: 3 1 0 2; s1: 1 3 2 0; s2: 0 2 1 3; s3: 0 3 2 1; s4: 0 1 2 3 |

Oracle scale: with a perfect regression, a balanced batch indicator and residual variance 1, `(T - 1/2) / (1/4)` equals ±2, so the PO-risk equals 4. `mean_weight_sq` above 4 means the propensity is no longer one half and the pseudo-outcome is heavier.

## What separates the two shifts

The concept-drift null is not an oracle risk of 4. Its outcome is `Y = X @ 1 + ε` on every column, and the random-forest outcome fit leaves a large pseudo-outcome remainder. The covariate-shift null is `Y = ε`, and that holdout risk sits next to 4. The shift in this covariate generator changes second moments, not means, so a linear propensity stays near one half: `mean weight^2` does not leave 4 and nothing is clipped. A rise in the covariate PO-risk is leftover nuisance error, not an inverse-propensity explosion.

Causal-forest `tau` is scored against the same random-forest pseudo-outcome as the other rows. `tau_mse` is the error against the population CATE, which is zero under covariate shift and `X @ (0, δ, 2δ, 3δ)` under concept drift.

| kind | method | null risk (sd) | alt risk (sd) | gap (sd) | null Var(tau) | alt Var(tau) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| concept | RF_delta | 61.553 (5.917) | 80.169 (6.610) | 18.617 (3.377) | 0.071 | 0.106 |
| concept | RF_IJ | 61.553 (5.917) | 80.169 (6.610) | 18.617 (3.377) | 1.917 | 2.961 |
| concept | CF | 60.931 (6.326) | 80.275 (6.989) | 19.344 (2.986) | 0.012 | 0.260 |
| covariate | RF_delta | 4.237 (0.272) | 14.493 (3.715) | 10.256 (3.516) | 0.005 | 0.051 |
| covariate | RF_IJ | 4.237 (0.272) | 14.493 (3.715) | 10.256 (3.516) | 0.137 | 1.534 |
| covariate | CF | 4.178 (0.284) | 60.580 (8.879) | 56.403 (8.728) | 0.011 | 0.012 |

Permutation of the holdout pseudo-outcome forest ranks columns 3, 2, 1 first in 5/5 concept-drift replicates. Those are the nonzero drift slopes, in the same order as `(3δ, 2δ, δ)`. Column 0 has slope 0 and does not enter that top three. Under covariate shift the same permutation puts columns 0–3 in the top four in 5/5 replicates. The population CATE there is zero; the random-forest `tau` still uses the shifted columns, while the causal forest's `tau_mse` stays at the null level.

The level of the PO-risk moves under both shifts. The causal-forest variance moves under concept drift and stays flat under covariate shift, which is the split between a real CATE change and a pure change in P(X). The jackknife and the tree-variance delta method move under both, because they are variances of the random-forest pseudo-outcome fit, and that fit gets less stable when the two arms separate. On these holdout rows the Wager correction floors 37% of the jackknife points at zero and the raw jackknife is about 35.8 times the corrected one. The corrected values are larger on the shifted draw in 10/10 paired replicates. `se_indep` is much smaller than the seed-to-seed standard deviation of the risk, because it only propagates `Var(tau)` and treats sample contributions as independent. The little-bootstrap standard error is the refit noise of the training risk, not that cross-seed spread.

