---
name: fsds-ltmle
description: Use FSDS two-arm variable importance to monitor a longitudinal TMLE incentive policy and change truncation or Super Learner weights without dropping confounders.
---

# FSDS for longitudinal TMLE

Use this when a dynamic incentive (coupon, subsidy, push, price) is evaluated with longitudinal targeted maximum likelihood estimation and a new batch may have moved the covariates or the response to the incentive. FSDS does not replace the TMLE update. It names which monitoring feature moved, which nuisance that threatens, and which knob to turn.

The implementation in this package is `experiments.fsds_ltmle`.

## What is identified

For a batch of trajectories `(L_t, A_t)` and a terminal outcome `Y`, the estimand is

```text
psi = E[Y^{a=1}] - E[Y^{a=0}]
```

`a=1` means the incentive is offered at every decision point. `a=0` means it is never offered. `L_t` is measured before `A_t`. The estimator is a stabilized longitudinal TMLE: at each time `t`, going backward, a Super Learner fits `E[Q_{t+1} | history through A_t]`, then a one-dimensional fluctuation uses the current-step clever covariate

```text
H_t = 1{A_t = a*_t} / g_t(a*_t | history before A_t)
```

with `g_t` clipped to `[truncation, 1 - truncation]`. The cumulative-product clever covariate is the efficient influence-function weight. It is not used for the point estimate, because a channel shift pushes those products into a few observations.

The Super Learner library is ridge and a random forest. Ridge sees the linear blip terms `A * redemption` and `A * severity`. It does not see `A * redemption^2`.

## Two arms, two nuisances

Compare an existing batch with a new batch. The domain label `W` is 0 for the existing batch and 1 for the new batch. `W` is not the incentive `A`.

### Covariate arm, which watches `g`

Fit a random-forest domain classifier on the monitoring matrix. Importance of a column is the increase in out-of-bag Brier score when that column is permuted inside each tree. A large value means the marginal picture of that column changed. That is the failure mode in which the propensity for the incentive moves toward 0 or 1 and the TMLE weights get large.

Code: `covariate_arm.rf_domain_vimp`.

### Concept arm, which watches `Q`

This is PermuCATE, following `permuCATE_vimp.py` in the FSDS repository.

1. Split the stacked batches in half, stratified on `W`.
2. Fit `mu0` and `mu1` with `RandomForestRegressor` on `W = 0` and `W = 1`.
3. Fit a logistic domain propensity and clip it.
4. Build the doubly robust pseudo-outcome

```text
phi = (Y - mu_W) * (W - pi) / (pi * (1 - pi)) + (mu1 - mu0)
```

5. Fit `tau` with `RandomForestRegressor` on `phi`.
6. For every column, fit `X_j | X_{-j}` with `RandomForestRegressor`. This conditional model is a forest for every column, including binary ones.
7. On the held-out half, permute the residual `X_j - nu_j(X_{-j})`, recompute `tau`, and record the increase in `E[(phi - tau)^2]`.

The outcome model and the conditional model are both random forests. Do not swap either of them for a linear model if the goal is to match this skill.

Code: `permucate.permu_cate_vimp` and `permucate.po_risk`.

PO-risk is the loss of that domain CATE. It is not the efficient influence function of the longitudinal incentive contrast. It is used here because a change in `P(Y | monitoring features, batch)` is the change that leaves the incentive outcome regression misspecified.

## What the controller is allowed to change

`controller.decide_action` compares each arm's peak with a null peak from two stable-versus-stable contrasts. The null peak is the larger of the two, so one quiet pair cannot raise an alarm. An arm fires when the new peak is at least five times the null peak:

```text
(peak - null_peak) / max(|null_peak|, 0.001) >= 5
```

A smaller gate treats a permutation wobble, such as a concept peak of 0.03 against a stable peak of 0.02, as drift. That wobble puts the forest in charge of a still-linear blip and increases the error. The drifts this panel actually plants move the peak by more than an order of magnitude.

The covariate arm also fires when the share of fitted propensities inside `(0.1, 0.9)` drops below 0.50.

| State | Threatened piece | Automatic change |
| --- | --- | --- |
| GREEN | neither | truncation stays 0.02, ridge penalty stays 0 |
| ORANGE_COVARIATE | propensity `g` | truncation moves to 0.10 |
| ORANGE_CONCEPT | outcome regression `Q` | ridge CV score is multiplied by `(1 + penalty * ridge weight)`, penalty = min(3, 0.75 * concept excess ratio) |
| RED | both | truncation moves to 0.12, the same penalty is applied, and `pause_dynamic_incentives` is set |

A red light means the dynamic rule should not be updated automatically. A person confirms the next step. The TMLE number is still computed so it can be looked at. It is not an authorization to ship a new policy.

Every action returns the adjustment set it was given and an empty dropped-feature list. A VIMP of zero is not evidence that a confounder can be removed. A very large VIMP is not evidence that the feature can be removed either. Severity, channel, and redemption stay in both `g` and `Q` because each sits on a path into the incentive, the outcome, or both.

The rankings are a review order. Without a causal graph that is strong enough to separate features, a global shift does not have a unique feature cause. Send the top of the list to a person. Do not treat it as an attribution verdict.

## How to read a rank in the incentive panel

The monitoring row for one user is the mean and the last value of severity, channel, redemption, and the two structural noise coordinates, the fraction of periods that were treated, and six pure-noise columns.

- Channel mean or channel last at the top of the covariate arm: the mix of traffic changed. Check the feed that builds channel before retuning the outcome model. Truncation is the TMLE response.
- Treat rate at the top of the concept arm: the association between being offered the incentive and `Y` changed. That is the blip moving. It is expected when the average effect changes sign. It is not, by itself, a reason to drop the exposure.
- Redemption mean or redemption last next to treat rate on the concept arm: the modifier of the blip moved. The forest in the Super Learner is the library member that can represent that nonlinearity. Ridge cannot.
- A pure-noise column at the top: do not act on it. Compare with the null peak. A single permutation draw moves those scores.

## Datasets

### Longitudinal incentive panel (the benchmark that is actually hard)

`dgp.simulate` draws the panel. The truth used in the report is `dgp.monte_carlo_ate`, which resimulates severity under always-treat and never-treat. Public logs do not provide that contrast, so this is the panel on which the estimator is scored.

What makes it longitudinal rather than a shifted table:

- Three decision points.
- Severity affects the probability of the incentive and the terminal outcome, and the incentive lowers the next severity. A regression that ignores that feedback is not the g-formula.
- The propensity logit contains `severity^2`, so a linear logistic `g` is not the true propensity.
- Covariate regime: channel's stationary mean jumps by 1.15. The blip is unchanged. The true contrast should move only by Monte Carlo error. The fitted propensity piles up above 0.9.
- Concept regime: the blip changes from `0.45 + 0.80 * redemption` to `-0.70 + 0.15 * redemption + 0.90 * redemption^2`. The covariate law is unchanged. The average incentive effect changes sign.
- Joint regime: both happen together.
- Covariates are correlated at lag one with correlation 0.45. The monitoring matrix adds six irrelevant Gaussian columns.

Regimes are `stable`, `covariate`, `concept`, and `joint`.

### A real batch an engineer already has

Build one dict per batch:

```python
batch = {"L": L, "A": A, "Y": Y}
```

`L` is float `(n, T, p)`, `A` is integer `(n, T)`, `Y` is float `(n,)`. Pass two batches to `controller.diagnose`. Pass the known confounders as `adjustment_set` to `decide_action`. Do not infer that set from the VIMP.

### What is easy to download and is not this benchmark

UCI Online Retail and Diabetes 130 can be turned into a panel, but they do not identify an always-treat versus never-treat contrast under time-varying confounding. Use them only after `A_t`, `L_t`, and `Y` have been defined, and do not quote their TMLE number as a ground-truth error. OpenML electricity (`data_id` 151) is a real time-ordered concept-drift table. It can exercise the two arms. It is not an incentive trajectory and this package does not run TMLE on it.

## Run

From the repository root, with numpy, scipy, scikit-learn, pandas, and matplotlib installed:

```bash
python -m experiments.fsds_ltmle --n 480 --trees 40 --q-trees 40 --perms 5 --mc 5000 --seeds 0 1 2 3
```

Outputs:

- `experiments/fsds_ltmle/results/metrics.csv` — one row per seed and regime
- `experiments/fsds_ltmle/results/BOARD.md` — absolute error, weights, and whether the planted driver landed in the top three
- `experiments/fsds_ltmle/results/ate_error.png`
- `experiments/fsds_ltmle/results/routing_weights.png`

Tests that do not run the full grid:

```bash
python -m experiments.fsds_ltmle.test_fsds_ltmle
```

The tests check the g-formula separation, the pseudo-outcome algebra, that PermuCATE's outcome and conditional fits are `RandomForestRegressor`, that the domain forest ranks a shifted column, that a red light does not drop severity, channel, or redemption, and that the TMLE recovers a known two-period effect when the incentive is randomized.

## Estimators in the report

- Frozen: Super Learner weights are chosen once on the pre-shift terminal regression and reused at every later Q step. Truncation stays 0.02. This is the static library.
- Refit: weights are re-chosen by cross-validation on the evaluation batch. Truncation stays 0.02. No FSDS input.
- FSDS-guided: same cross-validation as refit, plus the ridge penalty and the truncation from `decide_action`.

On a linear blip, cross-validation keeps most of the weight on ridge. Forcing the forest there raises the error, which is why the penalty stays off unless the concept arm clears the null by the factor above. On the quadratic blip, cross-validation already moves weight onto the forest. The concept arm names redemption and the exposure rate, and it applies that same move. On a channel shift, the covariate arm names channel and tightens truncation. The absolute-error gap from truncation alone is smaller than the seed standard deviation, because only the upper tail of the propensity is past 0.9. The rank is the signal that the feed changed.

The bundled four-seed run (`n=480`, 40 trees, 5 permutations) did this:

- Stable batches stayed green. FSDS matched the refit. Mean absolute error was 0.158, against 0.181 for the frozen weights.
- Covariate batches raised only the covariate arm. `channel_mean` or `channel_last` was the top covariate feature on every seed. Truncation moved to 0.10. The forest weight stayed near 0.14.
- Concept batches raised only the concept arm. `redemption_mean` was the top concept feature on every seed, with `treat_rate` or `redemption_last` next. The forest weight moved from 0.21 frozen to 1.00 guided. Mean absolute error was 0.129, against 0.139 frozen.
- Joint batches raised both arms and set the pause flag. Channel led the covariate arm and redemption led the concept arm on every seed.

Do not read the third digit of the absolute error as a win. The seed standard deviation on the covariate regime is about 0.18.

## Do not

- Do not drop a confounder because its VIMP is small, or because its drift rank is large.
- Do not describe a red light as an automatic new policy.
- Do not describe PO-risk as the influence function of `psi`.
- Do not replace the PermuCATE outcome forest or the conditional forest with a linear regressor and still call the result PermuCATE as specified here.
