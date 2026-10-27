# DDPM counterfactual attribution on a 10-site linear panel

Conditional-distribution counterfactual for a delayed, heterogeneous panel.
Ten sites share one IP-block spike. Each site has its own linear distributed-lag
kernel, so the site that moves today's purchase distribution is the one whose
kernel is open at that lag, not the site with the loudest anomaly.

This is a conditional counterfactual: the lagged IP feature is set to a
reference draw and the purchase index is regenerated. It is not a structural
effect, and the risk / recommendation / social channels are not separated.

## Landing case

`case/landing_case.tex` with `case/landing_case.bib` is the product write-up of the validation run.
From that directory: `pdflatex landing_case && bibtex landing_case && pdflatex landing_case && pdflatex landing_case`.

## Run

```python
import ddpm_cf_attribution
result = ddpm_cf_attribution.run()
```

```bash
pip install -r ddpm_cf_attribution/requirements.txt
python -m ddpm_cf_attribution.example
```

The full run trains a conditional DDPM (cosine schedule, noise prediction,
10% lag dropout) and writes `ddpm_cf_attribution/outputs/REPORT.md`.
A `--smoke` flag exercises the same path on a tiny panel.

The headline attribution is still the random-Fourier MMD². The ratio of that
score to the no-intervention baseline moves with bandwidth, because the
baseline sits on the null floor of the RBF kernel. The same kernel inverts
to a squared location shift, `shift2`, which is the bandwidth-stable reading.
The frozen kernel clause still judges the raw ratio.

Thresholds live in `protocol.py` and are the acceptance criteria. They were
fixed before the run.

## Datasets

The estimator is scored on simulated panels because a counterfactual path has
no observational ground truth. Three generators are the ones that match the
query; only the first is executed in the report.

| Generator | What is heterogeneous | What it identifies | Run |
|---|---|---|---|
| `TenSiteLinearCD` (`--family linear`) | Site-specific linear lag kernel, dose, isolation, one social-overlap site | Primary site, primary lag, primary block feature | default |
| `TenSiteTanhCD` (`--family tanh`) | Same kernels through `amp * tanh(unit · lags)` | Whether the ranking survives a nonlinear conditional mean | optional |
| Semi-synthetic retail panel | Replace the simulated covariates with a real store panel and keep the kernel | Same query on real seasonality | not bundled |

Sites in `TenSiteLinearCD`:

| Site | Name | Kernel | Role at the lag-10 query |
|---:|---|---|---|
| 0 | weekly_primary | bump at day 10, weight 0.95 | primary driver |
| 1 | weekly_secondary | same center, weight 0.50 | secondary |
| 2 | daily_driver | bump at day 3 | driver of a different query |
| 3 | immediate_driver | bump at day 1 | driver of a different query |
| 4 | weekly_weak | bump at day 10, weight 0.28 | weak weekly |
| 5 | decay_recent | exponential decay | recent lags dominate; L1 temporal construction |
| 6 | social_overlap | no lag kernel; overlap × lag 2 | social construction only |
| 7 | isolated_blocked | same potential kernel as site 0, transmission zeroed | exogenous isolation |
| 8 | null | flat | negative control |
| 9 | null_noisy | flat, larger noise | negative control with a higher MMD floor |

A real panel can be wired in with the same conditioner shape `(lagged block, covariates, overlap, site)`:

- **Rossmann Store Sales.** Store = site, promo or a competitor-open shock = the IP-block spike, daily sales = the purchase index. Store coefficients are heterogeneous; the lag of the shock is the object of the path curve.
- **M5 (Walmart), 10 stores.** Same map: event indicator → block feature, sales distribution → `Y`. Use a subset of 10 stores so the site embedding stays the one this model was sized for.
- **Semi-synthetic.** Keep the real store covariates and calendar, delete the real sales, and draw `Y` from `TenSiteLinearCD`. The oracle returns, and the nuisance correlation is no longer Gaussian.

Public uplift tables (Criteo, Hillstrom, IHDP) do not carry a lag path. They can host an L4 CATE check and nothing else in this protocol.

## Evaluation

Five falsification layers, in `evaluate.py`:

1. Relative ranking under dose, lag decay, covariate overlap, and exogenous isolation.
2. Placebo lags, a precision scan, and scale alignment of the ablation gate.
3. PIT, observational Wasserstein, HSIC ratio. Necessary, not sufficient.
4. R-learner CATE, mask gap, business-window alignment, gate.
5. Baseline definition, kernel bandwidth, architecture, E-value.

The weakest layer is the credibility bottleneck. Out-of-support conditioning
steps are excluded from the headline. MMD is reported as a ratio to the
no-intervention baseline.
