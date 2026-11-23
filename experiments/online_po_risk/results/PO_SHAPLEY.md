# Pseudo-outcome Shapley for online OOD interpretability

Same idea as predictive SHAP, applied to the DR pseudo-outcome model
`tau(X) ≈ φ`. The online OOD score is `mean(tau²)`; Shapley of `tau`
names which covariates drive that score.

**LaTeX writeup:** [`po_risk_shapley.tex`](po_risk_shapley.tex)
(argument + Table~1 three-way smoke).

## API (`experiments.online_po_risk.score`)

- `fit_dr_tau` — nuisances + `tau` (`ridge` or `rf`)
- `tau_shapley_importance` — mean `|SHAP|` of `tau`
- `ood_vimp` / `ood_vimp_wrapper` — three-way:
  1. PO Shapley + residual-permutation VIMP
  2. RF domain classifier
  3. MMD-LOCO
- `online_po_shapley` — sliding window (first half ref, second half new)

`run_comparison` also writes `online_po_shapley.csv` next to `online_pfi.csv`.

## Smoke (concept stream, signal coords `{2,3}`)

Post-shift window attribution (`ood_vimp_threeway_smoke.csv`):

| feature | po_shapley | po_vimp | rf_vimp | MMD_vimp | signal |
| --- | ---: | ---: | ---: | ---: | --- |
| x2 | **0.54** | **1.59** | 0.09 | ~0 | yes |
| x3 | **0.50** | **1.24** | 0.09 | ~0 | yes |
| others | ≤0.20 | ≤0.31 | higher | ~0 | no |

Concept shift moves `Y|X` without moving `P(X)`, so RF-domain and MMD stay
flat; Shapley on `tau` recovers the signal axes. Sliding-window smoke
(`online_po_shapley_smoke_summary.csv`): post `signal_shap > noise_shap`.
