# Pseudo-outcome Shapley for online OOD interpretability

Same idea as predictive SHAP, applied to the DR pseudo-outcome model
`tau(X) ≈ φ`. The online OOD score is `mean(tau²)`; Shapley of `tau`
names which covariates drive that score.

**LaTeX writeup:** [`po_risk_shapley.tex`](po_risk_shapley.tex)

**Fuller sweep:** `python -m experiments.online_po_risk.run_po_shapley --seeds 20`

## Window summary ($B=20$)

| scenario | phase | τ | top Shap→drv | top RF→drv | top MMD→drv | drv Shap | noise Shap | τ² |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| concept | pre | rf | 0.20 | 0.20 | 0.05 | 0.10 | 0.12 | 0.41 |
| concept | **post** | rf | **1.00** | 0.35 | 0.05 | **0.65** | 0.12 | **2.63** |
| concept | post | ridge | **1.00** | 0.35 | 0.05 | **0.91** | 0.15 | **4.29** |
| covariate | post | rf | 0.20 | **1.00** | **1.00** | 0.14 | 0.12 | 0.50 |
| covariate | post | ridge | 0.20 | **1.00** | **1.00** | 0.24 | 0.19 | 0.61 |

Concept: Shapley recovers `{2,3}`; RF/MMD do not.  
Covariate: RF/MMD recover `x0`; Shapley stays quiet (τ² barely moves).

## Artifacts

- `po_shapley_window.csv` / `_summary.csv`
- `po_shapley_feature_means.csv`
- `online_po_shapley.csv` / `_summary.csv`
- `po_shapley_full.log`
