# Site-level principal drivers

Four synthetic shifts. Each replicate is cut into six sites. Every site scores leave-one-covariate MMD, HSIC, and R-risk. The delivery is the mean of those site vectors, set next to the same scores on the pooled sample. The planted support is the first four coordinates. An arm fires when its aggregate selection AUC is at least 0.70.

Covariate shift, linear and nonlinear, is named by MMD and HSIC. Concept drift, linear and nonlinear, is named by R-risk. On nonlinear concept drift the site average clears 0.70 and the pooled R-risk does not.

## Layout

| File | Role |
| --- | --- |
| `dgp.py` | Linear and nonlinear covariate and concept generators |
| `mmd_hsic_rrisk.py` | Leave-one-covariate MMD, HSIC, and ridge R-risk |
| `site_principal_drivers.py` | Site split, aggregate, pooled score, LaTeX table |
| `results/site_scores.csv` | Per-site, aggregate, and pooled rows, five replicates |
| `results/synth_site_drivers.tex` | Mean AUC / top-4 F1 table |

Requires NumPy, pandas, and scikit-learn. From the repository root:

```bash
python3 -m experiments.synth_site_drivers
```

`--reps` sets the number of replicates (default 5). `--out-dir` sets where the CSV and the TeX table are written. The checked-in table is the five-replicate run.
