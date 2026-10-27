# Ten-site covariate and concept board

Each site is scored on its own burn-in against its own post-change window. MMD and HSIC are the covariate scores. R-risk is excess test MSE of a model fit on the burn-in. The forest column is the same gap with a random forest.

The label rules are the ten-site plan in `experiments.dgps`. Covariates are a synthetic correlated table. The base label uses `v3`, `v4`, and `v5`, not the planted coordinates.

```bash
python3 -m experiments.ten_site_board
```

`results/BOARD.md` is the reading of the table. The short version: the three scores are not one ranking, and no ranking names a unique driver.
