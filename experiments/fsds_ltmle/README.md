# FSDS diagnostics for a longitudinal TMLE

Two-arm feature attribution for a dynamic incentive, and the two knobs it is allowed to turn on a longitudinal TMLE: propensity truncation, and the ridge weight in the outcome Super Learner.

PermuCATE fits `RandomForestRegressor` for the outcome model and for every conditional residual model. The domain propensity is logistic. The covariate arm is an out-of-bag permutation importance from a random-forest domain classifier. The domain label is the batch, not the incentive.

The adjustment set is never edited from a VIMP. The procedure and the datasets are in `SKILL.md`.

```bash
python -m experiments.fsds_ltmle --n 480 --trees 40 --q-trees 40 --perms 5 --mc 5000 --seeds 0 1 2 3
python -m experiments.fsds_ltmle.test_fsds_ltmle
```

The run writes `results/BOARD.md`, `results/metrics.csv`, and two figures.
