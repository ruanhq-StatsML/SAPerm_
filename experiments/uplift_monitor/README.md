# Uplift monitor and Thompson sampling

A streaming uplift model, a batch monitor, and a four-policy bandit.

Batches 0–3 are fully randomized and fit the stale two-model logistic uplift. From batch 4, 30% of each batch stays randomized. That slice is the only place the monitor is valid. The other 70% is assigned by discounted Thompson sampling. Each arm is a policy: always control, always treat, the frozen uplift model, or an uplift model refit on the last two randomized windows. The reward is the binary outcome. The Beta posterior is multiplied by 0.85 each batch.

Two regimes start at batch 8.

- `concept` flips the sign of τ(x) and leaves P(X) alone.
- `covariate` shifts the first coordinate and leaves τ(x) alone.

The monitor calls the FSDS calculations in `fsds_vimp.py`: `MMD` and `compute_loco_mmd_batches`, `compute_hsic_loco_batches`, the R-learner loss from `vimp_loco_r_risk` with the deployed uplift plugged in as tau, PermuCATE `po_risk`, and `mmd_path` / `summarize_path_nonuniqueness`. Nuisance regressions are logistic and ridge. The original modules `chdir` onto a machine-local path and do not import here. Sign accuracy uses the true τ and is a simulation check, not an online score.

```bash
python3 -m experiments.uplift_monitor
```

The reading of the run is `results/BOARD.md`.
