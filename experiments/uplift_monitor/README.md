# Uplift monitor and Thompson sampling

A streaming uplift model, a batch monitor, and a four-policy bandit.

Batches 0–3 are fully randomized and fit the stale two-model logistic uplift. From batch 4, 30% of each batch stays randomized. That slice is the only place the monitor is valid. The other 70% is assigned by discounted Thompson sampling. Each arm is a policy: always control, always treat, the frozen uplift model, or an uplift model refit on the last two randomized windows. The reward is the binary outcome. The Beta posterior is multiplied by 0.85 each batch.

Two regimes start at batch 8.

- `concept` flips the sign of τ(x) and leaves P(X) alone.
- `covariate` shifts the first coordinate and leaves τ(x) alone.

`evaluate.py` is the uplift evaluation. `r_learner_observed_risk` follows `vimp_loco_r_risk` through `observed_r_risk`: random-forest outcome, logistic propensity, weighted-ridge tau refit on the batch. `permucate_test_po_risk` follows `permuCATE_vimp` through `po_risk_original`: stratified half split, random-forest outcome and pseudo-outcome models. `deployed_po_risk` is the same `po_risk` formula with the already deployed logistic uplift. The call uses MMD and that deployed score. The script refits tau, so its own risk is not the drift call. Forests use 40 trees; the registry default is 150. The LOCO loop and the permutation loop are not run on every batch.

The printed board opens with a call. Covariate fires when late MMD is at least five times the early MMD. Uplift fires when frozen PO-risk rises by 0.25 and exceeds the refit by 0.15.

```bash
python3 -m experiments.uplift_monitor
```

The reading of the saved run is `results/BOARD.md`.
