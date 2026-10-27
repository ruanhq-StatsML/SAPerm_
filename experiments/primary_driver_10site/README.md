# 10-site primary-driver benchmark

Real covariates come from the WhyShift ACS tasks (California, 2018): Income and Public Coverage. Ten sites share that empirical `P(X)`. After a burn-in, five sites receive heterogeneous concept drift, four stay clean, and one site randomizes its labels.

The headline site score, `delta_cf`, is a size-matched Brier gap between a burn-in reference model and a cross-fit local model, multiplied by how far the local model is from chance. The gap is evaluated on the whole window and on the upper and lower tails of `X1`, and the maximum is kept. That is the comparison used for pairwise concordance of concept-drift sites against clean sites.

```bash
python3 -m experiments.primary_driver_10site.test_dgp
python3 -m experiments.primary_driver_10site.run_benchmark --suite all --seeds 4
```

Tables and figures land in `experiments/primary_driver_10site/results/`. The narrative write-up is `RESULTS.md` in that folder. ACS extracts are cached under `cache/` and are not part of the source tree.
