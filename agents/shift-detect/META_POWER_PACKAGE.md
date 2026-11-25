# Meta × Embed package

Download: [`meta_power_package.zip`](meta_power_package.zip).

## Five scripts only

```
digest.py
interactions.py        # PO-risk, MMD, R-risk, power-loss, RF/DR/CF
run_benchmark_all.py   # → DataFrame
utils.py
subset_localize.py     # ORIGINAL localization (CFPerm → subset)
datasets/              # dataset-specific
```

```
digest → Digested → interactions / run_benchmark_all
                  → subset_localize
```

Results: `python/results/joint_shift_analysis.tex`, `manuscript_final.tex`.
