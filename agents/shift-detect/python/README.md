# shift-detect software (simple)

```
digest.py              # data digestion entry
interactions.py        # PO-risk, MMD, R-risk, power-loss, RF/DR/CF + localize_subset
subset_localize.py     # packaged CFPerm LOCO → driving feature subset
run_benchmark_all.py   # main → DataFrame (csv/parquet)
utils.py               # tiny shared helpers
datasets/              # dataset-specific only
```

Canonical joint table: `results/joint_shift_analysis.tex`.
Localization = packaged `subset_localize` (block CFPerm → coordinate subset), not a new method.

```bash
python digest.py --dataset modasense_emotion --audio --write
python run_benchmark_all.py --dataset modasense_emotion --smoke
python run_benchmark_all.py --dataset all --B 40
```

Curated numbers already in `results/` (scientific case + today tables).
Legacy one-off scripts: `_legacy/`.
