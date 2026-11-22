# shift-detect software (simple)

```
digest.py              # data digestion entry
interactions.py        # PO-risk, MMD, R-risk, power-loss, RF/DR/CF
run_benchmark_all.py   # main → DataFrame (csv/parquet)
utils.py               # tiny shared helpers
datasets/              # dataset-specific only
```

```bash
python digest.py --dataset modasense_emotion --audio --write
python run_benchmark_all.py --dataset modasense_emotion --smoke
python run_benchmark_all.py --dataset all --B 40
```

Curated numbers already in `results/` (scientific case + today tables).
Legacy one-off scripts: `_legacy/`.
