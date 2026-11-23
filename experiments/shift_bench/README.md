# shift_bench — wrap-up

大体逻辑就这样；细节后面再磨。

```
utils.py
interactions.py      # PO-risk / Shapley / MMD / R-risk
embed_periods.py     # time: discretize → embed → AlignHead → concat
digest.py            # BiasFinder / ModaSense / OpenML / period_text
digest_mm.py         # MSR-VTT/LCO: digest_concat | digest_aligned
run_benchmark_all.py # main → csv；--legacy-all → whole-methods power
```

```
raw → digest(+embed) → interactions.window_bundle → DataFrame
                     ↘ --legacy-all → benchmark_all_method
```

```bash
python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --scenario train_vs_test --B 20
python -m experiments.shift_bench.run_benchmark_all --dataset period_text --align-dim 32 --B 10
python -c "from experiments.shift_bench.digest_mm import digest_concat, digest_aligned"
```

PO-Shapley / online: `experiments/online_po_risk/`  
MSR-VTT/LCO: `docs/msrvtt-lco-digestion.md` + `digest_mm.py`
