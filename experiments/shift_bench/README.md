# shift_bench — 4–5 scripts（跟昨天同一套）

测试逻辑就这一条链。时间窗：**AlignHead 对齐 → concatenate**（不做 cross-modality fusion）。

```
digest.py                 # 1. 读数据 / 切窗 → Digested
embed_periods.py          # 2. 时间离散 → embed → AlignHead → concat 时间窗
interactions.py           # 3. 高阶交互：PO-risk / Shapley / MMD / R-risk / power-loss
run_benchmark_all.py      # 4. main：digest(+embed) → interactions → DataFrame
                          #    并接到 SAPerm 现成的 benchmark_all_method
utils.py                  # 5. 小工具：RF / folds / batch_xy
```

## Logic

```
raw (text+time 或 已有 X)
        │
        ├─ embed_periods.embed_concat(...)
        │     discretize time
        │     → embed each period
        │     → AlignHead（共享线性头，把各窗拉到同一空间）
        │     → concatenate 时间窗口（hstack K slots）
        ▼
   Digested  (X0,y0,X1,y1, names)
        │
        ├─ interactions.window_bundle(...)   # RF/DR/MMD + PO-Shapley
        │         └─ run_benchmark_all.py
        │
        └─ --legacy-all → benchmark_methods_distributionshift_testing.benchmark_all_method
```

## Run

```bash
python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --scenario train_vs_test --B 20

# Omni3 路径：离散时段 → embed → AlignHead → concat
python -m experiments.shift_bench.run_benchmark_all \
  --dataset period_text --backend hashing --n-periods 4 --align-dim 32 --B 10

# 关掉对齐头（只拼）
python -m experiments.shift_bench.run_benchmark_all --dataset period_text --no-align --B 5

# 接上现成 whole-methods power
python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --legacy-all --B 5
```

## 文件职责

| 文件 | 干什么 |
| --- | --- |
| `utils.py` | `batch_xy`, RF makers, folds |
| `interactions.py` | PO-risk、τ-Shapley、MMD、R-risk、`window_bundle` |
| `embed_periods.py` | `discretize` → `embed` → `AlignHead` → `concat` 时间窗 |
| `digest.py` | BiasFinder / ModaSense / OpenML / period_text |
| `run_benchmark_all.py` | 唯一 main |
