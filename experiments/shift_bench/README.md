# shift_bench — 4–5 scripts（跟昨天同一套）

不搞 cross-modality alignment。测试逻辑就这一条链：

```
digest.py                 # 1. 读数据 / 切窗 → Digested
embed_periods.py          # 2. 时间离散 → Omni3/Llama3 embed → 按 period 拼接
interactions.py           # 3. 高阶交互：PO-risk / Shapley / MMD / R-risk / power-loss
run_benchmark_all.py      # 4. main：digest(+embed) → interactions → DataFrame
                          #    并接到 SAPerm 现成的 benchmark_all_method
utils.py                  # 5. 小工具：RF / folds / batch_xy
```

## Logic

```
raw (text+time 或 已有 X)
        │
        ├─ embed_periods.embed_concat(...)   # 可选：Omni3 路径
        │     discretize time → embed each period → hstack（不对齐）
        ▼
   Digested  (X0,y0,X1,y1, names)
        │
        ├─ interactions.window_bundle(...)   # RF/DR/RR/MMD/C2ST + PO-Shapley
        │         └─ run_benchmark_all.py    # B reps → csv
        │
        └─ (optional) benchmark_methods_distributionshift_testing.benchmark_all_method
              现成 whole-methods power（RFPerm/CFPerm/DR/…）
```

## Run

```bash
# 已有特征窗（BiasFinder / ModaSense / OpenML）
python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --scenario train_vs_test --B 20

# Omni3/Llama3：按时段 embed 再拼（backend=hashing 可离线冒烟；llama3 需权重）
python -m experiments.shift_bench.run_benchmark_all --dataset period_text --backend hashing --n-periods 4 --B 10

# 接上 repo 根目录现成的 whole-methods benchmark
python -m experiments.shift_bench.run_benchmark_all --dataset biasfinder --scenario train_vs_test --legacy-all --B 5
```

## 文件职责（就这些）

| 文件 | 干什么 |
| --- | --- |
| `utils.py` | `batch_xy`, RF makers, folds |
| `interactions.py` | PO-risk、τ-Shapley、MMD、R-risk、RFPerm/DRPerm/…、`window_bundle` |
| `embed_periods.py` | `discretize_periods` → `embed_texts` → `concat_period_embeds` |
| `digest.py` | dataset 窗：BiasFinder / ModaSense / Adult·Bank·Credit / period_text |
| `run_benchmark_all.py` | 唯一 main：跑 B 次存 DataFrame |

`../online_po_risk/` 里的 active sampling / stream 仍是 online 专题；**测试 + Shapley + power** 走本目录。
