# 整体逻辑（~300 行骨架对照表）

> 先别跑训练。今晚 polish 按这个心智模型改。

## 数据流

```
nodes.csv + edges.csv
        │
        ▼
 FinGraph.from_csv          graph/models.py
        │
        ├─ holding_out / guarantee_out / subgraph_text
        │
        ▼
 TaskSampler                tasks/sample.py
   equity_forward | equity_reverse | guarantee_risk | refusal | threshold_trap(DPO)
        │
        ▼
 Solver GT
   compute_penetration      graph/penetration.py   ← 乘积穿透 + 环截断 + 红线
   assess_guarantee_risk    graph/risk.py          ← 担保边 + 是否并表控制担保方
        │
        ▼
 ShareGPT Builder           builders/sharegpt.py
   system + human(子图+问题) + gpt(<think>+结论)
   DPO: conversations + chosen + rejected
        │
        ▼
 pipeline.build             pipeline.py
   fin_kg_sft.jsonl / fin_kg_dpo.jsonl / dataset_info.snippet.json
        │
        ▼
 LLaMA-Factory              configs/*.yaml
   Qwen2.5-32B LoRA r=64 + ZeRO-2  ·  train_on_prompt=false
```

## 文件 ↔ 行数心智（polish 优先级）

| 优先级 | 文件 | 你该改什么 |
|---|---|---|
| P0 | `graph/penetration.py` | 穿透会计准则（交叉持股剔除是否改成按准则折算，而非简单截断） |
| P0 | `tasks/sample.py` | 问题改写风格、mix 配比、一对多 Top-K 逆向 |
| P1 | `graph/risk.py` | 担保圈多跳、历史已解除状态字段 |
| P1 | `builders/sharegpt.py` | system 约束措辞、多轮追问 |
| P2 | `pipeline.py` | CLI / 混入 general_mix 比例 |
| P2 | `configs/*` | 本机 model path、batch |

## 三条红线（脚本已写死，polish 时别拆）

1. **环**：path 上节点重复 → `stopped_on_cycle`，不乘进环边。
2. **精度**：中间 path product 展示 4 位 %，对外答案 2 位 %。
3. **阈值**：5% / 20% / 50% 在 `threshold_note`；DPO rejected 专治「加总当穿透 / 49% 当控股」。

## Demo GT 手算（polish 时用来对答案）

- 王五 → 恒泰：`0.8×0.10 + 0.8×0.55×0.30 = 8% + 13.2% = 21.20%`
- 华兴 → 宏图：`0.20 + 0.60×0.40 = 20% + 24% = 44.00%`
