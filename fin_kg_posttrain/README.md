# 金融图谱后训练流水线（股权穿透 + 图谱风险）

面向 **2×A100 + LLaMA-Factory + ShareGPT JSONL** 的端到端落地包。
今晚目标：边列表 → 穿透/风险算法 → ShareGPT JSONL → 挂到 LLaMA-Factory 开训。

---

## 0. 整体逻辑（一眼看完）

```
[图引擎 / CSV 边列表]
        │
        ▼
┌───────────────────────┐
│ 1. Graph Loader       │  nodes + edges（持股 / 担保 / 任职 …）
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Task Sampler       │  股权穿透 / 双向查询 / 担保风险 / 拒答陷阱
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. Solver             │  多路径穿透比例 + 风险传导路径（可验证 GT）
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. ShareGPT Builder   │  system + human(子图+问题) + gpt(<think>+结论)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 5. Mix & Export       │  SFT 70% 图谱 + 30% 通用占位；另导出 DPO 对
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 6. LLaMA-Factory      │  dataset_info.json + sft.yaml + ds_z2
│    Qwen2.5-32B LoRA   │  FORCE_TORCHRUN=1 llamafactory-cli train …
└───────────────────────┘
```

### 内化 vs 外挂边界

| 进权重（SFT 学推理） | 不进权重（GraphRAG / Text-to-Cypher） |
|---|---|
| 穿透公式、50%/20%/5% 红线、担保传导逻辑 | 每日质押、最新冻结、实时股价 |
| 如何读 YAML/子图并逐步推导 | 长尾孙公司、微股东 |

---

## 1. 今晚怎么跑

```bash
# 在仓库根目录（SAPerm）
# 今晚先 polish 逻辑；确认后再跑生成（先别开训）

python -m fin_kg_posttrain.scripts.build_dataset \
  --edges fin_kg_posttrain/data/demo/edges.csv \
  --nodes fin_kg_posttrain/data/demo/nodes.csv \
  --out-dir fin_kg_posttrain/data/generated \
  --n-sft 200 \
  --n-dpo 40

# 输出：
#   fin_kg_posttrain/data/generated/fin_kg_sft.jsonl
#   fin_kg_posttrain/data/generated/fin_kg_dpo.jsonl
#   fin_kg_posttrain/data/generated/dataset_info.snippet.json
```

把 `data/generated/*.jsonl` 拷到 LLaMA-Factory 的 `data/`，把 snippet 合并进 `data/dataset_info.json`，再：

```bash
FORCE_TORCHRUN=1 llamafactory-cli train configs/sft_qwen32b_lora_2xa100.yaml
```

（YAML 里请改 `model_name_or_path` 指向你本机 Qwen2.5-32B-Instruct。）

---

## 2. 数据包配比（SFT，默认 10 万条量级配方）

| 模块 | 占比 | 文件 / 生成器 |
|---|---|---|
| 多跳穿透 + CoT | 30% | `tasks/equity_penetration.py` |
| 双向 / 聚合查询 | 15% | `tasks/bidirectional.py` |
| 担保 / 风险传导 | 15% | `tasks/guarantee_risk.py` |
| 拒答 / 阈值陷阱 | 10% | `tasks/hard_negatives.py` |
| 金融通用 + 数学占位 | 30% | `data/demo/general_mix.jsonl`（自行扩） |

DPO：5k~10k 对，`chosen` = 正确 CoT，`rejected` = 伪控股 / 加总错误 / 方向颠倒。

---

## 3. 训练配方（2×A100 80G）

- 基座：**Qwen2.5-32B-Instruct**（或 QwQ-32B）
- 方式：BF16 LoRA `r=64` `alpha=128` `target=all` + DeepSpeed **ZeRO-2**
- `cutoff_len=8192`，全局 batch≈64，`lr=5e-5`，**2 epochs**
- **`train_on_prompt: false`**（只学 gpt，不背 human 里的子图）

详见 `configs/`。

---

## 4. 三条数据红线（脚本已内置）

1. **环路截断**：交叉持股 / 环形担保到环就停，CoT 写明“检测到环，停止穿透”。
2. **数值精度**：中间 4 位小数，最终比例 **2 位**（如 `21.20%`）。
3. **阈值敏感**：专门造 `49.9% vs 50.1%`、`4.9% vs 5.1%`、`19.9% vs 20.1%` 对比样本。

---

## 5. 目录

```
fin_kg_posttrain/
  graph/          # 图模型 + 穿透/风险求解
  tasks/          # 各类 ShareGPT 样本生成
  builders/       # JSONL 写出 / 校验
  configs/        # LLaMA-Factory + DeepSpeed
  data/demo/      # 可跑通的小图
  scripts/        # CLI
```
