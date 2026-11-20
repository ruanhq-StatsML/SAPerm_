# 金融/公平性知识表示后训练（接 SAPerm FT pipeline）

**主路径：real-world `dataset.zip` → 结构化 representation → LoRA FT → embedding CSV → 分布漂移检验。**  
`data/demo/` 里的小股权图只用于穿透算法单测，不是训练数据源。

---

## 0. 怎么接到你原来的 pipeline

你原来的链路：

```
fine_tuning_data_extraction.py
  → fine_tuning_data_wrapper_bert/gpt
  → Resulted_data/X_emb_*.csv
  → pipeline_example.py / benchmark_all_method
```

现在多一步「抽知识再打包」，入口是同级的：

```bash
python fine_tuning_kg_representation.py --corpus equity_evaluation
```

产出：

| 文件 | 用途 |
|---|---|
| `data/realworld/ft_text_<corpus>.csv` | 直接喂 `fine_tuning_data_wrapper_*`（`text_col=text`） |
| `data/realworld/nodes_*.csv` / `edges_*.csv` | 结构化图（可再进 ShareGPT / 多跳） |
| `data/realworld/sharegpt_sft_*.jsonl` | 可选指令微调 |
| `Resulted_data/X_emb_kgrep_*` | `--run-ft` 后的 embedding，对齐 `pipeline_example` |

---

## 1. 支持的 real corpora（均来自 `datasets/dataset.zip`）

- `equity_evaluation` — Equity-Evaluation-Corpus（Person/Gender/Race/Emotion）
- `biasfinder` — train/test_bias2
- `sarcasm` — GEN/HYP/RQ-sarc-notsarc
- `semeval_sarc` — train.En copy.csv
- `amazon_magazine` / `amazon_gift_cards`

---

## 2. Representation 长什么样

实体中心打包（示例，EEC）：

```text
实体: Alonzo
[人口属性]: 性别=male; 族裔=African-American
[表达]: 情绪=anger; 情绪词=angry
[模板]: <person subject> feels <emotion word>.
[原句]: Alonzo feels angry.
```

默认 **25% raw 原文** 混入，避免只认打包句式（对应你 FT 里的防遗忘混比）。

---

## 3. 目录

```
fin_kg_posttrain/
  realworld/     # loaders + extract_kg + pack
  bridge_to_ft.py
  graph/         # FinGraph +（demo 用）穿透/担保求解
  builders/      # ShareGPT JSONL
  configs/       # 可选 Qwen/LLaMA-Factory
  data/demo/     # synthetic fixture only
```

详见 `LOGIC.md`。
