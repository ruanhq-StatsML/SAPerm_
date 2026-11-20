# 整体逻辑（real-world → representation → 你的 FT pipeline）

> synthetic `data/demo/` 只做 solver 单测；正式路径从 `datasets/dataset.zip` 抽知识。

## 数据流（接到现有 pipeline）

```
datasets/dataset.zip
  (EEC / BiasFinder / sarcasm / Amazon …)
        │
        ▼
 realworld.loaders.load_corpus
        │
        ▼
 realworld.extract_kg.extract_bundle
   · 把数据集里已有的结构化字段变成 nodes/edges
   · 同时打成 entity-centric packed text（representation）
        │
        ├──────────────────────────────┐
        ▼                              ▼
 pack_for_ft                         pack_sharegpt_rows
 ft_text_*.csv                       sharegpt_sft/dpo_*.jsonl
 (text + label cols)                 (可选指令微调)
        │
        ▼
 fine_tuning_data_wrapper_bert/gpt     ← 你现有的 FT_data_extraction_utils
        │
        ▼
 Resulted_data/X_emb_kgrep_*.csv
        │
        ▼
 pipeline_example.py / benchmark_all_method
```

## 各语料抽什么（不是编造股权）

| corpus | 节点 / 边 | FT representation |
|---|---|---|
| `equity_evaluation` | Person→Gender/Race；Sentence→Emotion/Template | 人物档案 YAML 式打包 |
| `biasfinder` | Document→sentiment_label | 原文 + 标签边说明 |
| `sarcasm` / `semeval_sarc` | Document→sarcasm_label | 同上 |
| `amazon_*` | User→rates→ASIN（rating 作 ratio） | 商品/评分/评论打包 |

## 今晚 polish 优先改

1. `realworld/extract_kg.py` — 边类型是否够用、Amazon 聚合
2. `realworld/pack.py` — EEC 多跳问题改写、raw_mix 比例
3. `bridge_to_ft.py` — 与 wrapper 的 column_to_join 对齐

## 命令

```bash
# 只抽 representation（不训）
python fine_tuning_kg_representation.py --corpus equity_evaluation

# 接上 LoRA + embedding（要 GPU / transformers / peft）
python fine_tuning_kg_representation.py --corpus biasfinder --run-ft --ft-backend bert
```
