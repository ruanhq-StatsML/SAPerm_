# Amazon k=full DR / RR / CF (B=50, n_perm=150): five core arms

## MiniLM_L6 · DR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.04 | 0.04 | 0.04 | 0.02 | 50 |
| gift | sentiment_meta | 1.00 | 0.94 | 1.00 | 0.98 | 1.00 | 50 |
| gift | syntax_meta | 0.62 | 0.46 | 0.38 | 0.52 | 0.50 | 50 |
| gift | train_vs_heldout | 0.54 | 0.40 | 0.44 | 0.48 | 0.54 | 50 |
| cross | null_shift | 0.08 | 0.08 | 0.10 | 0.08 | 0.08 | 50 |
| cross | domain_shift | 0.44 | 0.48 | 0.54 | 0.42 | 0.50 | 50 |

## MiniLM_L6 · RR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.00 | 0.00 | 0.02 | 0.04 | 50 |
| gift | sentiment_meta | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 50 |
| gift | syntax_meta | 0.68 | 0.38 | 0.44 | 0.64 | 0.48 | 50 |
| gift | train_vs_heldout | 0.44 | 0.46 | 0.48 | 0.30 | 0.44 | 50 |
| cross | null_shift | 0.04 | 0.06 | 0.08 | 0.08 | 0.02 | 50 |
| cross | domain_shift | 0.06 | 0.04 | 0.00 | 0.04 | 0.00 | 50 |

## MiniLM_L6 · CF

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.08 | 0.08 | 0.08 | 0.10 | 0.08 | 50 |
| gift | sentiment_meta | 0.56 | 0.94 | 0.62 | 0.80 | 0.56 | 50 |
| gift | syntax_meta | 0.32 | 0.52 | 0.52 | 0.50 | 0.36 | 50 |
| gift | train_vs_heldout | 0.26 | 0.40 | 0.28 | 0.34 | 0.32 | 50 |
| cross | null_shift | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 | 50 |
| cross | domain_shift | 0.00 | 0.02 | 0.04 | 0.02 | 0.02 | 50 |

## qwen3-0.6b · DR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.08 | 0.04 | 0.04 | 0.06 | 50 |
| gift | sentiment_meta | 1.00 | 0.96 | 1.00 | 0.98 | 1.00 | 50 |
| gift | syntax_meta | 0.36 | 0.30 | 0.32 | 0.28 | 0.34 | 50 |
| gift | train_vs_heldout | 0.48 | 0.40 | 0.48 | 0.44 | 0.44 | 50 |
| cross | null_shift | 0.06 | 0.02 | 0.00 | 0.04 | 0.02 | 50 |
| cross | domain_shift | 0.80 | 0.74 | 0.86 | 0.74 | 0.76 | 50 |

## qwen3-0.6b · RR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.08 | 0.08 | 0.04 | 0.06 | 50 |
| gift | sentiment_meta | 0.98 | 1.00 | 0.96 | 0.98 | 0.96 | 50 |
| gift | syntax_meta | 0.12 | 0.14 | 0.14 | 0.12 | 0.16 | 50 |
| gift | train_vs_heldout | 0.16 | 0.12 | 0.14 | 0.16 | 0.16 | 50 |
| cross | null_shift | 0.06 | 0.02 | 0.02 | 0.04 | 0.08 | 50 |
| cross | domain_shift | 0.10 | 0.02 | 0.04 | 0.02 | 0.02 | 50 |

## qwen3-0.6b · CF

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.02 | 0.06 | 0.06 | 0.06 | 0.06 | 50 |
| gift | sentiment_meta | 0.54 | 0.78 | 0.56 | 0.68 | 0.54 | 50 |
| gift | syntax_meta | 0.10 | 0.20 | 0.20 | 0.20 | 0.20 | 50 |
| gift | train_vs_heldout | 0.10 | 0.12 | 0.12 | 0.12 | 0.10 | 50 |
| cross | null_shift | 0.04 | 0.10 | 0.10 | 0.08 | 0.08 | 50 |
| cross | domain_shift | 0.04 | 0.08 | 0.08 | 0.08 | 0.08 | 50 |

