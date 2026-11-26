# Amazon k=full DR / RR / CF (B=50, n_perm=150): five core arms

## MiniLM_L6 · DR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.04 | 0.05 | 0.05 | 0.03 | 38 |
| gift | sentiment_meta | 1.00 | 0.94 | 1.00 | 0.98 | 1.00 | 50 |
| gift | syntax_meta | 0.62 | 0.46 | 0.38 | 0.52 | 0.50 | 50 |
| gift | train_vs_heldout | 0.58 | 0.48 | 0.32 | 0.44 | 0.56 | 25 |
| cross | null_shift | 0.12 | 0.04 | 0.16 | 0.04 | 0.04 | 25 |
| cross | domain_shift | 0.50 | 0.67 | 0.38 | 0.54 | 0.50 | 12 |

## MiniLM_L6 · RR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | null_shift | 0.04 | 0.00 | 0.00 | 0.03 | 0.05 | 38 |
| gift | sentiment_meta | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 50 |
| gift | syntax_meta | 0.68 | 0.38 | 0.44 | 0.64 | 0.48 | 50 |
| gift | train_vs_heldout | 0.45 | 0.36 | 0.48 | 0.36 | 0.44 | 25 |
| cross | null_shift | 0.04 | 0.08 | 0.08 | 0.04 | 0.00 | 25 |
| cross | domain_shift | 0.10 | 0.08 | 0.00 | 0.00 | 0.00 | 12 |

## MiniLM_L6 · CF

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|

## qwen3-0.6b · DR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | sentiment_meta | 1.00 | 1.00 | 1.00 | — | — | 0 |

## qwen3-0.6b · RR

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|
| gift | sentiment_meta | 1.00 | 1.00 | 1.00 | — | — | 0 |

## qwen3-0.6b · CF

| 数据 | 场景 | without_meta | with_meta | with_spacy | with_vader4 | with_meta_decorr | min n |
|---|---|--:|--:|--:|--:|--:|--:|

