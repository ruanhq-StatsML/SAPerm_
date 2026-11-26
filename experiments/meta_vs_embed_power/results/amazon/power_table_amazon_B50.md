| 编码器 | 场景 | 检验 | without_meta | with_meta | Δ(with−without) | with_spacy | Δ(spacy−without) | meta_only |
|---|---|---|--:|--:|--:|--:|--:|--:|
| MiniLM_L6 | sentiment_meta | CFPerm | 0.94 | 1.00 | +0.06 | 1.00 | +0.06 | — |
| MiniLM_L6 | sentiment_meta | RRPerm | 1.00 | 1.00 | +0.00 | 1.00 | +0.00 | — |
| MiniLM_L6 | sentiment_meta | DRPerm | 1.00 | 0.18 | -0.82 | 1.00 | +0.00 | — |
| MiniLM_L6 | syntax_meta | CFPerm | 0.58 | 0.94 | +0.36 | 0.92 | +0.34 | — |
| MiniLM_L6 | syntax_meta | RRPerm | 0.88 | 0.34 | -0.54 | 0.60 | -0.28 | — |
| MiniLM_L6 | syntax_meta | DRPerm | 0.58 | 0.20 | -0.38 | 0.30 | -0.28 | — |
| MiniLM_L6 | train_vs_heldout | CFPerm | 0.46 | 0.80 | +0.34 | 0.78 | +0.32 | — |
| MiniLM_L6 | train_vs_heldout | RRPerm | 0.78 | 0.60 | -0.18 | 0.74 | -0.04 | — |
| MiniLM_L6 | train_vs_heldout | DRPerm | 0.36 | 0.28 | -0.08 | 0.48 | +0.12 | — |
| MiniLM_L6 | domain_shift | CFPerm | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | — |
| MiniLM_L6 | domain_shift | RRPerm | 0.00 | 0.16 | +0.16 | 0.02 | +0.02 | — |
| MiniLM_L6 | domain_shift | DRPerm | 0.36 | 0.56 | +0.20 | 0.56 | +0.20 | — |
