# Amazon B=50 聚合（DR/RR/CF，已完成的单元）

仅含 DR/RR/CF；RF 在 B=50 中不采用，以 B=150 表为准。n_rep 为该格实际完成的重复数，未满 50 的格仍在跑。
数据：`amazon_MiniLM_L6` 为 gift 同域场景；`amazon-cross_MiniLM_L6` 为 gift 对 magazine 的域漂移。
同一臂的 DR/RR 与 CF 可能分属两批检查点（主链 / 去相关），空缺处为 `--`。

| 数据 | 场景 | 臂 | RR | DR | CF | n_rep |
|---|---|---|--:|--:|--:|--:|
| amazon-cross_MiniLM_L6 | domain_shift | with_meta | 0.16 | 0.56 | 0.02 | 50 |
| amazon-cross_MiniLM_L6 | domain_shift | without_meta | 0.00 | 0.36 | 0.02 | 50 |
| amazon_MiniLM_L6 | null_shift | with_meta_decorr | 0.04 | 0.06 | 0.04 | 50 |
| amazon_MiniLM_L6 | null_shift | with_spacy_decorr | 0.08 | 0.06 | 0.04 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_meta | 1.00 | 0.18 | 1.00 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_meta_decorr | 1.00 | 1.00 | 0.98 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_meta_decorr_t02 | 1.00 | 1.00 | 0.92 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_meta_decorr_t02nv | 1.00 | 1.00 | 0.98 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_spacy | 1.00 | 1.00 | 1.00 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_spacy_decorr | 1.00 | 1.00 | 0.98 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | with_spacy_decorr_t02 | 1.00 | 1.00 | 0.96 | 50 |
| amazon_MiniLM_L6 | sentiment_meta | without_meta | 1.00 | 1.00 | 0.94 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_meta | 0.34 | 0.20 | 0.94 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_meta_decorr | 0.68 | 0.34 | 0.92 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_meta_decorr_t02 | 0.82 | 0.36 | 0.90 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_meta_decorr_t02nv | 0.90 | 0.74 | 0.70 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_spacy | 0.54 | 0.21 | 0.96 | 24 |
| amazon_MiniLM_L6 | syntax_meta | with_spacy_decorr | 0.82 | 0.48 | 0.94 | 50 |
| amazon_MiniLM_L6 | syntax_meta | with_spacy_decorr_t02 | 0.92 | 0.78 | 0.68 | 50 |
| amazon_MiniLM_L6 | syntax_meta | without_meta | 0.88 | 0.58 | 0.58 | 50 |
| amazon_MiniLM_L6 | train_vs_heldout | with_meta | 0.60 | 0.28 | 0.80 | 50 |
| amazon_MiniLM_L6 | train_vs_heldout | with_meta_decorr | 0.68 | 0.44 | 0.70 | 50 |
| amazon_MiniLM_L6 | train_vs_heldout | with_spacy_decorr | 0.72 | 0.46 | 0.72 | 50 |
| amazon_MiniLM_L6 | train_vs_heldout | without_meta | 0.78 | 0.36 | 0.46 | 50 |
