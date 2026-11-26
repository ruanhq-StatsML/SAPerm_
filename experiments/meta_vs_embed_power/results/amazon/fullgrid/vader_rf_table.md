# Amazon VADER-Focused RFPerm B=150

## VADER 列与 y 的 Spearman 相关（Amazon gift / cross vs gender）

| 特征 | gift ρ(y) | cross ρ(y) | gender ρ(y) |
|---|--:|--:|--:|
| vader_neg | -0.505 | -0.508 | -0.460 |
| vader_neu | -0.525 | -0.450 | +0.029 |
| vader_pos | +0.666 | +0.637 | +0.366 |
| vader_compound | +0.532 | +0.574 | +0.435 |
| polar_gap | +0.688 | +0.683 | +0.478 |
| sentiment_std | -0.209 | -0.134 | -0.059 |

## VADER 各臂 RFPerm Power & Δ(arm − without_meta)

| 编码器 | k | 场景 | 臂 | power | Δ(w/o) | ord power | ord Δ | y_AUC | AUC_W | VarRatio |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| MiniLM_L6 | full | null_shift | without_meta | 0.02 | +0.00 | 0.04 | +0.00 | 0.87 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_meta | 0.05 | +0.03 | 0.02 | -0.02 | 0.94 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_spacy | 0.03 | +0.01 | 0.05 | +0.01 | 0.90 | 0.48 | 1.00 |
| MiniLM_L6 | full | null_shift | with_vader_all | 0.05 | +0.03 | 0.05 | +0.01 | 0.93 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_vader4 | 0.03 | +0.01 | 0.03 | -0.01 | 0.93 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_m_no_vader | 0.03 | +0.01 | 0.03 | -0.01 | 0.89 | 0.48 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_neg | 0.03 | +0.01 | 0.03 | -0.01 | 0.90 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_neu | 0.03 | +0.01 | 0.03 | -0.01 | 0.89 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_pos | 0.05 | +0.03 | 0.05 | +0.01 | 0.92 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_compound | 0.03 | +0.01 | 0.03 | -0.01 | 0.90 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_polar_gap | 0.04 | +0.02 | 0.04 | +0.00 | 0.92 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_sentiment_std | 0.03 | +0.01 | 0.03 | -0.01 | 0.88 | 0.49 | 1.00 |
| MiniLM_L6 | full | sentiment_meta | without_meta | 0.87 | +0.00 | 0.89 | +0.00 | 0.87 | 0.65 | 1.00 |
| MiniLM_L6 | full | sentiment_meta | with_meta | 0.91 | +0.04 | 0.85 | -0.03 | 0.94 | 0.96 | 0.54 |
| MiniLM_L6 | full | sentiment_meta | with_spacy | 0.93 | +0.07 | 0.93 | +0.05 | 0.90 | 0.65 | 1.00 |
| MiniLM_L6 | full | sentiment_meta | with_vader_all | 0.76 | -0.11 | 0.76 | -0.13 | 0.93 | 0.96 | 0.54 |
| MiniLM_L6 | full | sentiment_meta | with_vader4 | 0.83 | -0.04 | 0.83 | -0.06 | 0.93 | 0.94 | 0.65 |
| MiniLM_L6 | full | sentiment_meta | with_m_no_vader | 0.94 | +0.07 | 0.94 | +0.05 | 0.89 | 0.65 | 1.00 |
| MiniLM_L6 | full | sentiment_meta | with_col_vader_neg | 0.85 | -0.01 | 0.85 | -0.03 | 0.90 | 0.70 | 0.96 |
| MiniLM_L6 | full | sentiment_meta | with_col_vader_neu | 0.84 | -0.03 | 0.84 | -0.05 | 0.89 | 0.68 | 0.98 |
| MiniLM_L6 | full | sentiment_meta | with_col_vader_pos | 0.85 | -0.02 | 0.85 | -0.04 | 0.92 | 0.79 | 0.88 |
| MiniLM_L6 | full | sentiment_meta | with_col_vader_compound | 0.85 | -0.02 | 0.85 | -0.04 | 0.90 | 0.89 | 0.76 |
| MiniLM_L6 | full | sentiment_meta | with_col_polar_gap | 0.83 | -0.03 | 0.83 | -0.05 | 0.92 | 0.87 | 0.79 |
| MiniLM_L6 | full | sentiment_meta | with_col_sentiment_std | 0.87 | +0.00 | 0.87 | -0.02 | 0.88 | 0.65 | 1.00 |
| MiniLM_L6 | full | syntax_meta | without_meta | 0.52 | +0.00 | 0.24 | +0.00 | 0.87 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_meta | 0.41 | -0.11 | 0.07 | -0.17 | 0.94 | 0.99 | 0.55 |
| MiniLM_L6 | full | syntax_meta | with_spacy | 0.49 | -0.03 | 0.10 | -0.14 | 0.90 | 0.76 | 0.92 |
| MiniLM_L6 | full | syntax_meta | with_vader_all | 0.47 | -0.05 | 0.47 | +0.23 | 0.93 | 0.69 | 0.99 |
| MiniLM_L6 | full | syntax_meta | with_vader4 | 0.49 | -0.03 | 0.49 | +0.25 | 0.93 | 0.69 | 0.99 |
| MiniLM_L6 | full | syntax_meta | with_m_no_vader | 0.48 | -0.04 | 0.48 | +0.24 | 0.89 | 0.99 | 0.54 |
| MiniLM_L6 | full | syntax_meta | with_col_vader_neg | 0.55 | +0.03 | 0.55 | +0.31 | 0.90 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_col_vader_neu | 0.48 | -0.04 | 0.48 | +0.24 | 0.89 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_col_vader_pos | 0.51 | -0.01 | 0.51 | +0.27 | 0.92 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_col_vader_compound | 0.49 | -0.03 | 0.49 | +0.25 | 0.90 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_col_polar_gap | 0.51 | -0.01 | 0.51 | +0.27 | 0.92 | 0.68 | 1.00 |
| MiniLM_L6 | full | syntax_meta | with_col_sentiment_std | 0.51 | -0.01 | 0.51 | +0.27 | 0.88 | 0.68 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | without_meta | 0.23 | +0.00 | 0.15 | +0.00 | 0.87 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_meta | 0.22 | -0.01 | 0.19 | +0.04 | 0.94 | 0.54 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_spacy | 0.22 | -0.01 | 0.17 | +0.03 | 0.90 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_vader_all | 0.19 | -0.05 | 0.19 | +0.04 | 0.93 | 0.54 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_vader4 | 0.18 | -0.05 | 0.18 | +0.03 | 0.93 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_m_no_vader | 0.24 | +0.01 | 0.24 | +0.09 | 0.89 | 0.54 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_vader_neg | 0.20 | -0.03 | 0.20 | +0.05 | 0.90 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_vader_neu | 0.22 | -0.01 | 0.22 | +0.07 | 0.89 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_vader_pos | 0.19 | -0.04 | 0.19 | +0.05 | 0.92 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_vader_compound | 0.19 | -0.05 | 0.19 | +0.04 | 0.90 | 0.53 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_polar_gap | 0.20 | -0.03 | 0.20 | +0.05 | 0.92 | 0.54 | 1.00 |
| MiniLM_L6 | full | train_vs_heldout | with_col_sentiment_std | 0.19 | -0.04 | 0.19 | +0.05 | 0.88 | 0.53 | 1.00 |
| MiniLM_L6 | full | null_shift | without_meta | 0.06 | +0.00 | 0.01 | +0.00 | 0.86 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_meta | 0.03 | -0.03 | 0.01 | +0.00 | 0.93 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_spacy | 0.03 | -0.03 | 0.01 | +0.00 | 0.89 | 0.50 | 1.00 |
| MiniLM_L6 | full | null_shift | with_vader_all | 0.03 | -0.03 | 0.03 | +0.02 | 0.92 | 0.50 | 1.00 |
| MiniLM_L6 | full | null_shift | with_vader4 | 0.03 | -0.03 | 0.03 | +0.02 | 0.93 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_m_no_vader | 0.02 | -0.04 | 0.02 | +0.01 | 0.89 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_neg | 0.03 | -0.03 | 0.03 | +0.01 | 0.90 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_neu | 0.01 | -0.05 | 0.01 | +0.00 | 0.88 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_pos | 0.01 | -0.05 | 0.01 | -0.01 | 0.91 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_vader_compound | 0.02 | -0.04 | 0.02 | +0.01 | 0.91 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_polar_gap | 0.03 | -0.03 | 0.03 | +0.02 | 0.92 | 0.49 | 1.00 |
| MiniLM_L6 | full | null_shift | with_col_sentiment_std | 0.02 | -0.04 | 0.02 | +0.01 | 0.87 | 0.49 | 1.00 |
| MiniLM_L6 | full | domain_shift | without_meta | 0.57 | +0.00 | 0.53 | +0.00 | 0.86 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_meta | 0.62 | +0.05 | 0.41 | -0.13 | 0.93 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_spacy | 0.64 | +0.07 | 0.49 | -0.04 | 0.89 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_vader_all | 0.59 | +0.01 | 0.59 | +0.05 | 0.92 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_vader4 | 0.60 | +0.03 | 0.60 | +0.07 | 0.93 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_m_no_vader | 0.62 | +0.05 | 0.62 | +0.09 | 0.89 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_vader_neg | 0.63 | +0.05 | 0.63 | +0.09 | 0.90 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_vader_neu | 0.63 | +0.05 | 0.63 | +0.09 | 0.88 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_vader_pos | 0.66 | +0.09 | 0.66 | +0.13 | 0.91 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_vader_compound | 0.57 | +0.00 | 0.57 | +0.04 | 0.91 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_polar_gap | 0.62 | +0.05 | 0.62 | +0.09 | 0.92 | 0.95 | 1.00 |
| MiniLM_L6 | full | domain_shift | with_col_sentiment_std | 0.60 | +0.03 | 0.60 | +0.07 | 0.87 | 0.95 | 1.00 |
| qwen3-0.6b | full | null_shift | without_meta | 0.02 | +0.00 | 0.03 | +0.00 | 0.94 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_meta | 0.03 | +0.01 | 0.02 | -0.01 | 0.96 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_spacy | 0.03 | +0.01 | 0.04 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_vader_all | 0.03 | +0.01 | 0.03 | +0.00 | 0.96 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_vader4 | 0.03 | +0.01 | 0.03 | +0.00 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_m_no_vader | 0.03 | +0.01 | 0.03 | +0.00 | 0.94 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_neg | 0.05 | +0.03 | 0.05 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_neu | 0.04 | +0.02 | 0.04 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_pos | 0.05 | +0.03 | 0.05 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_compound | 0.04 | +0.02 | 0.04 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_polar_gap | 0.05 | +0.03 | 0.05 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_sentiment_std | 0.04 | +0.02 | 0.04 | +0.01 | 0.94 | 0.49 | 1.00 |
| qwen3-0.6b | full | sentiment_meta | without_meta | 0.94 | +0.00 | 0.95 | +0.00 | 0.94 | 0.70 | 1.00 |
| qwen3-0.6b | full | sentiment_meta | with_meta | 0.93 | -0.01 | 0.90 | -0.05 | 0.96 | 0.93 | 0.68 |
| qwen3-0.6b | full | sentiment_meta | with_spacy | 0.95 | +0.01 | 0.93 | -0.02 | 0.95 | 0.70 | 1.00 |
| qwen3-0.6b | full | sentiment_meta | with_vader_all | 0.91 | -0.03 | 0.91 | -0.04 | 0.96 | 0.93 | 0.68 |
| qwen3-0.6b | full | sentiment_meta | with_vader4 | 0.93 | -0.01 | 0.93 | -0.02 | 0.95 | 0.89 | 0.78 |
| qwen3-0.6b | full | sentiment_meta | with_m_no_vader | 0.93 | -0.01 | 0.93 | -0.02 | 0.94 | 0.70 | 1.00 |
| qwen3-0.6b | full | sentiment_meta | with_col_vader_neg | 0.93 | -0.01 | 0.93 | -0.03 | 0.95 | 0.72 | 0.98 |
| qwen3-0.6b | full | sentiment_meta | with_col_vader_neu | 0.93 | -0.01 | 0.93 | -0.03 | 0.95 | 0.71 | 0.99 |
| qwen3-0.6b | full | sentiment_meta | with_col_vader_pos | 0.92 | -0.02 | 0.92 | -0.03 | 0.95 | 0.77 | 0.93 |
| qwen3-0.6b | full | sentiment_meta | with_col_vader_compound | 0.93 | -0.01 | 0.93 | -0.03 | 0.95 | 0.84 | 0.86 |
| qwen3-0.6b | full | sentiment_meta | with_col_polar_gap | 0.93 | -0.01 | 0.93 | -0.03 | 0.95 | 0.82 | 0.87 |
| qwen3-0.6b | full | sentiment_meta | with_col_sentiment_std | 0.93 | -0.01 | 0.93 | -0.03 | 0.94 | 0.70 | 1.00 |
| qwen3-0.6b | full | syntax_meta | without_meta | 0.14 | +0.00 | 0.01 | +0.00 | 0.94 | 0.77 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_meta | 0.16 | +0.02 | 0.01 | +0.00 | 0.96 | 0.95 | 0.71 |
| qwen3-0.6b | full | syntax_meta | with_spacy | 0.13 | -0.01 | 0.01 | +0.00 | 0.95 | 0.79 | 0.97 |
| qwen3-0.6b | full | syntax_meta | with_vader_all | 0.15 | +0.01 | 0.15 | +0.14 | 0.96 | 0.77 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_vader4 | 0.15 | +0.01 | 0.15 | +0.14 | 0.95 | 0.77 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_m_no_vader | 0.15 | +0.01 | 0.15 | +0.14 | 0.94 | 0.96 | 0.70 |
| qwen3-0.6b | full | syntax_meta | with_col_vader_neg | 0.15 | +0.01 | 0.15 | +0.14 | 0.95 | 0.76 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_col_vader_neu | 0.15 | +0.01 | 0.15 | +0.15 | 0.95 | 0.76 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_col_vader_pos | 0.15 | +0.01 | 0.15 | +0.14 | 0.95 | 0.76 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_col_vader_compound | 0.15 | +0.01 | 0.15 | +0.15 | 0.95 | 0.76 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_col_polar_gap | 0.14 | +0.00 | 0.14 | +0.13 | 0.95 | 0.76 | 1.00 |
| qwen3-0.6b | full | syntax_meta | with_col_sentiment_std | 0.14 | +0.00 | 0.14 | +0.13 | 0.94 | 0.76 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | without_meta | 0.16 | +0.00 | 0.15 | +0.00 | 0.94 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_meta | 0.17 | +0.01 | 0.18 | +0.03 | 0.96 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_spacy | 0.15 | -0.01 | 0.17 | +0.03 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_vader_all | 0.17 | +0.01 | 0.17 | +0.02 | 0.96 | 0.56 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_vader4 | 0.17 | +0.01 | 0.17 | +0.02 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_m_no_vader | 0.16 | +0.00 | 0.16 | +0.01 | 0.94 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_vader_neg | 0.18 | +0.02 | 0.18 | +0.03 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_vader_neu | 0.17 | +0.01 | 0.17 | +0.02 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_vader_pos | 0.17 | +0.01 | 0.17 | +0.03 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_vader_compound | 0.17 | +0.01 | 0.17 | +0.03 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_polar_gap | 0.18 | +0.02 | 0.18 | +0.03 | 0.95 | 0.55 | 1.00 |
| qwen3-0.6b | full | train_vs_heldout | with_col_sentiment_std | 0.18 | +0.02 | 0.18 | +0.03 | 0.94 | 0.55 | 1.00 |
| qwen3-0.6b | full | null_shift | without_meta | 0.05 | +0.00 | 0.00 | +0.00 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_meta | 0.07 | +0.01 | 0.01 | +0.01 | 0.96 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_spacy | 0.07 | +0.02 | 0.01 | +0.01 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_vader_all | 0.07 | +0.01 | 0.07 | +0.07 | 0.96 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_vader4 | 0.06 | +0.01 | 0.06 | +0.06 | 0.96 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_m_no_vader | 0.07 | +0.01 | 0.07 | +0.07 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_neg | 0.07 | +0.01 | 0.07 | +0.07 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_neu | 0.07 | +0.01 | 0.07 | +0.07 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_pos | 0.05 | +0.00 | 0.05 | +0.05 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_vader_compound | 0.06 | +0.01 | 0.06 | +0.06 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_polar_gap | 0.05 | +0.00 | 0.05 | +0.05 | 0.95 | 0.49 | 1.00 |
| qwen3-0.6b | full | null_shift | with_col_sentiment_std | 0.06 | +0.01 | 0.06 | +0.06 | 0.94 | 0.49 | 1.00 |
| qwen3-0.6b | full | domain_shift | without_meta | 0.75 | +0.00 | 0.54 | +0.00 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_meta | 0.67 | -0.08 | 0.52 | -0.02 | 0.96 | 0.97 | 1.01 |
| qwen3-0.6b | full | domain_shift | with_spacy | 0.72 | -0.03 | 0.57 | +0.03 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_vader_all | 0.73 | -0.02 | 0.73 | +0.19 | 0.96 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_vader4 | 0.73 | -0.03 | 0.73 | +0.19 | 0.96 | 0.97 | 1.01 |
| qwen3-0.6b | full | domain_shift | with_m_no_vader | 0.72 | -0.03 | 0.72 | +0.18 | 0.95 | 0.97 | 1.01 |
| qwen3-0.6b | full | domain_shift | with_col_vader_neg | 0.75 | -0.01 | 0.75 | +0.21 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_col_vader_neu | 0.73 | -0.03 | 0.73 | +0.19 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_col_vader_pos | 0.75 | +0.00 | 0.75 | +0.21 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_col_vader_compound | 0.73 | -0.02 | 0.73 | +0.19 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_col_polar_gap | 0.75 | -0.01 | 0.75 | +0.21 | 0.95 | 0.97 | 1.00 |
| qwen3-0.6b | full | domain_shift | with_col_sentiment_std | 0.74 | -0.01 | 0.74 | +0.20 | 0.94 | 0.97 | 1.00 |
