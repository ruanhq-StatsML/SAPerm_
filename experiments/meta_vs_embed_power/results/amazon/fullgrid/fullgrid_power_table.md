# Amazon Full-Grid DR / RR / CF (B=50, n_perm=150) + Diagnostics

| 编码器 | k | 场景 | 检验 | w/o | meta | Δmeta | spacy | Δspacy | vader4 | Δvader4 | meta_dec | Δm_dec | y_AUC (Z/M/S/V4/Md) | AUC_W (Z/M/S/V4/Md) | VarRatio (M/S/V4/Md) |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|---|
| MiniLM_L6 | full | sentiment_meta | DRPerm | 1.00 | — | — | — | — | — | — | — | — | 0.87/0.94/0.90/0.93/0.88 | 0.65/0.96/0.65/0.94/0.65 | 0.54/1.00/0.65/1.00 |
| MiniLM_L6 | full | sentiment_meta | RRPerm | 1.00 | — | — | — | — | — | — | — | — | 0.87/0.94/0.90/0.93/0.88 | 0.65/0.96/0.65/0.94/0.65 | 0.54/1.00/0.65/1.00 |
