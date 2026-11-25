# ModaSense 三模态：CFPerm 子集 → 业务逻辑

冲突率 text≠image / text≠audio / image≠audio = 1.

| scenario | top block | drivers | 业务价值 |
| --- | --- | --- | --- |
| text_sentiment | `text_meta` | vader_neg, compound, pos, polar_gap, neu | 文案情绪 KPI 告警，不牵动视听 |
| image_brightness | `image_meta` | brightness, hue_std, contrast, sat | 画面明暗/色温抽检，勿误判文案 |
| audio_energy | (weak) | — | 本窗 LOCO 信号弱；响度轴需更大 B |

**text_sentiment（CF text reject）**

| feature | 语义 | 动作 |
| --- | --- | --- |
| vader_neg | 负向质量 | 负面升温 |
| vader_compound | 综合极性 | 情绪路由 |
| vader_pos | 正向质量 | 正向占比 |
| vader_polar_gap | 正负冲突 | 矛盾/讽刺候选 |
| vader_neu | 中性/含糊 | 暧昧语气 |

**image_brightness**

| feature | 语义 | 动作 |
| --- | --- | --- |
| img_brightness | 明暗 | 氛围抽检 |
| img_hue_std | 色相离散 | 色彩多样性 |
| img_contrast | 对比度 | 视觉冲击 |
| img_sat_mean | 饱和度 | 色彩强度 |

一句话：CFPerm 点名的是可执行通道（文案 / 画面 / 配音），不是匿名维。
