# ModaSense 三模态专题：子集 × 业务逻辑 × 价值

冲突率 text≠image/audio/image≠audio = 1/1/1

## text_sentiment_shift
- Top block: `text_meta`
- Drivers: ['vader_neg', 'vader_compound', 'vader_pos', 'pos_adj', 'vader_polar_gap', 'vader_neu', 'pos_noun']
- Value: 文本模态被点名；驱动维 ['vader_neg', 'vader_compound', 'vader_pos', 'pos_adj', 'vader_polar_gap', 'vader_neu', 'pos_noun'] 直接对应情绪监控 KPI（极性、负面质量、冲突）。业务动作：对文案通道告警并回放样本，不牵动视觉/音频产线。

| feature | 业务语义 | 业务动作 |
| --- | --- | --- |
| `vader_neg` | 负向情感质量 | 负面情绪升温检测 |
| `vader_compound` | 文本综合情感极性 | 情绪路由 / 情感漂移告警轴 |
| `vader_pos` | 正向情感质量 | 正向内容占比监控 |
| `pos_adj` | (unnamed meta) | 需人工释义 |
| `vader_polar_gap` | 正负冲突强度 | 矛盾表述 / 讽刺候选 |
| `vader_neu` | 中性/含糊质量 | 回避与暧昧语气识别 |
| `pos_noun` | (unnamed meta) | 需人工释义 |

Blocks:
- `text_meta`: imp=0.3998, p=0.012, rej=True
- `image_meta`: imp=0.03193, p=0.012, rej=True
- `audio_meta`: imp=0, p=1.000, rej=False

## image_brightness_shift
- Top block: `image_meta`
- Drivers: ['img_brightness', 'img_hue_std', 'img_contrast', 'img_sat_mean']
- Value: 图像模态被点名；驱动维 ['img_brightness', 'img_hue_std', 'img_contrast', 'img_sat_mean'] 对应画面氛围 KPI。业务动作：抽检视觉素材明暗/色温变更，避免把画面问题误判成文案情感事故。

| feature | 业务语义 | 业务动作 |
| --- | --- | --- |
| `img_brightness` | 画面明暗 | 视觉氛围明/暗切换 |
| `img_hue_std` | 色相离散 | 色彩多样性 |
| `img_contrast` | 对比度 | 视觉冲击 / 清晰度 |
| `img_sat_mean` | 饱和度 | 色彩情绪强度 |

Blocks:
- `image_meta`: imp=5.192e-05, p=0.383, rej=False
- `text_meta`: imp=4.222e-05, p=0.975, rej=False
- `audio_meta`: imp=3.159e-05, p=0.383, rej=False

## audio_energy_shift
- Top block: `text_meta`
- Drivers: ['avg_token_len', 'flesch_proxy', 'pos_verb', 'verb_noun_ratio', 'vader_pos', 'stopword_ratio', 'vader_compound', 'pos_noun', 'type_token_ratio', 'vader_neu', 'pos_adj', 'vader_polar_gap']
- Value: 文本模态被点名；驱动维 ['avg_token_len', 'flesch_proxy', 'pos_verb', 'verb_noun_ratio', 'vader_pos', 'stopword_ratio', 'vader_compound', 'pos_noun', 'type_token_ratio', 'vader_neu', 'pos_adj', 'vader_polar_gap'] 直接对应情绪监控 KPI（极性、负面质量、冲突）。业务动作：对文案通道告警并回放样本，不牵动视觉/音频产线。

| feature | 业务语义 | 业务动作 |
| --- | --- | --- |
| `avg_token_len` | (unnamed meta) | 需人工释义 |
| `flesch_proxy` | (unnamed meta) | 需人工释义 |
| `pos_verb` | (unnamed meta) | 需人工释义 |
| `verb_noun_ratio` | 动名比 | 叙事动态 vs 名物化 |
| `vader_pos` | 正向情感质量 | 正向内容占比监控 |
| `stopword_ratio` | (unnamed meta) | 需人工释义 |
| `vader_compound` | 文本综合情感极性 | 情绪路由 / 情感漂移告警轴 |
| `pos_noun` | (unnamed meta) | 需人工释义 |
| `type_token_ratio` | (unnamed meta) | 需人工释义 |
| `vader_neu` | 中性/含糊质量 | 回避与暧昧语气识别 |
| `pos_adj` | (unnamed meta) | 需人工释义 |
| `vader_polar_gap` | 正负冲突强度 | 矛盾表述 / 讽刺候选 |

Blocks:
- `text_meta`: imp=0.0001234, p=0.914, rej=False
- `image_meta`: imp=2.496e-06, p=0.531, rej=False
- `audio_meta`: imp=0, p=1.000, rej=False
