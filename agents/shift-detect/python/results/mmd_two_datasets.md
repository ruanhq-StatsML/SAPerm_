# Two-dataset argument: MMD × ρ_PO × power

General-purpose shell: modality arrays in → MMD / power / ρ_PO out.
Same read on BiasFinder (embed∥meta) and ModaSense (text∥image∥audio).

## BiasFinder (embed vs meta)

| scenario | MMD embed | MMD meta | fine-top | RF e→e+m | ρ_PO | label |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| null_shift | 0.0159 | 0.0153 | embed | 0.03→0.03 | 1.07 | null_calibration |
| sentiment_meta | 0.102 | 0.151 | vader_sent | 0.02→0.08 | 0.25 | absorb_interaction |
| syntax_meta | 0.0501 | 0.119 | spacy_syntax | 0.11→0.07 | 0.19 | localize_interaction |
| train_vs_heldout | 0.0351 | 0.024 | embed | 1.00→1.00 | 1.03 | concept_residual |

## ModaSense (text / image / audio)

| scenario | MMD t/i/a | top | RF text→MM | ρ_PO |
| --- | --- | --- | ---: | ---: |
| null_shift | 0.0165/0.0157/0.0145 | **text** | 0.01→0.01 | 2.14 |
| text_sentiment_shift | 0.326/0.0112/0.0203 | **text** | 1.00→1.00 | 1.95 |
| image_brightness_shift | 0.0164/0.57/0.0136 | **image** | 0.00→0.05 | 0.62 |
| audio_energy_shift | 0.0115/0.014/0.542 | **audio** | 0.00→0.00 | 1.06 |

## Insights（两库同一套读法）

1. **MMD 定通道**：BiasFinder 上 sentiment/syntax 窗 meta（及 vader/syntax 细块）MMD 抬升；ModaSense 上路由模态 MMD 高一个数量级（text 0.33 / image 0.58 / audio 0.55），null 三模态同阶。
2. **Power 定能不能检**：BiasFinder heldout RF=1（概念残差）；sentiment 小涨。ModaSense text 窗 RF=1；image/audio 窗 MMD 高但 RF≈0 —— 动了 ≠ 能报警。
3. **ρ_PO 定拼特征有没有吸收**：BiasFinder sentiment/syntax ρ≈0.25/0.19（absorb）；heldout ρ≈1.03（residual）。ModaSense image ρ=0.62（吸收）；text ρ≈1.95（已检满，拼图不降 PO）。
4. **General-purpose**：两库都是 `{modality: array}` → 同三角对照；换任意多塔模型只需换数组。
