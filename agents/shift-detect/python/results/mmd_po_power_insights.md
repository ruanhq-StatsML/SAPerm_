# Insights: MMD × ρ_PO × power（ModaSense 三模态）

MMD：B=50，z-scored 块，batch0 vs batch1。  
Power / ρ_PO：`power_mm_modasense_full`（B=80）+ smoke 对照。

| scenario | MMD t / i / a | MMD 顶点 | RF (text→MM) | ρ_PO (MM/text) | 一句话 |
| --- | --- | --- | ---: | ---: | --- |
| null | 0.015 / 0.015 / 0.015 | — | 0.01→0.01 | 2.14 | 三模态 MMD 同阶，无漂移 |
| text_sentiment | **0.33** / 0.012 / 0.020 | **text** | **1.00→1.00** | 1.95 | 通道动了且能检；拼图不吸收 |
| image_brightness | 0.017 / **0.58** / 0.014 | **image** | 0.00→0.05 | **0.62** | 通道大动，RF 弱；拼进图像 **吸收** |
| audio_energy | 0.011 / 0.016 / **0.55** | **audio** | ~0（smoke） | ≳1（smoke） | 通道大动，尚未变成检测 power |

## Insights（够用）

1. **MMD 只回答「哪条模态的特征分布在动」** — 路由轴与 MMD 顶点一一对齐（text/image/audio），比看 power 更干净。  
2. **Power 回答「动了能不能拒绝」** — text 窗：高 MMD + RF=1；image/audio 窗：MMD 很高但 RF≈0。分布漂移 ≠ 检测 power。  
3. **ρ_PO 回答「拼进该模态有没有吃掉 PO 残差」** — image 窗 ρ=0.62（吸收成功）；text 窗 ρ≈1.95（已检满，拼图不降 PO）。  
4. **三者合读**：MMD 定模态 → power 定是否报警 → ρ_PO 定 concat 是否值得当 nuisance。不对齐本身就是结论（例如「画面在变但别当文案事故」）。
