# MMD × ρ_PO × power：三模态对照论证

设定：ModaSense \(X=[M_t,M_i,M_a]\)。同一窗口下：

| 量 | 问的是什么 | 不需要 / 需要 Y |
| --- | --- | --- |
| **MMD²**（RBF，z-scored 块） | 这一模态的特征分布，batch0 vs batch1 差多大？ | 不需要 Y |
| **RF power** | 只用这一模态，能不能拒绝 shift（对 Y）？ | 需要 Y |
| **ρ_PO** | 拼进更多模态后，PO-risk 相对 text-base 变小还是变大？ | 需要 Y |

三者对齐时结论最硬；**不对齐本身就是 insight**。

## 经验表（smoke B=6；full 跑完后以 full 为准）

| scenario | MMD_t | MMD_i | MMD_a | RF t/i/a | ρ(TI/T) | 读法 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| text_sentiment | **0.101** | 0.026 | 0.034 | 高 / — | ~2 | MMD 点名 text；路由对齐 |
| image_brightness | 0.024 | **0.157** | 0.021 | ~0 | ~1.6 | MMD 点名 image；RF 仍弱 |
| audio_energy | 0.022 | 0.020 | **0.148** | ~0 | ~1.9 | MMD 点名 audio；RF 仍弱 |
| null | ~0.03 | ~0.03 | ~0.03 | 噪声 | — | 三模态 MMD 同阶 |

## 论证（四条）

### 1. MMD 回答「哪个模态在动」

窗口按 `vader_compound` / `img_brightness` / `aud_rms_energy` 切时，对应模态的 MMD 高一个数量级。  
这是**纯分布**证据：还没碰标签，就已经知道 shift 坐落在哪条通道。

### 2. Power 回答「动了是否够检测」

- Text 路由：MMD_text 高且 RF 高 → 该模态既移动又与 Y 相关，检测 power 直观上升。  
- Image / audio 路由：MMD 很高但 RF≈0 → **通道动了，但对当前 Y 不够构成拒绝**。  
  Insight：不能把「MMD 显著」直接当成「power 会涨」；MMD 是 covariate 移动，power 是带 Y 的检验。

### 3. ρ_PO 回答「拼进去有没有吃掉 PO 残差」

ρ = PO(rich)/PO(text)。

- ρ≪1：新模态进 nuisance，吸收 covariate（BiasFinder sentiment 的 0.25 是经典例）。  
- ρ≳1（本 smoke 上 image/audio 窗常见）：拼了视觉/音频**没有**把 PO 压下去，甚至变差 → 与「RF 仍弱」一致：该轴移动对 PO 通道不是有效吸收器。  
- 与 MMD 合读：MMD_image 大 + ρ≳1 + RF≈0 ⇒ **视觉分布在变，但对 outcome/PO 无益**——正是跨模态冲突数据上该有的故事。

### 4. 三角对照能多挖出什么

| 模式 | MMD | Power | ρ_PO | 业务解读 |
| --- | --- | --- | --- | --- |
| 对齐检测 | 路由模态高 | 同模态高 | 可吸收或可共定位 | 该通道既是漂移源也是监控 KPI |
| 空转漂移 | 路由模态高 | 低 | ≳1 | 通道在变，但别当质量事故乱报；先看 Y/PO |
| 吸收成功 | 中高 | DR 不高 | ≪1 | 拼进 meta 修好 nuisance（BiasFinder 情感） |
| 概念残差 | 各模态不高或 embed 主导 | RF/DR 高 | ≈1 | 问题不在可命名模态块（heldout） |

一句话：**MMD 定模态，power 定能不能报警，ρ_PO 定拼特征有没有把残差吃掉。**  
三模态 use-case 里，这三件事分开读，比单看某一个数清楚得多。
