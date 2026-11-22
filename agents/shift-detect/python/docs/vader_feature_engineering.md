# VADER feature-engineering：为什么这样建、怎么论证

## 1. 在整条 digestion 里 VADER 扮演什么

```
read → preprocess → embed(Z) → meta(M) → schema indices → test / localize / subset
                              ↑
                         vader_sent block
```

BERT/`Z` 是稠密语义；**VADER 是显式、可命名的情感通道**。  
我们不拿 VADER 替换 embedding，而是把它做成 meta block，使得：

- localization 能点名 `vader_sent`（而不是 PCA 第 17 维）；
- subset 能点名 `vader_compound` / `vader_neg` / …；
- \(\rho_{PO}\) 能检验：情感 covariate 是否被这块吸收。

## 2. 具体造了哪些维（block = `vader_sent`）

| 特征 | 定义 | 工程含义 |
| --- | --- | --- |
| `vader_compound` | VADER 综合极性 \([-1,1]\) | 主情感轴；窗口路由常用 |
| `vader_pos` | 正部分强度 | 正极性质量 |
| `vader_neu` | 中性质量 | 中性/含糊 |
| `vader_neg` | 负部分强度 | 负极性质量 |
| `vader_polar_gap` | \(\|pos-neg\|\) | 两极分化程度（非简单 compound） |

实现（封装好的 lexicon+规则，非自训模型）：

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
s = SentimentIntensityAnalyzer().polarity_scores(text)
# → compound, pos, neu, neg; polar_gap = |pos - neg|
```

**为何用现成 VADER 而不是自训情感头？**

1. **Portable / 可复现**：无 GPU、无标签、版本钉死即可；
2. **可解释**：输出直接是情感语义，适合 CFPerm 命名 block；
3. **与 \(Z\) 正交的信息假设可检验**：若 VADER 只是 BERT 的冗余，则 \(\Delta\)CF≈0 且 \(\rho_{PO}≈1\)；经验上 sentiment 路由下 \(\rho_{PO}≈0.25\)，说明有增量交互。

## 3. 论证逻辑（对应三个统计命题）

### A. 作为检测特征 → power increase

在 **feature-only**、情感对齐的 DGP（`sentiment_shift`）上：

- baseline（长度等表面）power 弱；
- 加上 `vader_sent` 后 power ↑（如 0.40→1.00）。

命题：*当 shift 落在极性结构上，VADER 坐标提高拒绝率，且 null 上 Type-I 受控。*

### B. 作为交互通道 → \(\rho_{PO}\)

Real-data `sentiment_meta`（按 `vader_compound` 切窗）：

\[
\rho_{PO}=\frac{PO(Z{+}M)}{PO(Z)}\approx 0.25
\]

DR power 两侧≈0：不是“更响的警报”，而是 **VADER 进入 nuisance 后吸收了情感 covariate**。  
命题：*with/without VADER-meta 的 PO 比是 embed×情感 交互证据。*

### C. 作为 localization / subset → 驱动情感变动的特征

Digestion 后：

1. Block LOCO：情感场景下 carrier 指向 `vader_*`；
2. Subset：`vader_compound`, `vader_neg`, `vader_polar_gap`, `vader_neu`。

命题：*驱动情感 distribution shift 的 meta subset 可被置换/carrier 检验点名，而不是事后讲故事。*

若问“更负的文档贡献更大吗？”——应对 `vader_neg` 或阈值 \(\mathbf{1}\{compound<\tau\}\) 做同样 LOCO，而不是只报均值。

## 4. 设计选择（论证里常被问的）

| 选择 | 理由 |
| --- | --- |
| 保留 pos/neu/neg + compound | compound 是总结；分量用于 subset（负向驱动 vs 中性稀释） |
| 加 `polar_gap` | 捕捉“又正又负”的冲突，compound 会抵消 |
| 与 spaCy 分 block | 情感 vs 句法/NER 可分开 localize，避免一锅粥 |
| 不做领域重训 | 评测/归因要稳定；领域适配应作为**显式竞争方法**，不是默认可变 |

## 5. 一句话

VADER 特征工程的目的不是提高分类 accuracy，而是提供 **可检验、可命名的情感坐标**，从而在 data-digestion 之后同时支撑：**power increase**、**PO 交互证据**、**情感驱动 subset localization**。
