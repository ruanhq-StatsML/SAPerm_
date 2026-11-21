# 正向 interaction 与 power：实务解释

核心不是公式本身，而是两件事分别回答什么、合在一起又回答什么。

---

## 一句话

| 量 | 实务问题 |
|---|---|
| **$\hat I>0$（正向）** | 这对模态是不是在**一起**解释 domain / outcome gap（协同，而不是各干各的）？ |
| **power** | 若真实世界真有这种成对结构，我们的检验**多常能检出来**？ |
| **signed power** | 检出来的时候，符号是不是也对（说成协同而不是搞反）？ |

**正向 ≠ power。**  
正向是*方向/机制*；power 是*可重复可检出*。交付要报的是二者的合取（signed power）。

---

## 1. 正向 interaction 在说什么（机制）

\[
I(a,b)=\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)
\]

项目口径（README）：$I$ 大 / 为正 $\Rightarrow$ **A 与 B 一起**比单独更能解释 batch 间差异（synergy，不是单纯相关）。

实务翻译（以 text×video 为例）：

- **$\hat I_{\mathrm{TV}}>0$**：字幕与画面的**耦合**在驱动漂移/预测残差；只留 text 或只留 video 都不够，**融合才对症**。
- **$\hat I_{\mathrm{TV}}<0$**：二者偏拮抗或冗余对消——叠在一起并不按“越加越好”工作。
- **$\hat I\approx$ null**：没有稳定的二阶结构（主效应另算，$\Delta_m$ / $\pi_m$）。

这和 `interaction_shift` 的种植一致：新域里给 $(a,b)$ 加 **product**（$X$ 侧耦合 + $Y$ 侧交叉项）→ 真值就是“协同的成对结构”→ 期望 $\hat I>0$。

---

## 2. Power 在说什么（可信度）

Power = 在 **$H_1$** = `interaction_shift` on $(a,b)$ 下，

\[
\mathbb{P}\bigl(p(|I|)<\alpha\bigr).
\]

实务翻译：

- **高 power**：这种成对协同若存在，检验**经常能报警**（方法对该机制敏感）。
- **低 power**：即使种了 product，也经常漏检（方法对该机制钝）。
- **null 上 ≈α**：没有成对结构时不乱报警（type-I）。

注意：power 用的是 $|I|$，**不看符号**。它只保证“有非零交互”，不保证“是正向协同”。

---

## 3. 二者关联：signed power（交付要用的）

\[
\text{signed power}
=\mathbb{P}_{H_1^{+}}
\bigl(p(|I|)<\alpha \;\wedge\; \hat I>0\bigr).
\]

| 情况 | 实务读法 |
|---|---|
| 高 power + 多数 $\hat I>0$ | **可靠地检出正向协同** → 可写“该 pair 需要联合建模 / 融合” |
| 高 power + 符号乱跳 | 能感到“有交互”，但**机制方向不稳** → 勿写死 synergy |
| $\hat I>0$ 但 p 大（低 power） | 点估计像协同，**统计上站不住** → 探索性，不能当结论 |
| null 上 power≈α | 校准过，正向结论不是假阳性堆出来的 |

所以关联是：

> **正向**回答“是什么机制”；**power**回答“这机制检不检得到”；  
> **signed power** = 实务上唯一能同时支撑“存在 + 方向”的交付量。

---

## 4. 和 mean_shift / concept-drift 怎么分工（避免误读）

| 场景 | 期望 | 实务 |
|---|---|---|
| `mean_shift`（只挪 video） | $\pi_{\mathrm{video}}$ 大；$I$ 的 power **低** | “单通道变了”，不是融合问题 |
| `interaction_shift`（种 text×video） | `interact_PO_text_video` **signed power 高**；错 pair 近 α | “问题在这对的耦合” |
| `concept_drift`（$X$ 不动） | 经典 TST≈0；CF/DR 有 power；PO-on-$X$ 交互安静 | 交互板讲的是 $X$/PO 结构，不是 $Y\mid X$ combo 覆盖 |

T×A / T×V 的 **CF/DR power** 是另一张表（可检测性），**不要**写成 signed interaction。

---

## 5. 交付句式（可直接用）

1. Under `interaction_shift` on $(a,b)$, **signed power** of `interact_PO_{a}_{b}` is the rate at which we both reject and obtain $\hat I>0$.  
2. That quantifies：*when cross-modal synergy is real, how often do we correctly call it synergy?*  
3. Pair with null type-I ≈α and a non-planted-pair specificity check.

Companion `crosscorr_{a}_{b}`：只说“$X$ 上两模态变依赖了”；**不带** PO 意义上的正/负协同。机制方向仍看 $\mathrm{sign}(\hat I)$。
