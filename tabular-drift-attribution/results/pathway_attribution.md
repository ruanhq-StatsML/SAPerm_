# 七天 pathway 归因

锚点是峰值日。后面七个 batch 是七次前向步骤。TabDDPM 只在峰值日上训练一次，来自 `yandex-research/tab-ddpm` 的 `GaussianMultinomialDiffusion`（commit `b476257`），然后对后面每一个 batch 逐块做条件生成，中间不重训。每一步只重写一个特征块，其余列在反向链的每一步被钉回该行的加噪过程。传给去噪网络的标签从峰值日的标签分布里抽。观测行在训练前被夹到 [-5, 5]，预测的干净样本也夹回这个盒子；不夹的话，这几份真实表上的反向链会离开训练范围。

预测损失是峰值日逻辑回归传到当天的 Brier；合成数据和 MovieLens 上还有峰值日 ridge 的连续目标 MSE。PO-risk 按 PermuCATE：在峰值日（W=0）与当前步（W=1）合并样本上拟合 outcome、倾向和 τ 模型，用双重稳健伪结果
\[
\phi(Z)=\frac{(Y-\mu_T)(T-\pi)}{\pi(1-\pi)}+(\mu_1-\mu_0),
\]
在评估矩阵上算 \(E[(\phi-\hat\tau(X))^2]\)。outcome 与 τ 用随机森林（150 棵树，深度 5），倾向用平衡逻辑回归，倾向裁剪到 [0.01, 0.99]。当前步评估时 \(W=1\)。把一块换成 TabDDPM 的条件样本之后，PO-risk 与 Brier 的变动记为 Δ。排名占比是 |Δ| 占全部块 |Δ| 之和。MMD 比的是这一块和峰值日同一块。HSIC 比的是这一块和当天标签。最后一天给 Δ 的 bootstrap 区间，MMD 和 HSIC 用 40 次置换，最小 p 值是 0.024。七步之间用符号翻转检验，看绝对变动最大的那一块是否大于其余每一块。

## 合成数据：两种方法指到同一块

第 0 天四块都是标准正态，各 4 列：`stable`、`context`、`driver`、`noise`。结果只用 `stable` 和 `driver`。之后七天里 `driver` 的均值最多移 1.5，它的系数同时加大；`context` 只有较小的位置移动，不进结果。无条件样本对峰值日的 MMD 是 0.000。

PO-risk 从 0.005 升到 0.050，潜变量 MSE 从 0.337 升到 4.496。到第 7 天，`driver` 占 PO-risk 变动的 0.905、Brier 变动的 0.959、MSE 变动的 0.936、MMD 的 0.834。它的 PO 区间是 [-0.122, -0.062]，Brier 区间是 [-0.139, -0.126]，两端都离开 0。七步上 `driver` 对另外三块的符号翻转 p 值，PO-risk、Brier、MSE 都不超过 0.021。`context` 的 MMD p 值是 0.024，Brier 变动停在 0.001。HSIC 的差都小于 0.001，排不出块。

| 步 | PO-risk | Brier | MSE | driver 的 PO 占比 | Brier 占比 | MSE 占比 | MMD 占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.005 | 0.236 | 0.337 | 0.471 | 0.420 | 0.492 | 0.863 |
| 2 | 0.014 | 0.231 | 0.403 | 0.791 | 0.672 | 0.548 | 0.886 |
| 3 | 0.017 | 0.212 | 0.486 | 0.822 | 0.650 | 0.689 | 0.905 |
| 4 | 0.026 | 0.195 | 0.901 | 0.821 | 0.868 | 0.810 | 0.920 |
| 5 | 0.031 | 0.182 | 1.427 | 0.336 | 0.822 | 0.826 | 0.864 |
| 6 | 0.032 | 0.158 | 2.589 | 0.923 | 0.911 | 0.921 | 0.852 |
| 7 | 0.050 | 0.122 | 4.496 | 0.905 | 0.959 | 0.936 | 0.834 |

第 5 步的 PO 占比掉到 0.336，Brier、MSE、MMD 仍然在 `driver` 上。第 6、7 步四项又都回到 0.83 以上。

## IEEE-CIS：协变量没有分开，Brier 停在 card

欺诈率的局部峰值在交易日第 173 天（0.0495），向前走到第 180 天（0.0374）。六块是 amount、card、address、counts、timedelta、hour，单调时钟不进特征。无条件样本对峰值日的 MMD 是 0.002。第 180 天没有任何一块的 MMD 达到 p < 0.05。

PO-risk 的七步平均绝对变动最大的是 timedelta，但它和 card、address、counts 的符号翻转 p 值都在 0.178 以上，最后一天六块的 PO 区间都含 0。传到当天的 Brier 变动最大的是 card。card 对 amount、address 的符号翻转 p 值是 0.008，对 counts 是 0.028；对 hour 是 0.054，对 timedelta 是 0.292。第 180 天 card 的 Brier 区间是 [-0.023, -0.004]，它自己的 MMD p 值是 1。这一周里，card 动的是欺诈模型的预测，协变量块没有分开。

## MovieLens：MMD 看用户和小时，Brier 看题材

高分率（评分至少 4）的局部峰值在 2001-01-01（0.615），连续七个自然日走到 2001-01-08（0.483）。评分是结果，不进特征。用户块是性别、年龄、职业、邮编首位；影片块是八个题材指示；上下文只留小时，因为星期几在同一天里是常数。峰值日在筛过至少 20 次评分的用户和影片之后有 633 行。无条件样本对峰值日的 MMD 是 0.004，替换块的均方差大约 1.7。

七天的平均 MMD 是小时 0.108、用户 0.100、题材 0.012。小时对题材的符号翻转 p 值是 0.016，小时和用户分不开（p = 0.413）。最后一天三块的 MMD p 值都落在置换下限 0.024，题材的 MMD 本身只有 0.006。

预测损失指向题材。Brier 的七天平均绝对变动是题材 0.010、小时 0.005、用户 0.004。题材对用户的符号翻转 p 值是 0.029，对小时是 0.074。最后一天题材的 Brier 区间是 [-0.025, -0.010]，用户的 Brier 区间含 0。评分 MSE 也是题材领先（0.044 对用户 0.027），对用户的符号翻转 p 值是 0.125。PO-risk 的平均绝对变动是用户 0.009、题材 0.005，两者的符号翻转 p 值是 0.198。最后一天用户的 PO 区间是 [-0.048, -0.008]，题材的 PO 区间含 0。HSIC 的差不超过 0.002。

这一周协变量里动的是谁在打分、几点打分。题材的边际几乎没动，Brier 仍然把题材排在前面，因为峰值日的评分模型用题材。

## Home Credit：申请号顺序上分布没动，Brier 停在外部评分

`application_train.csv` 没有事件时间。行按 `SK_ID_CURR` 排成 40 段。违约率的局部峰值在第 31 段（0.083），向前到第 38 段（0.080），整段违约率停在 0.08 附近。五块是 external（三列 `EXT_SOURCE`）、amount、personal、region、other。每段抽 700 行。无条件样本对峰值段的 MMD 是 0.003。

五块的 MMD 都在 0.005 以下。七步上 MMD 最大的 external 对其余每一块的符号翻转 p 值都不低于 0.266。最后一天只有 other 的 MMD p 值是 0.049，统计量本身是 0.004。

Brier 把 external 单独标出来。七天平均绝对变动是 0.023，对其余四块的符号翻转 p 值都是 0.008。最后一段的 Brier 区间是 [-0.044, -0.006]。PO-risk 分不开 external 和 amount（平均绝对变动 0.005 和 0.004，符号翻转 p = 0.253）。最后一段这两块的 PO 区间都不含 0，personal、region、other 的区间含 0。HSIC 的差都小于 0.001。

沿申请号往前走，协变量没有分开。峰值日的违约模型靠外部评分，所以把这一块换成条件样本之后，传到后面各段的 Brier 会动。

## 怎么分开 sampler bias 和归因

这是两套量，不要混读。

**生成器有没有偏**：看峰值日上的无条件 TabDDPM 样本对真实峰值的 MMD（`fidelity_mmd`），以及把某一块在峰值日上重画时，该块 imputation MMD（`anchor_impute_mmd`）。MMD 很小说明生成器没有整体漂走。

**相对贡献是不是 sampler 造成的**：对 PO-risk 和 Brier 都做峰值日 floor。floor 是只在峰值日上、用 TabDDPM 重画该块时的 Δ；excess = 最后一步 Δ 减 floor。`pathway_tables.tex` 的 **sampler-floor 表**同时给出 PO 与 Brier 的 floor、day、excess、raw share、excess share 和 verdict。带上这张表后，可以对业务说清楚三件事：**生成器**（重画就会动）、**pathway**（后面 batch 比重画还多出来的部分）、**模型依赖**（excess 大但协变量 MMD 不显著）。这比单看 raw share 稳得多——例如 MovieLens 用户块 PO raw share 只有 0.52，扣掉 floor 后 excess share 是 0.89，说明最后一步的 PO 排序里一大块其实是峰值重画就能解释的假象。

Verdict 规则（表内 Brier 列）：|excess| < |floor| → `sampler`；excess ≥ 2×|floor| 且 day MMD 的 p < 0.05 → `drift`；excess 够大但 MMD 不显著 → `model`；介于中间 → `thin`；两边都接近 0 → `none`。

### sampler-floor 表（最后一步，PO + Brier）

**合成**（fidelity MMD 0.000）：driver 的 PO excess share 0.971、Brier excess share 0.915，verdict `drift`——pathway 真漂移，不是生成器重画 alone。stable/noise 的 verdict 是 `sampler`。

| 块 | PO floor | PO day | PO excess | PO raw → excess share | Brier verdict |
| --- | --- | --- | --- | --- | --- |
| driver | -1.18 | -531 | -530 | 0.966 → 0.971 | drift |
| context | -1.46 | -15.7 | -14.3 | 0.029 → 0.026 | drift |
| stable | -1.27 | -1.65 | -0.37 | 0.003 → 0.001 | sampler |
| noise | -0.17 | -1.37 | -1.21 | 0.002 → 0.002 | sampler |

**IEEE-CIS**（fidelity 0.002）：Brier 上 card `sampler`、timedelta `thin`；PO raw 最大是 counts（0.39），excess 后仍是 counts（0.43），card 的 PO excess share 从 0.19 升到 0.29——生成器重画对 card 的 PO 解释力比 Brier 更明显。

**MovieLens**（fidelity 0.004）：用户 PO floor -14.96、day -3.46，excess +11.5，Brier verdict `sampler`——**Brier 上主要是生成器/pathway 混合，PO 上 raw 0.52 扣 floor 后 excess share 0.89**，业务要讲“用户块 PO 排序别只看 raw”。题材 Brier raw 0.59 → excess 0.27，verdict `sampler`。

**Home Credit**（fidelity 0.003）：external Brier excess share 0.51 → 0.17、verdict `sampler`（后面 slice 的 Brier 变动没超过峰值重画 external）；region verdict `model`（Brier excess 0.29，MMD 不显著）。PO 上 external excess share 0.50，仍最大但已扣掉 floor。

**两种归因是否一致**：Brier/MSE 看峰值日预测；PO-risk 看 DR 伪结果与 τ；MMD 看协变量。三者可以不一致——MovieLens 仍是 MMD 偏用户/小时、Brier 偏题材。

**给业务的三句话**：① `fidelity_mmd` + anchor impute MMD 过不过关（生成器）；② sampler-floor 里 excess share 谁排前面（pathway 相对贡献）；③ verdict 是 sampler 还是 drift/model（生成器 vs 真漂移 vs 模型敏感）。

表在 `latex/pathway_attribution.tex`，数字由 `results/pathway_tables.tex` 从 `results/pathway.json` 生成。四份一起重算用 `python -m pathway.run_pathway`。只补 MovieLens 和 Home Credit 用 `python -m pathway.run_pathway extra`。TabDDPM 仓库放在 `/tmp/tab-ddpm`。
