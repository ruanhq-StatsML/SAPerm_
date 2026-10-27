# 10-site 异构概念漂移：primary driver 识别

真实协变量来自 WhyShift 使用的 ACS 2018 加州 person 文件。ACS Income 用官方 10 个协变量，标签是收入是否超过 5 万美元。ACS Public Coverage 用官方 19 个协变量，标签是公共医疗保险。每个 replicate 把样本随机切成 10 个 site，因此 site 之间共享同一个经验分布 P(X)，协变量偏移不是这个基准要测的东西。

时间是合成的 batch 序号。前 `burn=4` 个 batch 所有 site 都使用真实基线标签。从 `t*=8` 起：

- 4 个 clean site 继续使用基线标签。
- 3 个 linear site 只在 `X1 >` 全局分位数的子组里，把标签改成 site 特定的线性半空间。三个方向分别是 `(1, 0.35)`、`(0.25, 1)`、`(-0.85, 0.55)`，方向再按与基线标签更不一致的一侧取向。
- 2 个 nonlinear site 在激活后的全部样本上改写标签，分别是 `X1 X2` 的乘积阈值和 `sin(X1) + X3^2`。
- 1 个 noise site 把激活后的标签换成 Bernoulli(0.5)。它是共识破坏者，不是 primary driver。

结构化规则在被改写的行上匹配基线正例率，所以边际标签偏移不能代替概念漂移。Noise site 故意不匹配。

## Δ_cf 在比较什么

参考模型只用 burn-in。每个 site 在最后 6 个 batch 上做交叉拟合。Δ_cf 是等样本量参考模型与本地模型的 Brier 差，再乘上本地模型相对抛硬币的技能 `max(2(AUC-0.5), 0)`。等样本量是为了不让“大参考模型对小本地模型”这条样本量差距本身变成漂移证据。技能门是为了把 noise site 从 primary driver 里拿掉：noise 的本地概念没有可学的结构，交叉拟合 AUC 接近 0.5。

线性规则只改写 `X1` 高于中位数的子组。WKHP 这类离散变量在中位数上有大量打结，`X1 > median` 往往远小于一半样本，整窗 Brier 会被没改写的行稀释。所以 headline Δ_cf 取三个预先指定区域里的最大值：全样本、burn-in 上 `X1` 的上尾、以及它的补集。区域只用协变量，不用标签。`delta_cf_global` 是只看全样本的版本，用来对照这种稀释。

PO τ² 是同一对预测的均方差，对应伪结果分解里的概念项。不乘技能时，noise site 也会很大。RF domain AUC 和 MMD 是协变量阴性对照，在这个 DGP 里不应当把 CD site 排到 clean site 前面。

## 配对一致率

一致率是 `(CD site, clean site)` 配对里 `score(CD) > score(clean)` 的比例，对 seed 取平均。目标是 Δ_cf 高于 0.8。

| 数据集 | 方法 | CD > clean | CD > noise | 线性 CD > clean | 非线性 CD > clean |
| --- | --- | --- | --- | --- | --- |
| acs_income | `delta_cf` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `delta_cf_global` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `tau_l2_gated` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `pareto_score` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `tau_l2` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `excess_pooled` | 1.000 (min 1.000) | 0.400 (min 0.400) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_income | `label_tvd` | 0.500 (min 0.100) | 0.050 (min 0.000) | 0.500 (min 0.083) | 0.500 (min 0.125) |
| acs_income | `c2st_joint` | 0.850 (min 0.800) | 0.350 (min 0.200) | 0.750 (min 0.667) | 1.000 (min 1.000) |
| acs_income | `domain_auc_rf` | 0.487 (min 0.400) | 0.500 (min 0.000) | 0.521 (min 0.333) | 0.438 (min 0.250) |
| acs_income | `mmd` | 0.412 (min 0.300) | 0.500 (min 0.000) | 0.375 (min 0.167) | 0.469 (min 0.125) |
| acs_pubcov | `delta_cf` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `delta_cf_global` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `tau_l2_gated` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `pareto_score` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `tau_l2` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `excess_pooled` | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| acs_pubcov | `label_tvd` | 0.613 (min 0.350) | 0.000 (min 0.000) | 0.604 (min 0.417) | 0.625 (min 0.250) |
| acs_pubcov | `c2st_joint` | 0.950 (min 0.850) | 0.450 (min 0.200) | 0.917 (min 0.750) | 1.000 (min 1.000) |
| acs_pubcov | `domain_auc_rf` | 0.438 (min 0.350) | 0.200 (min 0.000) | 0.396 (min 0.167) | 0.500 (min 0.250) |
| acs_pubcov | `mmd` | 0.600 (min 0.500) | 0.200 (min 0.000) | 0.562 (min 0.417) | 0.656 (min 0.375) |

Pareto 轴是 Δ_cf 和技能门控后的 PO τ²。秩是严格支配该 site 的其他 site 数加一。表里的 `pareto_score` 是秩的相反数，所以一致率仍然是“CD 排在 clean 前面”的配对比例。

## 漂移是否真的写进去了

| 数据集 | site 类型 | 激活后翻转率 | 规则与基线的一致率 | 激活后正例率 | 基线正例率 |
| --- | --- | --- | --- | --- | --- |
| acs_income | clean | 0.000 | 1.000 | 0.406 | 0.406 |
| acs_income | linear | 0.131 | 0.427 | 0.400 | 0.414 |
| acs_income | noise | 0.498 | 1.000 | 0.491 | 0.410 |
| acs_income | nonlinear | 0.555 | 0.436 | 0.396 | 0.420 |
| acs_pubcov | clean | 0.000 | 1.000 | 0.369 | 0.367 |
| acs_pubcov | linear | 0.259 | 0.476 | 0.378 | 0.372 |
| acs_pubcov | noise | 0.496 | 1.000 | 0.495 | 0.362 |
| acs_pubcov | nonlinear | 0.492 | 0.506 | 0.371 | 0.374 |

## 时间定位

在线规则用每条曲线的前两个点做校准，阈值是校准最大值加 0.01。命中指估计的 batch 与 `t*` 相差不超过 2。导数法报告的是 Δ_cf 一阶差分最大的窗口末端。

| 数据集 | site 类型 | 在线命中率 | 误报率 | 在线 MAE | 导数 MAE |
| --- | --- | --- | --- | --- | --- |
| acs_income | clean | 0.062 | 0.375 | 4.167 | 4.062 |
| acs_income | linear | 0.833 | 0.000 | 1.917 | 2.833 |
| acs_income | noise | 0.250 | 0.000 | 4.000 | 3.250 |
| acs_income | nonlinear | 1.000 | 0.000 | 1.625 | 3.000 |
| acs_pubcov | clean | 0.062 | 0.125 | 3.000 | 3.875 |
| acs_pubcov | linear | 1.000 | 0.000 | 1.417 | 3.000 |
| acs_pubcov | noise | 0.000 | 0.000 | 3.000 | 4.000 |
| acs_pubcov | nonlinear | 1.000 | 0.000 | 1.500 | 3.000 |

## 线性 / 非线性类型分辨

概念组是预先指定的三组充分统计量：`(X1, X2)`、`X1 X2`、`(sin X1, X3^2)`。组 VIMP 是该组交叉拟合 Brier 相对截距模型的下降。类型判定取 VIMP 最大的组。`type_call_all` 用最近窗口的全部行；`type_call_sub` 只用 `X1` 高于全局分位数的子组。准确率只在真实 CD site 上计算。Top-5 精度是按 Δ_cf 取出的 5 个 site 里真正是 CD site 的比例。

| 数据集 | 判定 | 类型准确率 | Top-5 精度 |
| --- | --- | --- | --- |
| acs_income | `type_call_all` | 0.700 | 1.000 |
| acs_income | `type_call_sub` | 0.750 | 1.000 |
| acs_pubcov | `type_call_all` | 0.700 | 1.000 |
| acs_pubcov | `type_call_sub` | 0.900 | 1.000 |

| 判定 | 真实机制 | 召回 |
| --- | --- | --- |
| `type_call_all` | linear | 0.500 |
| `type_call_all` | product | 1.000 |
| `type_call_all` | sin_square | 1.000 |
| `type_call_sub` | linear | 0.917 |
| `type_call_sub` | product | 0.375 |
| `type_call_sub` | sin_square | 1.000 |

## 特征支撑恢复

每个 CD site 的种植支撑：线性和乘积是 `{X1, X2}`，正弦平方是 `{X1, X3}`。AUROC 是该 site 的特征分数对这个 0/1 支撑的排序。`shift_vimp` 是概念间隙 `E[(μ_local-μ_ref)^2]` 在条件置换该坐标后的下降。`delta_pred_vimp` 是最近窗口预测型置换重要度减去 burn-in 的同一重要度。`l1_coef` 是最近窗口 L1 逻辑回归系数的绝对值。

| 数据集 | site 类型 | 分数 | AUROC |
| --- | --- | --- | --- |
| acs_income | linear | `delta_pred_vimp` | 0.651 |
| acs_income | linear | `l1_coef` | 0.786 |
| acs_income | linear | `pred_vimp` | 0.927 |
| acs_income | linear | `shift_vimp` | 0.906 |
| acs_income | nonlinear | `delta_pred_vimp` | 1.000 |
| acs_income | nonlinear | `l1_coef` | 0.930 |
| acs_income | nonlinear | `pred_vimp` | 1.000 |
| acs_income | nonlinear | `shift_vimp` | 0.789 |
| acs_pubcov | linear | `delta_pred_vimp` | 0.944 |
| acs_pubcov | linear | `l1_coef` | 0.841 |
| acs_pubcov | linear | `pred_vimp` | 0.995 |
| acs_pubcov | linear | `shift_vimp` | 0.953 |
| acs_pubcov | nonlinear | `delta_pred_vimp` | 0.985 |
| acs_pubcov | nonlinear | `l1_coef` | 0.879 |
| acs_pubcov | nonlinear | `pred_vimp` | 1.000 |
| acs_pubcov | nonlinear | `shift_vimp` | 0.985 |

## 聚合器

每个 site 上传一条概念间隙 VIMP，先截断负值再归一成单纯形。真值向量把每个 CD site 的种植坐标加一。`delta_cf_weighted_shift` 用 Δ_cf 当 site 权重，clean 和 noise 接近 0 时不会进入聚合。IVW 用置换重复的方差，方差下界 1e-4。

| 数据集 | 聚合器 | AUROC | P@K | Spearman |
| --- | --- | --- | --- | --- |
| acs_income | `delta_cf_weighted_shift` | 0.917 | 0.750 | 0.697 |
| acs_income | `ivw_shift` | 0.869 | 0.667 | 0.597 |
| acs_income | `median_shift` | 0.869 | 0.667 | 0.608 |
| acs_income | `primary_sites_only` | 0.893 | 0.667 | 0.664 |
| acs_income | `uniform_delta_predictive` | 0.869 | 0.833 | 0.600 |
| acs_income | `uniform_l1` | 0.988 | 0.917 | 0.798 |
| acs_income | `uniform_predictive` | 0.988 | 0.917 | 0.798 |
| acs_income | `uniform_shift` | 0.869 | 0.667 | 0.597 |
| acs_income | `uniform_shift_without_noise` | 0.869 | 0.667 | 0.589 |
| acs_pubcov | `delta_cf_weighted_shift` | 1.000 | 1.000 | 0.634 |
| acs_pubcov | `ivw_shift` | 0.995 | 0.917 | 0.629 |
| acs_pubcov | `median_shift` | 0.984 | 0.833 | 0.615 |
| acs_pubcov | `primary_sites_only` | 0.995 | 0.917 | 0.630 |
| acs_pubcov | `uniform_delta_predictive` | 0.875 | 0.833 | 0.494 |
| acs_pubcov | `uniform_l1` | 0.927 | 0.667 | 0.552 |
| acs_pubcov | `uniform_predictive` | 0.990 | 0.917 | 0.623 |
| acs_pubcov | `uniform_shift` | 0.995 | 0.917 | 0.629 |
| acs_pubcov | `uniform_shift_without_noise` | 1.000 | 1.000 | 0.635 |

LOSO influence 是 `1 - Spearman(全体均匀聚合, 去掉该 site 后的聚合)`。CD site 应当比 clean site 更拉动聚合。

| 数据集 | site 类型 | 平均 influence |
| --- | --- | --- |
| acs_income | clean | 0.033 |
| acs_income | linear | 0.013 |
| acs_income | noise | 0.018 |
| acs_income | nonlinear | 0.021 |
| acs_pubcov | clean | 0.008 |
| acs_pubcov | linear | 0.006 |
| acs_pubcov | noise | 0.006 |
| acs_pubcov | nonlinear | 0.005 |

## 强度、子组比例、渐进激活

这一格只在 ACS Income 上做。`α` 是激活后真正改写标签的概率。`subgroup_q` 是线性规则使用的 `X1` 分位：只有高于该分位的行会被改写。非线性 site 在主设定里仍然改写全部激活行。渐进激活用 4 个 batch 把权重从 0 拉到 1。

WKHP 在 40 小时有一个尖峰，0.5 分位和 0.7 分位都落在 40，所以 `X1 > q` 在这两个分位上是同一批人（大约 23%）。0.7 那一列和 0.5 完全重复，表里不再单列。0.8 和 0.9 分位分别落到 45 和 50 小时，尾部才会变薄。

| α | 激活 | 线性子组分位 | 平均一致率 | 最差 seed |
| --- | --- | --- | --- | --- |
| 0.4 | abrupt | 0.5 | 0.783 | 0.650 |
| 0.4 | abrupt | 0.8 | 0.617 | 0.450 |
| 0.4 | abrupt | 0.9 | 0.550 | 0.450 |
| 0.4 | gradual | 0.5 | 0.750 | 0.650 |
| 0.4 | gradual | 0.8 | 0.600 | 0.450 |
| 0.4 | gradual | 0.9 | 0.550 | 0.450 |
| 0.7 | abrupt | 0.5 | 1.000 | 1.000 |
| 0.7 | abrupt | 0.8 | 0.967 | 0.950 |
| 0.7 | abrupt | 0.9 | 0.817 | 0.750 |
| 0.7 | gradual | 0.5 | 1.000 | 1.000 |
| 0.7 | gradual | 0.8 | 0.950 | 0.900 |
| 0.7 | gradual | 0.9 | 0.683 | 0.500 |
| 1.0 | abrupt | 0.5 | 1.000 | 1.000 |
| 1.0 | abrupt | 0.8 | 1.000 | 1.000 |
| 1.0 | abrupt | 0.9 | 0.850 | 0.750 |
| 1.0 | gradual | 0.5 | 1.000 | 1.000 |
| 1.0 | gradual | 0.8 | 1.000 | 1.000 |
| 1.0 | gradual | 0.9 | 0.867 | 0.800 |

## 怎么读这些数

在 α=1 的主设定上，Δ_cf 把 5 个结构化 CD site 全部排在 4 个 clean site 和 noise site 前面，两个数据集、4 个 seed 的配对一致率都是 1。Pareto 秩同样是 1。Top-5 精度是 1：Δ_cf 最高的 5 个 site 就是 5 个 CD site。

这个分离不是协变量臂做出来的。RF domain AUC 和 MMD 的一致率落在 0.5 附近或更低，逻辑 domain AUC 的 site 均值也在 0.5 左右。边际标签总变差的一致率同样接近 0.5，因为结构化规则匹配了基线正例率。联合 C2ST 能看出 CD 和 clean 的差别，但会把 noise site 排得更靠前：它只检查联合分布变了，不检查新标签还能不能被 X 解释。

大参考模型的 Brier 差（`excess_pooled`）在 ACS Income 上对 noise 的一致率只有 0.40。Noise 的标签和尖锐的基线模型不一致，超额风险看起来像很强的漂移。等样本量再乘技能之后，这个假驱动消失。α 降到 0.4 时，区域最大值的一致率掉到 0.8 以下，只看全样本的版本掉到 0.45 左右。弱的部分改写是这个分数的工作边界。

时间曲线在 t*=8 之后上升，窗口长度为 4，所以分数要到窗口完全落进漂移之后（大约 batch 11）才饱和。在线命中允许 ±2 个 batch。非线性和 Public Coverage 的线性 site 命中率是 1，Income 线性大约 0.83。Income 的 clean site 有大约 0.38 的在线误报：阈值是校准段最大值加 0.01，小窗口上的噪声会越线，但越线的位置远离 t*。导数法的误差大约是一个窗口长度，它标的是饱和点而不是起点。

类型分辨用的是预先指定的机制字典，不是从 site 标签里反推的。全样本上，乘积和正弦平方都能被对应的概念组认出来，线性 site 只有大约一半，因为子组外的标签仍然是基线结果，加法模型会和基线结构混在一起。把判定限制在 X1 上尾之后，线性召回升到 0.9 以上，正弦平方仍然对，乘积规则反而不再独占最优，因为乘积漂移写在全部行上，上尾里的截断让线性拟合看起来也不差。

特征支撑上，预测型置换重要度和概念间隙置换重要度都能把种植坐标排到前面。Income 上这些坐标本来就是基线结果的预测变量，所以 clean site 的预测重要度也会指向它们；Δ_cf 加权的概念间隙聚合把 AUROC 从均匀聚合的 0.87 提高到 0.92。Leave-one-site-out 的秩影响在所有 site 类型上都很小，而且 clean 不比 CD 小。site 级 Δ_cf 能指出 primary driver，均匀 VIMP 聚合的影响度不能。
