# DDPM 反事实归因评估报告

查询：如果 10 天前客群 A 的 IP 块处在参考路径上，今天客群 B 的购买分布会怎样。
估计量是条件分布反事实的随机傅里叶 MMD²，不是结构因果效应。

## 语义声明

- This is a conditional-distribution counterfactual, not a structural causal effect. Replacing the lagged IP-block feature answers p(Y | history with the anomaly set to a reference draw, covariates as observed). It does not, by itself, identify a path-specific effect.
- Transmission channels are not separable. A single site-specific distributed lag stands in for risk-control, recommendation, and social paths; those paths are not individually identified.
- MMD is unitless. The reported attribution is the random-Fourier MMD^2 and its ratio to the no-intervention baseline MMD^2.
- Steps whose counterfactual conditioning vector falls outside the reference support do not enter the headline attribution.

## 数据生成过程

10 个站点，48 个时间窗口，每个窗口 40 天，异常日 t0=18，spike=1.0。条件均值对 14 天分布式滞后线性，条件方差随信号轻度放大。站点嵌入是模型能看到的唯一站点信息；隔离标记不是特征。训练窗口 36 个，验证窗口 12 个（时间上的后 20%）。

## 主驱动

主驱动不是异常最大的站点，而是在查询 lag 上传导核最大的站点。十个站点在第 18 天收到同一类 IP 冲击，剂量按验证窗口变化；站点 7 的潜在核与站点 0 相同，但隔离把有效核置零；站点 6 的重叠通道开在 lag 2，回答不了 lag 10 的问题；站点 2 和 3 的核分别集中在第 3 天和第 1 天。Oracle 在 lag 10 的排序是 [0, 1, 4, 5, 2, 3, 6, 7, 8, 9]，估计的归一化 Δ 排序是 [0, 1, 4, 7, 8, 6, 3, 5, 2, 9]。估计的主驱动站点是 0（weekly_primary），oracle 主驱动是 0（weekly_primary），两者一致。站点 0 的路径峰值在第 10 天，对应周尺度桶（lags 6–14）。 原始 Δ 的前三名是站点 0、4、1，和 oracle 的前三名一致，4 和 1 的次序对调了（0.165 对 0.143，oracle 是 0.24 对 0.43）。归一化分数会把基线接近 0 的空站点抬上去，主驱动判断以原始 Δ 和路径峰值一起看。

| 站点 | 名称 | 通道 | oracle 均值偏移 (lag 10) | DDPM Δ | Δ/Δ_base | 路径峰值 (天) | 路径形状 | 支持集通过率 |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 0 | weekly_primary | risk | 0.807 | 0.412 | 30.09 | 10 | peak_then_decline | 1.00 |
| 1 | weekly_secondary | risk | 0.425 | 0.143 | 9.18 | 10 | peak_then_decline | 1.00 |
| 2 | daily_driver | social | 0.000 | 0.020 | 1.42 | 3 | peak_then_decline | 1.00 |
| 3 | immediate_driver | risk | 0.000 | 0.034 | 2.31 | 1 | irregular | 1.00 |
| 4 | weekly_weak | social | 0.238 | 0.165 | 3.92 | 11 | peak_then_decline | 1.00 |
| 5 | decay_recent | social | 0.008 | 0.046 | 1.74 | 1 | irregular | 1.00 |
| 6 | social_overlap | social | 0.000 | 0.028 | 2.88 | 2 | peak_then_decline | 1.00 |
| 7 | isolated_blocked | risk | 0.000 | 0.028 | 3.52 | 15 | monotone_rise | 0.92 |
| 8 | null | risk | 0.000 | 0.026 | 2.93 | 4 | peak_then_decline | 1.00 |
| 9 | null_noisy | risk | 0.000 | 0.024 | 1.29 | 7 | peak_then_decline | 1.00 |

特征层主驱动是 **b0**。 b0 干预的 Δ = 0.412，干扰块 mask 的 Δ = 0.043，协变量 mask 的 Δ = 0.112。

## 核心结论

- 单次查询（验证集中 dose = 1 的一个窗口，站点 0，horizon 10）：Δ^cf = 0.631（95% bootstrap CI [0.426, 0.852]）。
- 归一化分数 Δ^cf / Δ^base = 30.29（基线 Δ^base = 0.021）。
- 模型符号：反事实均值比观测均值低 1.009（异常抬高了购买指数）。
- 路径形状（站点 0，对验证窗口取平均）：peak_then_decline，峰值在第 10 天。
- 门控（站点 0，hourly / daily / weekly）= 0.17, 0.15, 0.68。
- 反事实条件向量的支持集通过率 = 1.00。
- 阶段 1 MVP 排序一致率 = 0.85，决策：proceed。
- 归因是否被校准关卡接受：否。L3 的 Wasserstein 没有过冻结的 0.10 阈值；路径数字仍与 oracle 对照，结论见综合判定。
- 观测校准第一次未通过，已按预先写好的规则加长训练并加宽网络后重试。

## L1 相对排序

- 剂量-反应：配对一致率 0.80，Spearman 0.67。未通过。in-support episodes only。
- 时间距离（衰减核站点 5）：Spearman -0.76，逐步不增比例 0.77。通过。
- 周延迟站点上的同一 Spearman = 0.74。这个构造假设在延迟核上不成立，不把它算作方法失败。
- 重叠度（社交站点 6，lag 2）：配对一致率 0.87，Spearman 0.82。通过。风控站点 0 的重叠度 Spearman = -0.13。
- 外生隔离：配对 Cohen's d = -1.48，p = 0.000。通过。开放站点平均 Δ = 0.412，隔离站点平均 Δ = 0.028。
- 综合：3 / 4 个构造通过。层结果：通过。

## L2 时间定位

- 真实 lag 的 Δ = 0.412，无干预基线 Δ = 0.078。
- Placebo lag 3：Δ = 0.058。通过。
- Placebo lag 5：Δ = 0.071。通过。
- Placebo lag 15：Δ = 0.049。通过。
- Placebo lag 20：Δ = 0.072。通过。
- 精度扫描峰值在第 10 天。通过。
- 门控与站点尺度对齐：通过。长滞后站点的周级权重大于小时级：通过。
  - 站点 0：主导尺度 w，期望 w。通过。
  - 站点 2：主导尺度 d，期望 d。通过。
  - 站点 3：主导尺度 h，期望 h。通过。
- 综合：通过。

## L3 观测校准

- PIT 的 KS p = 0.445。通过。
- 观测 Wasserstein / std(Y) = 0.245。未通过。
- 特征条件 oracle 的地板是 0.180，同分布抽样地板是 0.080。未观测项 U 让任何只看特征的模型都过不了 0.10 这条线；DDPM 高出 oracle 地板 0.065。
- HSIC 比值 = 1.009。通过。
- Observational calibration is necessary and not sufficient for counterfactual accuracy.
- 综合：未通过。

## L4 代理一致性

- Uplift（R-learner CATE，处理变量是 lag-10 的 IP 块）= 1.005，rollout 的均值偏移 = 0.794。通过。
- Mask 差值 Δ_cf − Δ_mask = 0.369（Δ_mask = 0.043）。通过。
- 业务超额的峰值在第 10 天，归因峰值在第 10 天。通过。
- 门控与长期传导一致：通过。
- 综合：4 个代理一致，状态 pass。

## L5 敏感性

- 基线选择 CV = 0.13。通过。分数：{"reference_residual": 0.4118, "reference_mean": 0.4131, "reference_sample": 0.5326, "zero": 0.4155}。
- 核宽 CV（归一化分数）= 0.70。未通过。分数：[7.7035, 13.8823, 31.6283]。
- 架构 CV = 0.16。通过。分数：{"ddpm": 0.4118, "quantile_mvp": 0.3083, "linear_gaussian": 0.408}。
- E-value = 4.06（Cohen's d = 0.92）。通过。高于警戒线。
- 综合：未通过。

## 支持集与 FSDS

反事实参考路径把异常日放回潜在参考值，条件向量落在训练边际里（坐标标准化绝对值的阈值是 4.0）。为了检查定位器，另做了一次人为外推：把 lag-10 的 b2 推到均值以外 4.5 个标准差。FSDS 最高的坐标是 b2_lag10（前三名 b2_lag10=1.42, b0_lag1=0.00, b0_lag3=0.00）。最近邻距离从 4.05 降到钳制后的 2.63。严重程度在这个例子里对应钳制；真实归因路径没有触发回退。

## 综合判定

- 可信度：**中等**
- 瓶颈层：L3 的 Wasserstein 阈值低于未观测混杂的地板；L5 的归一化比值随核宽变化
- 建议：路径峰值、站点排序和 b0 这一层可以对上 oracle，业务上可以用来指出「哪一个站点、哪一个 lag、哪一个块」。Δ/Δ_base 的倍数随核宽从 7.7 变到 31.6，不要把这个倍数写成一个稳定的效应量。Wasserstein 0.245 比特征条件 oracle 的地板 0.180 高 0.065，PIT 和 HSIC 已通过；0.10 的阈值落在地板下面，是这条 DGP 上的协议限制，不是峰值找错。E-value 4.06，未观测混杂仍要用随机实验确认。

## 训练

家族 linear。扩散步数 60，隐藏宽度 192，最优验证噪声损失 0.3966。余弦日程，历史条件 dropout 10%，RFF 维度 100。训练回合损失从 0.5325 到 0.4071。做了一次预先规定的重训。 网络 softmax（站点 0）= [0.06589880585670471, 0.04468104988336563, 0.8894201517105103]；验收用的是消融权重，不是 softmax。

