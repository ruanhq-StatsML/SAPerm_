# Uplift 评估对照脚本

评估写在 `evaluate.py`。能对上脚本的两列，和不能冒充脚本的一列，分开记。

## 脚本里怎么评

`r_learner_observed_risk` 走到 `vimp_loco_r_risk` 的 `observed_r_risk` 为止。

1. `make_folds` 把样本切成 5 折。
2. `cross_nuisance_fit`：结果模型是 `rf_regressor`，倾向得分是 `logistic_classifier`。森林的 `max_features = round(p/3)`，`min_samples_leaf = round(sqrt(n)/2)`，和 `make_rf_regression` 一样。树的棵数是 40，注册表默认是 150。
3. `Y_tilde = Y - mu`，`W_tilde = W - e`，`e` 截断在 0.01。
4. τ 用 `_fit_tau_rlearner_weighted` 当场拟合：标准化之后的 ridge，权重是 `W_tilde^2`。
5. 风险是 `mean((Y_tilde - tau_hat * W_tilde)^2)`。

这里的 τ 是这一步拟合出来的，不是外面传进来的 uplift。LOCO 那一圈（去掉一列再重拟合）没有放进每个 batch。

`permucate_test_po_risk` 走到 `permuCATE_vimp` 的 `po_risk_original` 为止。

1. `train_test_split(..., test_size=0.5, stratify=W)`。
2. 训练集上，处理组和对照组各拟合一个随机森林，得到 `mu1`、`mu0`。倾向得分是逻辑回归。
3. 伪结果是脚本里的那条：`(mu1 - mu0) + (Y - mu_T) (W - pi) / (pi (1-pi))`。
4. τ 是这个伪结果上的随机森林。
5. `po_risk` 在留出的一半上计算。

置换那一圈（`n_perm` 次打乱残差）没有放进每个 batch。

`deployed_po_risk` 调用的是同一个 `po_risk`，但 `mu0`、`mu1`、τ 来自已经部署的双逻辑回归，`pi` 用随机片上已知的 0.5。这不是 `permuCATE_vimp`。那个函数会重拟合。

判断只用两样东西：MMD，和这份部署的 PO-risk。late MMD 至少是 early 的 5 倍，判协变量。冻结模型的 PO-risk 上升至少 0.25，并且比同期重拟合高出至少 0.15，判 uplift。脚本自己重拟合出来的 R-risk 和 PO-risk 不进这个判断。

## 四次重复

| 机制 | 阶段 | MMD | 脚本 R-risk | 脚本 PO | 部署 PO 冻结 | 部署 PO 重拟合 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| τ 换号 | 4–7 | 0.0011 | 0.217 | 1.350 | 0.826 | 0.842 | |
| τ 换号 | 8–15 | 0.0008 | 0.215 | 1.807 | 1.478 | 0.946 | uplift |
| 第一列平移 | 4–7 | 0.0013 | 0.217 | 1.528 | 0.821 | 0.840 | |
| 第一列平移 | 8–15 | 0.1055 | 0.188 | 1.274 | 0.718 | 0.754 | covariate |

换号之后，脚本的 R-risk 从 0.217 到 0.215。它每个 batch 都重拟合 τ，所以冻结模型已经错了，这一列也不会升。脚本的 PO-risk 从 1.35 升到 1.81，重拟合之后的损失变大，但它评的是新拟合的森林，不是那份已经部署的模型。部署的 PO-risk 从 0.83 升到 1.48，同期重拟合是 0.95。判断用的是这一对比。

平移之后，MMD 从 0.0013 到 0.106。脚本 R-risk 从 0.217 到 0.188。部署的 PO-risk 从 0.82 降到 0.72，没有高于重拟合。判断是协变量。后段流量 0.87 仍在冻结模型上，因为两条策略的价值都在 0.81。流量不改判断。

## 没有照抄的地方

- 森林是 40 棵。`vimp_loco_r_risk` 里注册表的 `ntree` 是 150。
- 每个 batch 停在 `observed_r_risk` 和 `po_risk_original`。LOCO 那一圈和 `n_perm` 置换那一圈没有跑。
- `vimp_loco_r_risk` 调用 `cross_nuisance_fit` 时没有传入 `seed`，脚本里的折因此固定是 2026。这里用的是这个 batch 的 seed。损失里的倾向得分截断是 0.01，和脚本外层那一次截断一样。
- `permuCATE_vimp` 的 `train_test_split` 没有 `random_state`。这里传入了 `random_state=seed`。两个臂的森林用同一个 `seed`，和 `permuCATE_fit_nuisance` 一样。
- 倾向得分那一折如果只剩一个处理组，这里用训练集均值。脚本仍去拟合逻辑回归。
