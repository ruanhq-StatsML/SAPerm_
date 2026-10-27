# tabular-drift-attribution

Mini-batch 级表格漂移归因。对每一个特征块，用**其余特征**当条件，从参考批次生成这一块的反事实，再按块和按特征排序。

单次 \(\rho\) 只改条件向量里的一格，条件太瘦。这里每一块的条件是该行剩下的全部列。

## 快速开始

```bash
pip install -r requirements.txt
python run.py --config configs/synthetic.yaml
```

合成面板不需要外部数据。从第 3 个 batch 起，`block_1` 被平移 1.5。`stub` 生成器用参考批次里、在其余列上最近的行，捐出目标块。

## 输入

`data.loaders.load_*()` 返回：

- `X_batched`: 每个 batch 一个 `(N_t, F)` 数组
- `blocks`: `{block_name: [column index, ...]}`
- `gt`: `{t: set(block_name)}`，没有真值时为空集
- `col_names`: 列名
- `F`: 列数

生成器只在第 0 个 batch 上拟合。后面的 batch 都和这个参考比。

## 输出

`attribution.json` 的每一行：

```json
{
  "t": 3,
  "metric": "mean_diff",
  "block_ranked": {"block_1": 1.5, "block_0": 0.2},
  "block_prop": {"block_1": 0.7, "block_0": 0.1},
  "feature_ranked": {"f25": 1.5, "f26": 1.4},
  "top": "block_1",
  "gt_blocks": ["block_1"],
  "top_hit": true,
  "n_valid": 771,
  "valid_ratio": 0.964
}
```

`contribution` 是这一批的块贡献，等于该块距离除以全部块距离，和 `block_prop` 是同一个数。`feature_contribution` 是绝对差最大的十个特征各自占总绝对差的比例。`top_hit` 在没有真值时是 `null`。

## 贡献怎么定义

参考批次是第 0 批。对后面的每一批、每一个块 \(b\)，条件是这一行里 \(b\) 以外的列。反事实块从参考条件分布里抽：

\[
X^{cf}_{t,b} \sim P_{\mathrm{ref}}(X_b \mid X_{-b}=X_{t,-b}).
\]

块贡献是观测块和这个反事实块的距离份额：

\[
c_t(b) = \frac{d(X_{t,b}, X^{cf}_{t,b})}{\sum_{b'} d(X_{t,b'}, X^{cf}_{t,b'})}.
\]

合成面板在漂移前四块各约 25%。`block_1` 从第 3 批被平移 1.5 之后，它的贡献升到 0.337，并且在五个漂移批次上都是第一名。

## 可切换的生成器

| 名字 | 条件 | 说明 |
| --- | --- | --- |
| `stub` | 其余列上的最近参考行 | 测试和默认路径，不训练模型 |
| `ctgan` | 其余列的批次中位箱 | 离散条件，连续列先分箱 |
| `tabforge` | 不使用条件 | 无条件扩散，再按参考边缘重排 |
| `diffltm` | 等式和区间 | 需要事先训好 `experiment=conditional` 的检查点 |

`generator_args` 原样传给对应引擎。DiffLTM 需要 `repo_path` 和 `run_dir`。

## 距离

`metric` 取 `mean_diff`、`wasserstein`、`mmd`、`h_div`。排名用的是观测块和反事实块的距离，不是一个标量 \(\rho\)。

## 质量门

反事实行要落在参考批次的分位区间里，并且标准化后的 k 近邻距离不能远过参考云。`density_tau: 0.5` 表示距离不超过参考中位近邻距离的 2 倍。高维下不要求每一列都落在区间内，一行里 95% 的列在区间内即可。

## 扩展

新数据集：在 `data/loaders.py` 添加 `load_xxx()`，返回上面的五个字段，并登记到 `run.py` 的 `LOADERS`。

新生成器：继承 `BaseEngine`，实现 `fit` / `generate`。`generate` 的 `X_cond` 是目标块以外的列。

MovieLens、IEEE-CIS、Home Credit 的配置在 `configs/`。原始文件不在仓库里。MovieLens-1M 放到 `data/movielens/ml-1m/`。IEEE-CIS 的 `train_transaction.csv` 放到 `data/ieee_cis/`。Home Credit 的 `application_train.csv` 放到 `data/home_credit/`。这三份数据没有块级真值。全时段的块贡献和种植平移的命中率在 `results/panel_effect.md`，用 `python notes/panel_effect.py` 重算。
