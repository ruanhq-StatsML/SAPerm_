# tabular-drift-attribution

Mini-batch 级表格漂移归因。对每一个特征块做一次迭代更新：条件是**今天其余特征**和**昨天的全量特征**，从参考批次生成这一块，写回后再更新下一块。

## 快速开始

```bash
pip install -r requirements.txt
python run.py --config configs/synthetic.yaml
python characterize.py --dataset synthetic --model stub
```

`DATASETS[name].forward(model)` 对这个数据集逐个 batch 做一次 forward。`model` 从 `GENERATIVE_MODELS` 里取，取到之后再做 data digestion，把离散列交给这个模型。

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

参考批次是第 0 批。从第 1 批起，工作行 \(Z_t\) 从当天观测 \(X_t\) 开始，昨天 \(X_{t-1}\) 保持全量观测。按块的顺序，每一块 \(b\) 用当前工作行里的其余列和昨天的全量特征重抽，并写回 \(Z_t\)：

\[
Z_{t,b} \leftarrow P_{\mathrm{ref}}(X_b \mid X_{-b}=Z_{t,-b},\; X_{t-1}).
\]

同一套更新交给 stub、CTGAN、TabFORGE，以及 DiffLTM 的 SMC、Harpoon、rejection。`n_sweeps` 大于 1 时，下一次扫描看到的是上一轮写回的其余列。

块贡献是观测块和这个反事实块的距离份额：

\[
c_t(b) = \frac{d(X_{t,b}, Z_{t,b})}{\sum_{b'} d(X_{t,b'}, Z_{t,b'})}.
\]

合成面板在漂移前最大一块是 0.282。`block_1` 从第 3 批被平移 1.5 之后，贡献在 0.325 到 0.329，五个漂移批次都是第一名。份额不动是因为列独立：+1.5 落在 free 的那一块，175 个条件列解释不了它。

## 条件了几列，哪些要离散

每一块 25 列是 free。条件是当天其余 75 列，加上昨天的 100 列，合计 175。昨天的这一块也在条件里，当天的这一块不在。

离散只看参考批：取值像整数，并且不同取值不超过 32。合成面板的 100 列都是正态，一列都不离散，175 个条件全是 ranges。`python characterize.py` 把这个计数和四个生成器的吃法写到 `results/synthetic_digestion.md`。

stub 和 TabFORGE 不离散这 175 列。CTGAN 因为只能条件在类别上，把这 175 个连续列分箱；那是模型的限制。DiffLTM 不把它们收成等式。

## 可切换的生成器

| 名字 | 条件 | 说明 |
| --- | --- | --- |
| `stub` | 今天其余列 + 昨天全量 | 参考行上的最近邻，不训练模型 |
| `ctgan` | 今天其余列 + 昨天全量的批次中位箱 | 离散条件，连续列先分箱 |
| `tabforge` | 同上，用来重排名次 | 无条件样本按参考捐赠行的名次重排 |
| `diffltm` | conditional sampling schema | 每行一份 `bounds` / `equality` / `ranges` / `weights`；目标块是 free |

`generator_args` 原样传给对应引擎。DiffLTM 需要 `repo_path` 和 `run_dir`。检查点的列是当天特征加上 `yest__` 滞后列，数值列在前，类别列在后。类别条件用 DiffLTM 的字符串编码，数值条件是以观测值为中心的区间。这份 schema 不带线性约束。SMC、Harpoon、rejection 各生成一条；`cfg` 用 `channels()` 编码同一份 schema，需要 `method: cfg`。

## 距离

`metric` 取 `mean_diff`、`wasserstein`、`mmd`、`nammd`、`h_div`。排名用的是观测块和反事实块的距离，不是一个标量 \(\rho\)。

## 质量门

反事实行要落在参考批次的分位区间里，并且标准化后的 k 近邻距离不能远过参考云。`density_tau: 0.5` 表示距离不超过参考中位近邻距离的 2 倍。高维下不要求每一列都落在区间内，一行里 95% 的列在区间内即可。

## 扩展

新数据集：在 `data/loaders.py` 添加 `load_xxx()`，返回上面的五个字段，并登记到 `run.py` 的 `LOADERS`。

新生成器：继承 `BaseEngine`，实现 `fit` / `generate`。`generate` 的 `X_cond` 是今天目标块以外的列，`X_yesterday` 是昨天的全量特征。

MovieLens、IEEE-CIS、Home Credit 的配置在 `configs/`。原始文件不在仓库里。MovieLens-1M 放到 `data/movielens/ml-1m/`。IEEE-CIS 的 `train_transaction.csv` 放到 `data/ieee_cis/`。Home Credit 的 `application_train.csv` 放到 `data/home_credit/`。这三份数据没有块级真值。全时段的块贡献和种植平移的命中率在 `results/panel_effect.md`，用 `python notes/panel_effect.py` 重算。从峰值日向前七步的 TabDDPM pathway 在 `results/pathway_attribution.md`，表在 `latex/pathway_attribution.tex`。合成面板、IEEE-CIS、MovieLens-1M 和 Home Credit 都在这一份里。
