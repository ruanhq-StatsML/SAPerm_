"""Accuracy paths on rolled digits, with and without pseudo-outcome weights.

Each registered query rule is run twice from the same source-trained network:
once with a flat weight, once with the per-row pseudo-outcome residual.
``results/accuracy_paths.png`` is the test accuracy after every round.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deep_active_sampling.data import (
    make_digits_stream,
    make_pendigits_stream,
    make_satimage_stream,
)
from experiments.deep_active_sampling.query_strategies import schemes_with_po_risk
from experiments.deep_active_sampling.sampling import deep_active_sampling

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
SEEDS = (0, 1)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def run_once(stream: dict, seed: int, schemes: dict) -> dict:
    print(f"seed {seed}: {stream['name']}, {len(schemes)} schemes", flush=True)
    out = deep_active_sampling(
        stream["X"],
        stream["Y"],
        batch_size=32,
        query_strategy=schemes,
        ref_batch_size=stream["ref_batch_size"],
        X_test=stream["X_test"],
        Y_test=stream["Y_test"],
        max_rounds=5,
        epochs_per_round=3,
        warm_epochs=4,
        hidden=64,
        lr=0.08,
        n_perm=4,
        rf_estimators=30,
        seed=seed,
        optimizer="SGD",
    )
    shifted = stream["pool_shifted"]
    schemes_out = {}
    for name, result in out["schemes"].items():
        selected = result["selected_idx"]
        summary = dict(result["summary"])
        summary["frac_shifted"] = float(shifted[selected].mean()) if selected.size else float("nan")
        summary["loss_curve"] = result["loss_curve"]
        summary["acc_curve"] = result["acc_curve"]
        schemes_out[name] = summary
        print(
            f"  {name:28s} acc {summary['acc_final']:.3f}  "
            f"lift {summary['acc_lift']:+.3f}  shifted {summary['frac_shifted']:.3f}",
            flush=True,
        )
    return {"seed": seed, "schemes": schemes_out}


def _average(runs: list[dict]) -> dict:
    names = list(runs[0]["schemes"])
    schemes = {}
    for name in names:
        rows = [run["schemes"][name] for run in runs]
        schemes[name] = {
            "strategy": rows[0]["strategy"],
            "weight": rows[0]["weight_mode"],
            "acc_start": _mean([row["acc_start"] for row in rows]),
            "acc_final": _mean([row["acc_final"] for row in rows]),
            "acc_lift": _mean([row["acc_lift"] for row in rows]),
            "acc_auc": _mean([row["acc_auc"] for row in rows]),
            "loss_final": _mean([row["loss_final"] for row in rows]),
            "frac_shifted": _mean([row["frac_shifted"] for row in rows]),
            "acc_curve": np.mean([np.asarray(row["acc_curve"], dtype=float) for row in rows], axis=0),
        }
    return {"schemes": schemes}


# k-center is the same furthest-first rule as core-set. Counting both would
# pull the mean toward that one curve.
_MEAN_SKIP = {"kcenter"}


def _group(schemes: dict, weight: str, field: str) -> float:
    vals = [
        row[field]
        for row in schemes.values()
        if row["weight"] == weight and row["strategy"] not in _MEAN_SKIP
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _curves_for(schemes: dict, weight: str) -> tuple[list[str], np.ndarray]:
    rows = [
        (row["strategy"], np.asarray(row["acc_curve"], dtype=float))
        for row in schemes.values()
        if row["weight"] == weight and row["strategy"] not in _MEAN_SKIP
    ]
    rows.sort(key=lambda item: item[0])
    names = [name for name, _curve in rows]
    stack = np.stack([curve for _name, curve in rows], axis=0)
    return names, stack


def save_accuracy_plot(avg: dict, path: Path, title: str, chance: float = 0.1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = avg["schemes"]
    names, uniform = _curves_for(schemes, "uniform")
    _po_names, weighted = _curves_for(schemes, "po_risk")
    rounds = np.arange(uniform.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    path_ax = axes[0]
    for stack, color, label in (
        (uniform, "#4c78a8", "Unweighted"),
        (weighted, "#e45756", "PO-risk weight"),
    ):
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        path_ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        path_ax.plot(rounds, mean, color=color, linewidth=2.4, label=label)
    path_ax.axhline(chance, color="0.45", linewidth=0.8, linestyle="--", label="Chance")
    path_ax.set_title("Mean test accuracy across query rules")
    path_ax.set_xlabel("Round (0 = source model)")
    path_ax.set_ylabel("Test accuracy")
    path_ax.set_xticks(rounds)
    hi = float(np.max(np.concatenate([uniform.ravel(), weighted.ravel()])))
    path_ax.set_ylim(0.0, min(1.0, max(0.45, hi + 0.08)))
    path_ax.legend(frameon=False, loc="upper left")

    delta = weighted[:, -1] - uniform[:, -1]
    order = np.argsort(delta)
    bar_ax = axes[1]
    colors = ["#e45756" if value < 0 else "#4c78a8" for value in delta[order]]
    bar_ax.barh(np.arange(len(order)), delta[order], color=colors)
    bar_ax.set_yticks(np.arange(len(order)))
    bar_ax.set_yticklabels([names[i] for i in order], fontsize=8)
    bar_ax.axvline(0.0, color="black", linewidth=0.8)
    bar_ax.set_xlabel("Final accuracy, PO-risk minus unweighted")
    bar_ax.set_title("Where the pseudo-outcome weight moves the end")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_board(avg: dict, n_seeds: int, *, title: str, lead: str, n_classes: int) -> str:
    schemes = avg["schemes"]
    uniform_acc = _group(schemes, "uniform", "acc_final")
    po_acc = _group(schemes, "po_risk", "acc_final")
    uniform_shift = _group(schemes, "uniform", "frac_shifted")
    po_shift = _group(schemes, "po_risk", "frac_shifted")
    uniform_auc = _group(schemes, "uniform", "acc_auc")
    po_auc = _group(schemes, "po_risk", "acc_auc")
    best_name = max(schemes, key=lambda name: schemes[name]["acc_final"])
    best = schemes[best_name]
    if po_acc > uniform_acc + 0.01:
        acc_sentence = f"伪结果权重把终点准确率的平均从 {uniform_acc:.3f} 抬到 {po_acc:.3f}。"
    elif uniform_acc > po_acc + 0.01:
        acc_sentence = f"伪结果权重没有抬高终点准确率。无权重平均 {uniform_acc:.3f}，加权平均 {po_acc:.3f}。"
    else:
        acc_sentence = f"终点准确率几乎一样：无权重 {uniform_acc:.3f}，伪结果权重 {po_acc:.3f}。"
    lines = [
        f"# {title}",
        "",
        lead,
        "",
        "查询规则接的是 deep active learning 里能在同一个分类网络上算的那些：不确定性、dropout 版、BALD、BatchBALD、core-set、k-center、k-means、cluster-margin、BADGE、baseline、BIM、DeepFool、ALBL。每条规则跑两遍，一遍权重是平的，一遍权重是参考窗上拟合的伪结果残差。VAAL、WAAL、learning-loss、GCN 和半监督那一组要换训练器，没有接进这个网络。",
        "",
        f"种子 {', '.join(str(s) for s in SEEDS)}。左图是各规则的平均路径，色带是规则之间的标准差，虚线是 {n_classes} 类的随机猜测。k-center 和 core-set 是同一条最远点规则，平均和右图只计 core-set 一次。右图是终点准确率的差，伪结果权重减去无权重。",
        "",
        "![Test accuracy by round](accuracy_paths.png)",
        "",
        f"路径下的面积：无权重 {uniform_auc:.3f}，伪结果权重 {po_auc:.3f}。抽到的目标行比例：无权重 {uniform_shift:.3f}，伪结果权重 {po_shift:.3f}。池子里这个比例是 0.50。",
        "",
        acc_sentence,
        f"终点最高的是 `{best_name}`，准确率 {best['acc_final']:.3f}，比起点高 {best['acc_lift']:+.3f}。",
        "",
        "| 方案 | 查询 | 权重 | 起点 | 终点 | lift | 面积 | 目标行 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(schemes, key=lambda name: -schemes[name]["acc_final"])
    for name in order:
        row = schemes[name]
        lines.append(
            "| {name} | {strategy} | {weight} | {start:.3f} | {acc:.3f} | {lift:+.3f} | {auc:.3f} | {frac:.3f} |".format(
                name=name,
                strategy=row["strategy"],
                weight=row["weight"],
                start=row["acc_start"],
                acc=row["acc_final"],
                lift=row["acc_lift"],
                auc=row["acc_auc"],
                frac=row["frac_shifted"],
            )
        )
    lines.extend(
        [
            "",
            "k-center 与 core-set 的数字相同，因为两者都是嵌入上的最远点。上面的平均没有把它算第二次。",
            "",
            f"重复次数 {n_seeds}。",
            "",
        ]
    )
    return "\n".join(lines)


DIGITS_LEAD = (
    "数据是 sklearn 的 8×8 手写数字，像素留在 0 到 1。按源图像逐像素做标准化会把滚进空白边缘的墨水放大到几百，分类器学不动，所以这里不再除以那个标准差。"
    "源分布是原图像。目标分布把每张图沿水平方向滚两个像素，标签不变。参考窗前一半是源，后一半是目标。池子里一半原图、一半滚过的图。测试集全是滚过的图。"
)

DATASETS = {
    "digits": {
        "factory": make_digits_stream,
        "title": "手写数字上的准确率路径",
        "figure_title": "Rolled digits: accuracy path with pseudo-outcome weights",
        "lead": DIGITS_LEAD,
        "n_classes": 10,
        "tex_name": "Digits",
        "tex_shift": "8$\\times$8 digits, pixels in $[0,1]$, target rolled two columns",
    },
    "pendigits": {
        "factory": make_pendigits_stream,
        "title": "手写笔数字上的准确率路径",
        "figure_title": "Translated pen digits: accuracy path with pseudo-outcome weights",
        "lead": (
            "数据是 UCI 的手写笔数字。十六个坐标先除以 100，落在 0 到 1。"
            "目标行的八个横坐标各加 0.25，等于把笔迹沿 x 平移，数字标签不变。"
            "参考窗前一半是原笔迹，后一半是平移后的笔迹。池子里一半对一半。测试集全是平移后的笔迹。"
        ),
        "n_classes": 10,
        "tex_name": "Pen digits",
        "tex_shift": "16 stroke coordinates in $[0,1]$, target $x$-coordinates $+0.25$",
    },
    "satimage": {
        "factory": make_satimage_stream,
        "title": "卫星图像上的准确率路径",
        "figure_title": "Shifted satimage bands: accuracy path with pseudo-outcome weights",
        "lead": (
            "数据是 Statlog 的 Landsat 卫星图像，OpenML 上的表已经按特征标准化过。"
            "目标行的前 9 个波段各加 1.25，地物类别不变。这是传感器偏移，不是改标签。"
            "参考窗前一半是原光谱，后一半是偏移后的光谱。池子里一半对一半。测试集全是偏移后的光谱。"
        ),
        "n_classes": 6,
        "tex_name": "Satimage",
        "tex_shift": "36 standardized bands, first 9 bands of the target $+1.25$",
    },
}


def _result_dir(key: str) -> Path:
    if key == "digits":
        return RESULT_DIR
    return RESULT_DIR / key


def run_dataset(key: str) -> dict:
    spec = DATASETS[key]
    schemes = schemes_with_po_risk()
    runs = []
    n_classes = spec["n_classes"]
    for seed in SEEDS:
        stream = spec["factory"](seed=seed)
        n_classes = int(stream.get("n_classes", n_classes))
        runs.append(run_once(stream, seed, schemes))
    avg = _average(runs)
    out_dir = _result_dir(key)
    out_dir.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / n_classes
    save_accuracy_plot(avg, out_dir / "accuracy_paths.png", spec["figure_title"], chance=chance)
    board = render_board(
        avg,
        n_seeds=len(runs),
        title=spec["title"],
        lead=spec["lead"],
        n_classes=n_classes,
    )
    (out_dir / "BOARD.md").write_text(board)
    payload = {"dataset": key, "n_classes": n_classes, "runs": runs, "average": avg}
    (out_dir / "curves.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"wrote {out_dir / 'BOARD.md'}", flush=True)
    return payload


def _tex_escape(name: str) -> str:
    return name.replace("_", r"\_")


def _load_average(key: str) -> dict:
    payload = json.loads((_result_dir(key) / "curves.json").read_text())
    payload["dataset"] = key
    if "n_classes" not in payload:
        payload["n_classes"] = DATASETS[key]["n_classes"]
    return payload


def save_mean_path_figure(payloads: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(payloads), figsize=(4.15 * len(payloads), 3.35), sharey=True)
    if len(payloads) == 1:
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        _names, uniform = _curves_for(payload["average"]["schemes"], "uniform")
        _po_names, weighted = _curves_for(payload["average"]["schemes"], "po_risk")
        rounds = np.arange(uniform.shape[1])
        chance = 1.0 / int(payload["n_classes"])
        for stack, color, label in (
            (uniform, "#4c78a8", "Unweighted"),
            (weighted, "#e45756", "PO-risk"),
        ):
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            ax.plot(rounds, mean, color=color, linewidth=2.2, label=label)
        ax.axhline(chance, color="0.45", linewidth=0.8, linestyle="--", label="Chance")
        ax.set_title(DATASETS[payload["dataset"]]["tex_name"])
        ax.set_xlabel("Round")
        ax.set_xticks(rounds)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Test accuracy")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_latex(keys: list[str] | None = None) -> Path:
    """Write a short note from the saved curves. Missing datasets are skipped."""
    if keys is None:
        keys = [key for key in ("digits", "pendigits", "satimage") if (_result_dir(key) / "curves.json").exists()]
    payloads = [_load_average(key) for key in keys]
    latex_dir = HERE / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    figure_name = "mean_paths.png"
    save_mean_path_figure(payloads, latex_dir / figure_name)

    strategy_names = _curves_for(payloads[0]["average"]["schemes"], "uniform")[0]
    deltas = {}
    for payload in payloads:
        names, uniform = _curves_for(payload["average"]["schemes"], "uniform")
        _po, weighted = _curves_for(payload["average"]["schemes"], "po_risk")
        if names != strategy_names:
            raise ValueError(f"{payload['dataset']} strategies do not match the first dataset")
        deltas[payload["dataset"]] = weighted[:, -1] - uniform[:, -1]

    summary_rows = []
    for payload in payloads:
        schemes = payload["average"]["schemes"]
        key = payload["dataset"]
        summary_rows.append(
            {
                "key": key,
                "name": DATASETS[key]["tex_name"],
                "shift": DATASETS[key]["tex_shift"],
                "classes": int(payload["n_classes"]),
                "start": _group(schemes, "uniform", "acc_start"),
                "uniform": _group(schemes, "uniform", "acc_final"),
                "po": _group(schemes, "po_risk", "acc_final"),
                "auc_u": _group(schemes, "uniform", "acc_auc"),
                "auc_p": _group(schemes, "po_risk", "acc_auc"),
                "shift_u": _group(schemes, "uniform", "frac_shifted"),
                "shift_p": _group(schemes, "po_risk", "frac_shifted"),
            }
        )

    summary_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Mean over query rules and two seeds. k-center is omitted because it repeats core-set. Share is the fraction of queried pool rows that come from the target. Chance is $1/C$.}",
        r"\label{tab:po-weight-summary}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & $C$ & Unweighted & PO-risk & AUC plain & AUC PO & Share plain & Share PO \\",
        r"\midrule",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"{row['name']} & {row['classes']} & {row['uniform']:.3f} & {row['po']:.3f} & "
            f"{row['auc_u']:.3f} & {row['auc_p']:.3f} & {row['shift_u']:.3f} & {row['shift_p']:.3f} \\\\"
        )
    summary_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    header = " & ".join(row["name"] for row in summary_rows)
    delta_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Final test accuracy with the pseudo-outcome weight minus the unweighted run, in percentage points. Positive means the weight finished higher on that rule.}",
        r"\label{tab:po-weight-delta}",
        r"\begin{tabular}{l" + "r" * len(summary_rows) + "}",
        r"\toprule",
        f"Rule & {header} \\\\",
        r"\midrule",
    ]
    order = np.argsort(np.mean([deltas[row["key"]] for row in summary_rows], axis=0))[::-1]
    for idx in order:
        cells = " & ".join(f"{100.0 * deltas[row['key']][idx]:+.1f}" for row in summary_rows)
        delta_lines.append(f"\\texttt{{{_tex_escape(strategy_names[idx])}}} & {cells} \\\\")
    mean_cells = " & ".join(
        f"{100.0 * float(np.mean(deltas[row['key']])):+.1f}" for row in summary_rows
    )
    delta_lines.extend(
        [
            r"\midrule",
            f"Mean & {mean_cells} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    by_key = {row["key"]: row for row in summary_rows}
    digits = by_key["digits"]
    pen = by_key["pendigits"]
    sat = by_key["satimage"]
    digit_schemes = next(p for p in payloads if p["dataset"] == "digits")["average"]["schemes"]
    bim = digit_schemes["adversarial_bim"]
    bim_po = digit_schemes["adversarial_bim_po"]
    tex = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath}",
            r"\usepackage{float}",
            r"\begin{document}",
            r"\section*{Pseudo-outcome weights for deep active sampling}",
            "",
            r"\subsection*{Reference window and the online loop}",
            "",
            "Each stream is one matrix. The first 280 rows are the source domain and the next 280 are the target, so the reference window has 560 rows. "
            r"Inside that window the domain indicator is $T=0$ on the first half and $T=1$ on the second. "
            "The remaining 400 rows are the pool, built so that half of them are target rows, and the 200-row test set is target only. "
            "Labels are already observed. The loop decides which of those rows are allowed into SGD. "
            "It is not paying for a new label.",
            "",
            "A two-layer ReLU network with a linear head, width 64, is warm-started for four epochs on the source half only. "
            "That state is copied, so every rule starts from the same weights. "
            "Each copy has its own SGD, with learning rate 0.08, momentum 0.9, and weight decay $10^{-4}$. The optimizer is not reset between rounds. "
            "Gradients are clipped at norm 5. "
            "Each of five rounds queries 32 pool rows and then takes three epochs over the cumulative queried set. "
            "Round 0 in the figure is the warm start, before any pool row is used. "
            "Test accuracy is the fraction of target test rows whose predicted class matches the label.",
            "",
            r"\subsection*{Doubly robust row weight}",
            "",
            r"The nuisance is fit on the reference window only. "
            r"$\mu_0$ and $\mu_1$ are ridge regressions of the integer class code, not of a one-hot label. "
            r"The propensity $\pi$ is logistic, clipped to $[0.01,\,0.99]$. "
            r"$\tau$ is a ridge regression of the pseudo-outcome $\phi$ on $X$, not of the class code. "
            "Dimension importances from PermuCATE, a domain forest, and MMD leave-one-covariate-out are computed and stored. "
            "They are not the weight plotted here.",
            "",
            r"For a row $(X,Y,T)$ the doubly robust pseudo-outcome and its squared residual are",
            r"\begin{equation}",
            r"\phi=(Y-\mu_T)\frac{T-\pi}{\pi(1-\pi)}+(\mu_1-\mu_0),",
            r"\qquad w=(\phi-\tau(X))^2,",
            r"\label{eq:po-weight}",
            r"\end{equation}",
            r"where $\mu_T=\mu_1$ if $T=1$ and $\mu_0$ otherwise. "
            r"Every pool row is scored with $T=1$, including the source rows in the mixture. "
            r"A source row whose label sits near $\mu_0$ and away from $\mu_1$ therefore receives a large $w$. "
            "The sampling weight is this per-row residual. A flat comparison uses no weight at all.",
            "",
            r"\subsection*{How the weight changes a query}",
            "",
            r"The raw residual is replaced by its rank percentile in $(0,1]$, with ties sharing the average rank. "
            "That percentile multiplies a higher-is-better acquisition score. "
            "A constant weight is a constant percentile, so the original ranking is unchanged. "
            "Uncertainty rules, including the dropout averages and BALD, take the top of score times percentile. "
            "Random sampling draws without replacement with probability proportional to the percentile. "
            "Core-set and BADGE multiply distance, or the gradient norm, by the same percentile. "
            r"BIM and DeepFool use $-\lVert\eta\rVert^2$, so a smaller flip of the predicted class is queried first, and that score is then multiplied by the percentile. "
            "Cluster-margin shortlists ten times the batch by weighted margin, clusters the pool by average linkage into 20 groups, and round-robins from the smallest cluster. "
            "BatchBALD is the greedy stand-in: high BALD and low overlap of the mean predictive distribution with rows already chosen. "
            "The reference implementation calls \\texttt{batchbald\\_redux}, which is not a dependency here. "
            "Active learning by learning is a bandit over entropy, margin, and core-set. "
            "Its reward is the accuracy of the batch just returned, measured with the network that queried it, divided by the probability of that arm. "
            "The update happens before the SGD step, so the model has not yet fit those labels. "
            "k-center uses the same furthest-first rule as core-set and is left out of every mean below.",
            "",
            "VAAL, WAAL, learning-loss, the GCN pair, and the semi-supervised trainers are not in the comparison. They replace the network.",
            "",
            r"\subsection*{Three covariate shifts}",
            "",
            "The training budget above is shared. Seeds are 0 and 1. "
            "The band in the figure is the standard deviation across rules, not across seeds. "
            "Pixels and pen coordinates are left on $[0,1]$. Dividing digit pixels by the source standard deviation blows up blank borders after a roll, so that standardization is not used.",
            "",
            "Digits are the sklearn $8\\times 8$ images scaled by 16. Target images are rolled two columns to the right and the label is unchanged. "
            "Pen digits are the UCI stroke table scaled by 100. The eight horizontal coordinates of a target row, the even columns, are shifted by $+0.25$. "
            "Satimage is the OpenML Statlog Landsat table, already standardized to mean zero and unit variance. "
            "The first nine bands of a target row are shifted by $+1.25$. The land-cover label is unchanged. "
            "In each case the reference is source then target, the pool is mixed half and half, and the test set is target only.",
            "",
            r"\subsection*{What the paths show}",
            "",
            r"Figure~\ref{fig:po-weight-paths} is the mean test accuracy. "
            r"Table~\ref{tab:po-weight-summary} averages the rules. "
            r"Table~\ref{tab:po-weight-delta} is the same comparison one rule at a time.",
            "",
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{{figure_name}}}",
            r"\caption{Mean test accuracy by round on the target test set. Round 0 is the source-only warm start. The dashed line is chance, $1/C$. The band is the standard deviation across rules.}",
            r"\label{fig:po-weight-paths}",
            r"\end{figure}",
            "",
            *summary_lines,
            "",
            *delta_lines,
            "",
            "Scoring the mixed pool as $T=1$ is expected to prefer source rows, and it does. "
            f"The target share of the queried batch falls from {digits['shift_u']:.3f} to {digits['shift_p']:.3f} on digits, "
            f"from {pen['shift_u']:.3f} to {pen['shift_p']:.3f} on pen digits, and "
            f"from {sat['shift_u']:.3f} to {sat['shift_p']:.3f} on satimage. "
            "The pool rate is 0.50 in every stream.",
            "",
            "Target accuracy moves when those source rows are the wrong training data. "
            f"On digits the unweighted mean ends at {digits['uniform']:.3f} and the weighted mean at {digits['po']:.3f}. "
            f"The area under the path falls from {digits['auc_u']:.3f} to {digits['auc_p']:.3f}. "
            f"On pen digits the same shift of the batch changes the endpoint only from {pen['uniform']:.3f} to {pen['po']:.3f}. "
            f"On satimage the source network already scores {sat['start']:.3f} on the target before any pool row is queried, "
            f"and the two paths finish at {sat['uniform']:.3f} and {sat['po']:.3f}. "
            "The residual does what $T=1$ specifies. It does not raise average target accuracy.",
            "",
            "The mean hides a few rules that finish higher. "
            f"On digits, BIM with the weight ends at {bim_po['acc_final']:.3f} with target share {bim_po['frac_shifted']:.3f}, "
            f"against {bim['acc_final']:.3f} and {bim['frac_shifted']:.3f} unweighted. "
            "Margin is higher with the weight on both digit streams. "
            "Least confidence and entropy, with and without dropout, account for most of the loss on digits. "
            "Those rule-level gains do not change the average.",
            "",
            r"\end{document}",
            "",
        ]
    )
    path = latex_dir / "accuracy_paths.tex"
    path.write_text(tex)
    print(f"wrote {path}", flush=True)
    return path


def main(datasets=None):
    if not datasets:
        datasets = ["digits"]
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        known = ", ".join(DATASETS)
        raise SystemExit(f"unknown dataset {unknown[0]!r}; known datasets: {known}")
    for key in datasets:
        run_dataset(key)
    if any(key != "digits" for key in datasets) or len(datasets) > 1:
        write_latex()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
