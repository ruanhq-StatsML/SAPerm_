"""Accuracy paths on rolled digits, with and without pseudo-outcome weights.

Each registered query rule is run twice from the same source-trained network:
once with a flat weight, once with the per-row pseudo-outcome residual.
``results/accuracy_paths.png`` is the test accuracy after every round.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deep_active_sampling.data import make_digits_stream
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


def run_once(seed: int, schemes: dict) -> dict:
    stream = make_digits_stream(seed=seed)
    print(f"seed {seed}: digits, {len(schemes)} schemes", flush=True)
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


def save_accuracy_plot(avg: dict, path: Path) -> None:
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
    path_ax.axhline(0.1, color="0.45", linewidth=0.8, linestyle="--", label="Chance")
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
    fig.suptitle("Rolled digits: accuracy path with pseudo-outcome weights", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_board(avg: dict, n_seeds: int) -> str:
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
        "# 手写数字上的准确率路径",
        "",
        "数据换成 sklearn 的 8×8 手写数字，像素留在 0 到 1。按源图像逐像素做标准化会把滚进空白边缘的墨水放大到几百，分类器学不动，所以这里不再除以那个标准差。源分布是原图像。目标分布把每张图沿水平方向滚两个像素，标签不变。参考窗前一半是源，后一半是目标。池子里一半原图、一半滚过的图。测试集全是滚过的图。",
        "",
        "查询规则接的是 deep active learning 里能在同一个分类网络上算的那些：不确定性、dropout 版、BALD、BatchBALD、core-set、k-center、k-means、cluster-margin、BADGE、baseline、BIM、DeepFool、ALBL。每条规则跑两遍，一遍权重是平的，一遍权重是参考窗上拟合的伪结果残差。VAAL、WAAL、learning-loss、GCN 和半监督那一组要换训练器，没有接进这个网络。",
        "",
        f"种子 {', '.join(str(s) for s in SEEDS)}。左图是各规则的平均路径，色带是规则之间的标准差，虚线是 10 类的随机猜测。k-center 和 core-set 是同一条最远点规则，平均和右图只计 core-set 一次。右图是终点准确率的差，伪结果权重减去无权重。",
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


def main():
    schemes = schemes_with_po_risk()
    runs = [run_once(seed, schemes) for seed in SEEDS]
    avg = _average(runs)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_accuracy_plot(avg, RESULT_DIR / "accuracy_paths.png")
    board = render_board(avg, n_seeds=len(runs))
    (RESULT_DIR / "BOARD.md").write_text(board)
    (RESULT_DIR / "curves.json").write_text(json.dumps(_jsonable({"runs": runs, "average": avg}), indent=2))
    print(f"wrote {RESULT_DIR / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
