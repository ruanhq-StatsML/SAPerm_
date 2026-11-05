"""Compare weighted and unweighted query rules on one shifted stream.

The pool is a dataloader of source rows and target rows. The target flips the
class weights on the shifted coordinates. Every scheme starts from the same
source-trained network. ``results/BOARD.md`` is the reading of the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.deep_active_sampling.data import make_shift_stream
from experiments.deep_active_sampling.query_strategies import ACTIVE_SAMPLING_SCHEMES
from experiments.deep_active_sampling.sampling import deep_active_sampling

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"

# The full registry is ACTIVE_SAMPLING_SCHEMES. This run keeps one unweighted
# rule and its weighted counterpart for each family, plus the pseudo-outcome
# weight and the product of the two weights.
COMPARE_SCHEMES = {
    name: ACTIVE_SAMPLING_SCHEMES[name]
    for name in (
        "random",
        "entropy",
        "margin",
        "least_confidence",
        "coreset",
        "kmeans",
        "badge",
        "top_weight_pooled",
        "random_pooled",
        "entropy_pooled",
        "margin_pooled",
        "least_confidence_pooled",
        "coreset_pooled",
        "kmeans_pooled",
        "badge_pooled",
        "entropy_po",
        "random_po",
        "entropy_both",
        "bald",
        "bald_pooled",
        "cluster_margin",
        "cluster_margin_pooled",
        "entropy_rf",
        "entropy_mmd",
        "entropy_po_vimp",
        "random_rf",
        "random_mmd",
    )
}

SEEDS = (0, 1)
QUERY_BATCH = 32


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
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr))


def run_once(seed: int) -> dict:
    stream = make_shift_stream(
        n_source=200,
        n_shift_ref=200,
        n_pool=320,
        n_test=240,
        p=8,
        n_classes=4,
        n_shift=3,
        shift=1.25,
        pool_shift_rate=0.5,
        seed=seed,
    )
    print(f"seed {seed}: fitting nuisance and sampling {len(COMPARE_SCHEMES)} schemes", flush=True)
    out = deep_active_sampling(
        stream["X"],
        stream["Y"],
        batch_size=QUERY_BATCH,
        query_strategy=COMPARE_SCHEMES,
        ref_batch_size=stream["ref_batch_size"],
        X_test=stream["X_test"],
        Y_test=stream["Y_test"],
        max_rounds=5,
        epochs_per_round=3,
        warm_epochs=4,
        hidden=48,
        lr=0.08,
        n_perm=6,
        rf_estimators=40,
        seed=seed,
        optimizer="SGD",
    )
    shifted = stream["pool_shifted"]
    schemes = {}
    for name, result in out["schemes"].items():
        selected = result["selected_idx"]
        summary = dict(result["summary"])
        summary["frac_shifted"] = float(shifted[selected].mean()) if selected.size else float("nan")
        summary["loss_curve"] = result["loss_curve"]
        summary["acc_curve"] = result["acc_curve"]
        schemes[name] = summary
        print(
            f"  {name:24s} acc {summary['acc_final']:.3f}  "
            f"lift {summary['acc_lift']:+.3f}  shifted {summary['frac_shifted']:.3f}",
            flush=True,
        )
    return {
        "seed": seed,
        "shift_idx": stream["shift_idx"],
        "feature_weights": out["feature_weights"],
        "schemes": schemes,
    }


def _average(runs: list[dict]) -> dict:
    names = list(runs[0]["schemes"])
    weight_keys = ("po_vimp", "rf_vimp", "MMD_vimp", "pooled")
    weights = {
        key: np.mean([run["feature_weights"][key] for run in runs], axis=0)
        for key in weight_keys
    }
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
            "loss_drop": _mean([row["loss_drop"] for row in rows]),
            "frac_shifted": _mean([row["frac_shifted"] for row in rows]),
            "acc_curve": np.mean([row["acc_curve"] for row in rows], axis=0),
        }
    return {"weights": weights, "schemes": schemes, "shift_idx": runs[0]["shift_idx"]}


def _top(weight, k: int = 3) -> list[int]:
    return [int(i) for i in np.argsort(-np.asarray(weight))[:k]]


def _group_mean(schemes: dict, weight_name: str, field: str, skip=()) -> float:
    vals = [
        row[field]
        for name, row in schemes.items()
        if row["weight"] == weight_name and name not in skip
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def render_board(avg: dict, n_seeds: int) -> str:
    weights = avg["weights"]
    schemes = avg["schemes"]
    shift_idx = [int(i) for i in avg["shift_idx"]]
    ranking = {
        "PermuCATE": _top(weights["po_vimp"]),
        "domain forest": _top(weights["rf_vimp"]),
        "MMD-LOCO": _top(weights["MMD_vimp"]),
        "pooled": _top(weights["pooled"]),
    }
    # top_weight ignores the query rule, so it is not in the "weight added
    # on top of a rule" average. It is the ceiling of the pooled score.
    skip_ceiling = {"top_weight_pooled"}
    uniform_shift = _group_mean(schemes, "uniform", "frac_shifted")
    pooled_shift = _group_mean(schemes, "pooled", "frac_shifted", skip=skip_ceiling)
    po_shift = _group_mean(schemes, "po_risk", "frac_shifted")
    uniform_acc = _group_mean(schemes, "uniform", "acc_final")
    pooled_acc = _group_mean(schemes, "pooled", "acc_final", skip=skip_ceiling)
    best_name = max(schemes, key=lambda name: schemes[name]["acc_final"])
    best = schemes[best_name]
    entropy = schemes["entropy"]
    entropy_pooled = schemes["entropy_pooled"]
    ceiling = schemes["top_weight_pooled"]

    lines = [
        "# 带分布偏移权重的深度主动采样",
        "",
        "查询规则沿用 deep active learning 里的监督策略：random、entropy、margin、least confidence、BALD、cluster-margin、core-set、k-means、BADGE。权重是后加的。样本权重是秩百分位。常数权重不改变原来的排序。",
        "",
        "参考窗前一半是源分布（T = 0），后一半是目标分布（T = 1）。目标分布把前 3 个坐标平移 1.25 个标准差，并把这 3 个坐标上的类别权重反号。池子里一半是源、一半是目标。测试集全是目标。网络先在源分布上热身，每个方案从同一组参数出发，优化器是不重置的 SGD。",
        "",
        "维度权重有三份，再平均成一份：PermuCATE 伪结果风险的条件置换下降、随机森林域分类器的重要度、RBF MMD 的留一坐标下降。`pooled` 用这份合并权重，算每个点离源均值的加权平方距离。`rf`、`mmd`、`po_vimp` 是这三份拆开之后各自的距离。`po_risk` 是把池子里的点都当成 T = 1 之后的逐行伪结果残差，标签在这里是已经观测到的，采样决定的是哪些行进入 SGD。`both` 是合并距离和伪结果残差这两种秩的乘积。",
        "",
        f"两次重复（种子 {', '.join(str(s) for s in SEEDS)}）。表里是均值。",
        "",
        "## 维度权重有没有指到平移的坐标",
        "",
        f"真正被平移、并且类别规则被反号的坐标是 `{shift_idx}`。",
        "",
        "| 方法 | 前三名 |",
        "| --- | --- |",
    ]
    for name, top in ranking.items():
        lines.append(f"| {name} | `{top}` |")
    hit = len(set(ranking["pooled"]) & set(shift_idx))
    lines.extend(
        [
            "",
            f"合并后的前三名里有 {hit} 个落在平移坐标上。",
            "",
            "## 采样曲线",
            "",
            "准确率的起点是源模型在目标测试集上的表现，每个方案相同。lift 是终点减去起点。shifted 是抽到的行里真正来自目标分布的比例。池子里这个比例是 0.50。",
            "",
            "| 方案 | 查询 | 权重 | 终点准确率 | lift | 准确率面积 | 终点损失 | 损失下降 | 目标行比例 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    order = sorted(schemes, key=lambda name: -schemes[name]["acc_final"])
    for name in order:
        row = schemes[name]
        lines.append(
            "| {name} | {strategy} | {weight} | {acc:.3f} | {lift:+.3f} | {auc:.3f} | {loss:.3f} | {drop:+.3f} | {frac:.3f} |".format(
                name=name,
                strategy=row["strategy"],
                weight=row["weight"],
                acc=row["acc_final"],
                lift=row["acc_lift"],
                auc=row["acc_auc"],
                loss=row["loss_final"],
                drop=row["loss_drop"],
                frac=row["frac_shifted"],
            )
        )
    curve_len = len(next(iter(schemes.values()))["acc_curve"])
    queried = [i * QUERY_BATCH for i in range(curve_len)]
    header = "| 方案 | " + " | ".join(str(n) for n in queried) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in queried) + " |"
    lines.extend(
        [
            "",
            "## 准确率路径",
            "",
            f"列是已经送进 SGD 的池子行数。0 是只在源分布上热身、还没抽池子的时候。之后每列多 {QUERY_BATCH} 行。数字是两个种子的平均测试准确率。",
            "",
            header,
            rule,
        ]
    )
    for name in order:
        curve = schemes[name]["acc_curve"]
        cells = " | ".join(f"{value:.3f}" for value in curve)
        lines.append(f"| {name} | {cells} |")
    if pooled_shift > uniform_shift + 0.02:
        shift_sentence = (
            f"加上合并权重之后，抽到的目标行从 {uniform_shift:.3f} 升到 {pooled_shift:.3f}。"
            "权重把查询从源分布的行上挪开了。"
        )
    elif uniform_shift > pooled_shift + 0.02:
        shift_sentence = (
            f"合并权重并没有多抽目标行：无权重是 {uniform_shift:.3f}，合并权重是 {pooled_shift:.3f}。"
        )
    else:
        shift_sentence = (
            f"合并权重和无权重抽到的目标行比例接近，分别是 {pooled_shift:.3f} 和 {uniform_shift:.3f}。"
        )
    if pooled_acc > uniform_acc + 0.01:
        acc_sentence = (
            f"终点准确率也分开了：合并权重平均 {pooled_acc:.3f}，无权重平均 {uniform_acc:.3f}。"
        )
    elif uniform_acc > pooled_acc + 0.01:
        acc_sentence = (
            f"终点准确率没有跟着目标行一起上升。无权重平均 {uniform_acc:.3f}，合并权重平均 {pooled_acc:.3f}。"
        )
    else:
        acc_sentence = (
            f"终点准确率几乎一样：无权重 {uniform_acc:.3f}，合并权重 {pooled_acc:.3f}。"
        )
    entropy_sentence = (
        f"同一条 entropy 规则，无权重终点准确率 {entropy['acc_final']:.3f}、目标行 {entropy['frac_shifted']:.3f}；"
        f"乘上合并权重之后是 {entropy_pooled['acc_final']:.3f} 和 {entropy_pooled['frac_shifted']:.3f}。"
    )
    component_rows = (
        ("随机森林", schemes["entropy_rf"]),
        ("MMD", schemes["entropy_mmd"]),
        ("PermuCATE", schemes["entropy_po_vimp"]),
    )
    component_bits = [
        f"{label} {row['acc_final']:.3f}（目标行 {row['frac_shifted']:.3f}）" for label, row in component_rows
    ]
    best_component = max(component_rows, key=lambda item: item[1]["acc_final"])[0]
    component_sentence = (
        "三份维度权重拆开后只乘在 entropy 上："
        + "，".join(component_bits)
        + f"。其中终点准确率最高的是{best_component}。"
        f"只按随机森林做加权随机，目标行比例是 {schemes['random_rf']['frac_shifted']:.3f}；"
        f"只按 MMD 是 {schemes['random_mmd']['frac_shifted']:.3f}。"
    )
    bald = schemes["bald"]
    bald_pooled = schemes["bald_pooled"]
    cluster = schemes["cluster_margin"]
    cluster_pooled = schemes["cluster_margin_pooled"]
    extra_sentence = (
        f"BALD 无权重终点准确率 {bald['acc_final']:.3f}、目标行 {bald['frac_shifted']:.3f}；"
        f"乘上合并权重是 {bald_pooled['acc_final']:.3f} 和 {bald_pooled['frac_shifted']:.3f}。"
        f"Cluster-margin 无权重是 {cluster['acc_final']:.3f} 和 {cluster['frac_shifted']:.3f}，"
        f"加权后是 {cluster_pooled['acc_final']:.3f} 和 {cluster_pooled['frac_shifted']:.3f}。"
    )
    po_sentence = (
        f"把池子全部当成 T = 1 的伪结果残差，抽到的目标行比例是 {po_shift:.3f}。"
        "这个权重用了已经观测到的标签，和只看坐标的合并权重不是同一件事。"
    )
    ceiling_sentence = (
        f"只按合并权重取最大的 `top_weight_pooled` 抽到的目标行比例是 {ceiling['frac_shifted']:.3f}，"
        f"终点准确率 {ceiling['acc_final']:.3f}。这是这份权重本身的上限，没有再和不确定性相乘。"
    )
    early = schemes["entropy_pooled"]["acc_curve"]
    late_gap = float(early[-1] - schemes["entropy"]["acc_curve"][-1])
    plain = schemes["entropy"]["acc_curve"]
    path_sentence = (
        f"路径上，entropy × 合并权重在第 {QUERY_BATCH} 行就已经到 {early[1]:.3f}，"
        f"第 {3 * QUERY_BATCH} 行 {early[3]:.3f}，第 {4 * QUERY_BATCH} 行 {early[4]:.3f}，最后 {early[-1]:.3f}。"
        f"无权重的 entropy 同期是 {plain[1]:.3f}、{plain[3]:.3f}、{plain[4]:.3f}、{plain[-1]:.3f}。"
        f"两条路径的终点差是 {late_gap:.3f}。"
        f"只按合并权重取最大的方案第 {QUERY_BATCH} 行是 {schemes['top_weight_pooled']['acc_curve'][1]:.3f}，"
        f"起步更快，第 {3 * QUERY_BATCH} 行被 entropy × 合并权重超过。"
    )
    lines.extend(
        [
            "",
            "## 读这张表",
            "",
            path_sentence,
            shift_sentence,
            acc_sentence,
            entropy_sentence,
            component_sentence,
            extra_sentence,
            po_sentence,
            ceiling_sentence,
            f"终点准确率最高的是 `{best_name}`，{best['acc_final']:.3f}，比起点高 {best['acc_lift']:+.3f}。",
            "",
            "CIFAR-100 没有进这次比较。`load_cifar100_arrays` 可以把平坦化的 CIFAR-100 送进同一个 `IndexedArrayDataset`，采样循环吃的就是这个 dataloader。这次用的是上面这个已知平移坐标的流，这样维度权重和抽到的行都能对上号。",
            "",
            f"重复次数 {n_seeds}。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    runs = [run_once(seed) for seed in SEEDS]
    avg = _average(runs)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    board = render_board(avg, n_seeds=len(runs))
    (RESULT_DIR / "BOARD.md").write_text(board)
    payload = {"runs": runs, "average": avg}
    (RESULT_DIR / "curves.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"wrote {RESULT_DIR / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
