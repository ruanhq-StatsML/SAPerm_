"""Count conditioned columns and the discrete split, then say what each generator does.

The synthetic panel is the check. Every column is an independent standard
normal, so none are discretized. Each block update still conditions on 175
columns: 75 from today and 100 from yesterday. The free block is the other 25.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.digestion import block_account, digest, generator_plan
from data.loaders import load_synthetic_panel
from run import run


GENERATORS = ("stub", "ctgan", "tabforge", "diffltm")


def synthetic_report(loader_args=None) -> str:
    args = {"N": 800, "T": 8, "F": 100, "seed": 0}
    if loader_args:
        args.update(loader_args)
    data = load_synthetic_panel(**args)
    names = data["col_names"]
    digested = digest(data["X_batched"][0], names)
    lines = [
        "# 合成面板：条件了多少列，哪些列要离散",
        "",
        "800 个实体、8 天、100 列，四块各 25 列。列之间独立，都是标准正态。",
        "从第 3 天起，`block_1`（f25–f49）每列加 1.5。同一行号是同一个实体，所以昨天就是上一天的那一行。",
        "",
        "工作行从当天观测开始。更新一块时，这一块是 free，不进入条件。",
        "条件是当天其余列，加上昨天的全量列（含昨天的这一块）。写回之后，下一块看到的当天其余列里已经有这个新值。昨天始终是观测到的上一天。",
        "",
        f"参考批上的消化：离散 {len(digested['discrete'])} 列，连续 {len(digested['continuous'])} 列。",
        "规则是参考批里取值像整数，并且不同取值不超过 32。正态列不满足，所以一列都不离散。",
        "",
        "| 块 | free | 今天其余 | 昨天全量 | 条件合计 | equality | ranges |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for block, cols in data["blocks"].items():
        account = block_account(names, cols, digested["discrete"])
        lines.append(
            f"| {block} | {len(account['free'])} | {len(account['conditioned_today'])} "
            f"| {len(account['conditioned_yesterday'])} | "
            f"{len(account['equality']) + len(account['ranges'])} | "
            f"{len(account['equality'])} | {len(account['ranges'])} |"
        )
    lines += [
        "",
        "四块的个数一样，因为每块都是 25 列。`block_1` 的 free 是 f25–f49。",
        "它的 175 个条件列全是 ranges，equality 是空的。",
        "",
        "贡献几乎不动是这个设计的结果。+1.5 只加在 free 的那一块上，条件列里没有当天的这个平移。",
        "参考行也都没有这个平移，所以反事实还在 0 附近，距离留在 `block_1`。",
        "从第 4 天起，昨天的 `block_1` 已经带了平移，但这 25 列对每一行参考的距离差不多，匹配仍由其余列决定。",
        "真实表里列是相依的，同一条条件会把反事实拉走，份额才会动。合成面板比那容易。",
        "",
        "## 四个生成器怎么吃这些列",
        "",
        "下面是 `block_1` 这一次更新。stub、CTGAN、TabFORGE、DiffLTM 走同一套列，消化不一样。",
        "",
    ]
    account = block_account(names, data["blocks"]["block_1"], digested["discrete"])
    for name in GENERATORS:
        plan = generator_plan(name, account, digested)
        lines.append(f"- `{plan['generator']}`：{plan['note']}")
    lines += [
        "",
        "CTGAN 的分箱是模型限制，不是数据里这些列是类别。DiffLTM 不把它们收成等式。",
        "stub 和 TabFORGE 也不离散。",
        "",
    ]
    cfg = {
        "dataset": "synthetic",
        "seed": 0,
        "generator": "stub",
        "metric": "mean_diff",
        "loader_args": args,
    }
    results = run(cfg)
    lines.append("stub 在这份面板上的块贡献：")
    lines.append("")
    lines.append("| 天 | block_0 | block_1 | block_2 | block_3 | 第一名 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        prop = row["contribution"]
        flag = "block_1" if row["top"] == "block_1" else "未漂移"
        lines.append(
            f"| {row['t']} | {prop['block_0']:.3f} | {prop['block_1']:.3f} | "
            f"{prop['block_2']:.3f} | {prop['block_3']:.3f} | {flag} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/synthetic_digestion.md")
    args = parser.parse_args()
    text = synthetic_report()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text, end="")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
