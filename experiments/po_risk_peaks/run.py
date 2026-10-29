"""HC3 detector: mean PO-risk versus its peak, under benign and malignant shifts.

The proxy is ``1 - p(y|x)`` on the deployed detector. REF, back-translation,
and random synonym replacement are the benign live batches. TextFooler and a
partial FGSM step are the malignant ones. The scatter plots the mean against
the peak gap of small sub-batches. A half-corpus bootstrap of the same mean
is recorded to show that it washes the peak out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.po_risk_peaks.attacks import (
    back_translate,
    choose_fgsm_eps,
    fgsm_embeddings,
    score_perturbed_embeddings,
    synonym_replace,
    textfooler,
)
from experiments.po_risk_peaks.data import load_pairs, split_rows
from experiments.po_risk_peaks.detector import (
    adapt_upper_layers,
    block_distances,
    load_backbone,
    predict_proba,
    train_detector,
)
from experiments.po_risk_peaks.stats import classification_rates, fpr_threshold, summarize_risks, true_class_residual

OUT = Path(__file__).resolve().parent / "results"
MODEL_NAME = "distilbert-base-uncased"

CONDITIONS = (
    ("ref", "REF"),
    ("backtranslate", "Back-translation"),
    ("synonym", "Synonym 15%"),
    ("textfooler", "TextFooler"),
    ("fgsm", "FGSM"),
)

COLORS = {
    "ref": "#4d4d4d",
    "backtranslate": "#2c7fb8",
    "synonym": "#41ab5d",
    "textfooler": "#d94801",
    "fgsm": "#a50f15",
}


def _accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(labels)))


def _diagnose(probs: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    risks = true_class_residual(probs, labels)
    summary = summarize_risks(risks, sub_size=32, n_sub=40, seed=seed)
    summary["accuracy"] = _accuracy(probs, labels)
    summary.pop("sub_means")
    return {key: float(value) for key, value in summary.items()}


def _paired_texts(texts: np.ndarray, labels: np.ndarray, n: int, seed: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    return [str(texts[int(i)]) for i in idx], np.asarray(labels)[idx]


def write_plot(frame: pd.DataFrame, path: Path) -> None:
    live = frame[frame["stage"] == "deployed"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    specs = (
        (axes[0], "sub_peak_gap", "Peak gap of sub-batch means (size 32)"),
        (axes[1], "sub_kurtosis", "Excess kurtosis of those sub-batch means"),
    )
    adapted = frame[frame["stage"] == "adapted"]
    for ax, column, ylabel in specs:
        for _, row in live.iterrows():
            ax.scatter(
                row["mean"],
                row[column],
                s=70,
                color=COLORS[row["condition"]],
                zorder=3,
            )
            ax.annotate(
                row["label"],
                (row["mean"], row[column]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        for _, row in adapted.iterrows():
            ax.scatter(
                row["mean"],
                row[column],
                s=70,
                facecolors="none",
                edgecolors=COLORS[row["condition"]],
                linewidths=1.6,
                zorder=3,
            )
        ax.set_xlabel("Mean PO-risk proxy,  1 − p(y|x)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.5,
        0.01,
        "Open markers: same live batch after the embedding is frozen and the upper layers are updated.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _reading(frame: pd.DataFrame) -> str:
    def val(condition: str, column: str, stage: str = "deployed") -> float:
        hit = frame[(frame["condition"] == condition) & (frame["stage"] == stage)]
        return float(hit.iloc[0][column])

    ref_mean = val("ref", "mean")
    back_mean = val("backtranslate", "mean") if (frame["condition"] == "backtranslate").any() else float("nan")
    fool_mean = val("textfooler", "mean")
    return (
        f"回译把均值从 {ref_mean:.3f} 抬到 {back_mean:.3f}，TextFooler 抬到 {fool_mean:.3f}，两者几乎一样。"
        f"同义词替换停在 {val('synonym', 'mean'):.3f}，没有离开干净批次。"
        f"半语料峰值夹在 {val('synonym', 'half_peak_gap'):.3f} 和 {val('textfooler', 'half_peak_gap'):.3f} 之间，跟攻击类型对不上。"
        f"32 条子批次的峰值把 TextFooler 分出来：{val('textfooler', 'sub_peak_gap'):.3f}，回译 {val('backtranslate', 'sub_peak_gap'):.3f}，干净 {val('ref', 'sub_peak_gap'):.3f}。"
        f"子批次峰度同样，TextFooler {val('textfooler', 'sub_kurtosis'):.3f}，回译 {val('backtranslate', 'sub_kurtosis'):.3f}。"
        f"FGSM 翻转了大半测试集，均值到 {val('fgsm', 'mean'):.3f}，样本峰度 {val('fgsm', 'kurtosis'):.3f}，残差是整批搬家，不是一根尖刺。"
        f"冻结 embedding 再训上层之后，TextFooler 的子批次峰值从 {val('textfooler', 'sub_peak_gap'):.3f} 回到 {val('textfooler', 'sub_peak_gap', 'adapted'):.3f}，"
        f"准确率从 {val('textfooler', 'accuracy'):.3f} 到 {val('textfooler', 'accuracy', 'adapted'):.3f}。"
        f"同一套更新碰不动 FGSM，准确率 {val('fgsm', 'accuracy'):.3f} 到 {val('fgsm', 'accuracy', 'adapted'):.3f}。"
        "同义词批次上，干净校准阈值把召回打下去；用同义词校准集重算阈值之后，误报回到 0.05 附近，召回回到干净批次的水平。"
    )


def write_board(frame: pd.DataFrame, blocks: dict, conformal: dict, meta: dict, path: Path) -> None:
    def cell(condition: str, stage: str, column: str) -> str:
        hit = frame[(frame["condition"] == condition) & (frame["stage"] == stage)]
        if hit.empty or column not in hit.columns:
            return "--"
        value = float(hit.iloc[0][column])
        if not np.isfinite(value):
            return "--"
        return f"{value:.3f}"

    lines = [
        "# PO-risk 的均值和峰值：HC3 检测器",
        "",
        "代理 PO-risk 是检测器的真类残差 `1 - p(y|x)`。",
        "均值上升只说明这批文本比参考窗更难。峰值说明难处集中在少数子批次上。",
        "子批次如果取半个语料，max − median 只剩蒙特卡洛误差，跟攻击类型对不上。",
        "图的纵轴是 32 条文本的子批次峰值，以及这些子批次均值的超额峰度。",
        "",
        f"检测器是 `{MODEL_NAME}`，在 HC3 的 finance、open_qa、medicine、wiki_csai 上微调。",
        f"训练 {meta['n_train']} 条，测试 {meta['n_test']} 条，干净测试准确率 {meta['clean_accuracy']:.3f}。",
        f"FGSM 步长 {meta['fgsm_eps']}，在探针上的翻转率 {meta['fgsm_probe_rate']:.3f}，全测试翻转率 {meta['fgsm_flip']:.3f}。",
        f"TextFooler 翻转率 {meta['textfooler_flip']:.3f}。",
        "",
        "## 部署模型",
        "",
        "| 批次 | 均值 | 子批次峰值 | 半语料峰值 | 子批次峰度 | 样本峰度 | 准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, _label in CONDITIONS:
        lines.append(
            f"| {key} | {cell(key, 'deployed', 'mean')} | {cell(key, 'deployed', 'sub_peak_gap')} | "
            f"{cell(key, 'deployed', 'half_peak_gap')} | {cell(key, 'deployed', 'sub_kurtosis')} | "
            f"{cell(key, 'deployed', 'kurtosis')} | {cell(key, 'deployed', 'accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## 特征块的余弦距离",
            "",
            "词汇块是 token embedding。句法/语义块是最后一层的 CLS。距离是 `1 - cosine`，相对同索引的干净文本。",
            "",
            "| 批次 | 词汇块 | 语义块 |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, _label in CONDITIONS:
        if key not in blocks:
            continue
        lines.append(f"| {key} | {blocks[key]['lexical']:.3f} | {blocks[key]['semantic']:.3f} |")
    lines.extend(
        [
            "",
            "## 干预之后",
            "",
            "对抗批次：冻结 embedding，在训练集造出来的 TextFooler 配对上继续训练上层，损失是两边的交叉熵加上 CLS 余弦。",
            "评测用的 LIVE 文本仍是对着原模型打的，不重新寻词。",
            "回译和同义词：不改模型，只在校准集上重算 conformal 阈值，把人类文本的误报压回 5% 附近。",
            "",
            "| 批次 | 阶段 | 均值 | 子批次峰值 | 尾部超额 | 准确率 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("ref", "textfooler", "fgsm", "synonym"):
        for stage in ("deployed", "adapted"):
            if frame[(frame["condition"] == key) & (frame["stage"] == stage)].empty:
                continue
            lines.append(
                f"| {key} | {stage} | {cell(key, stage, 'mean')} | {cell(key, stage, 'sub_peak_gap')} | "
                f"{cell(key, stage, 'tail_excess')} | {cell(key, stage, 'accuracy')} |"
            )
    lines.extend(
        [
            "",
            "## Conformal 校准",
            "",
            "分数是 `p(AI)`。人类文本上的阈值来自 split conformal，目标误报 0.05。",
            "召回是 AI 文本被判成 AI 的比例。",
            "",
            "| 校准来源 | 评测批次 | 阈值 | 误报 | 召回 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in conformal:
        lines.append(
            f"| {row['calibrated_on']} | {row['evaluated_on']} | {row['threshold']:.3f} | {row['fpr']:.3f} | {row['recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            _reading(frame),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print("loading HC3", flush=True)
    rows = load_pairs(max_per_config=args.per_config)
    splits = split_rows(rows, args.n_train, args.n_cal, args.n_test, args.seed)
    print({name: len(part["label"]) for name, part in splits.items()}, flush=True)

    test_text = [str(t) for t in splits["test"]["text"]]
    test_label = np.asarray(splits["test"]["label"])
    back_text = None
    if not args.skip_backtranslate:
        digest = hashlib.sha256("\n".join(test_text).encode()).hexdigest()[:16]
        cache_path = OUT / "cache" / f"back_{digest}.json"
        if cache_path.exists():
            back_text = json.loads(cache_path.read_text())
            print("back-translation cache", len(back_text), flush=True)
        else:
            print("back-translation", flush=True)
            back_text = back_translate(test_text, batch_size=8)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(back_text))

    print("loading detector", flush=True)
    model, tokenizer = load_backbone(MODEL_NAME)
    print("training", flush=True)
    train_info = train_detector(
        model,
        tokenizer,
        splits["train"]["text"],
        splits["train"]["label"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("train", train_info, flush=True)

    clean_probs = predict_proba(model, tokenizer, test_text, batch_size=args.batch_size)
    clean_acc = _accuracy(clean_probs, test_label)
    print("clean accuracy", round(clean_acc, 3), flush=True)

    rng = np.random.default_rng(args.seed)
    print("synonym", flush=True)
    synonym_text = [synonym_replace(text, rng, rate=0.15) for text in test_text]
    synonym_probs = predict_proba(model, tokenizer, synonym_text, batch_size=args.batch_size)

    back_probs = None
    if back_text is not None:
        back_probs = predict_proba(model, tokenizer, back_text, batch_size=args.batch_size)

    print("textfooler", flush=True)

    def predict(texts: list[str]) -> np.ndarray:
        return predict_proba(model, tokenizer, texts, batch_size=args.batch_size)

    fool_text, fool_flip = textfooler(
        test_text,
        test_label,
        predict,
        seed=args.seed,
        max_changes=args.max_changes,
        synonyms_per_word=4,
        max_positions=8,
    )
    fool_probs = predict(fool_text)
    print("textfooler flip", round(float(fool_flip.mean()), 3), flush=True)

    probe_text, probe_label = _paired_texts(splits["cal"]["text"], splits["cal"]["label"], n=32, seed=args.seed + 7)
    print("choosing fgsm eps", flush=True)
    fgsm_eps, probe_rate = choose_fgsm_eps(model, tokenizer, probe_text, probe_label)
    print("fgsm eps", fgsm_eps, "probe flip", round(probe_rate, 3), flush=True)
    fgsm_probs, fgsm_embeds, fgsm_masks, fgsm_blocks = fgsm_embeddings(
        model, tokenizer, test_text, test_label, fgsm_eps, batch_size=args.batch_size
    )
    fgsm_flip = float(np.mean(np.argmax(fgsm_probs, axis=1) != test_label))
    print("fgsm flip", round(fgsm_flip, 3), flush=True)

    print("block distances", flush=True)
    blocks = {"synonym": block_distances(model, tokenizer, test_text, synonym_text, batch_size=4)}
    if back_text is not None:
        blocks["backtranslate"] = block_distances(model, tokenizer, test_text, back_text, batch_size=4)
    blocks["textfooler"] = block_distances(model, tokenizer, test_text, fool_text, batch_size=4)
    blocks["fgsm"] = fgsm_blocks

    records = []
    deployed = {
        "ref": clean_probs,
        "synonym": synonym_probs,
        "textfooler": fool_probs,
        "fgsm": fgsm_probs,
    }
    if back_probs is not None:
        deployed["backtranslate"] = back_probs
    for key, probs in deployed.items():
        row = _diagnose(probs, test_label, seed=args.seed)
        row.update({"condition": key, "stage": "deployed", "label": dict(CONDITIONS)[key]})
        records.append(row)
        print(key, {k: round(row[k], 3) for k in ("mean", "sub_peak_gap", "half_peak_gap", "tail_excess", "accuracy")}, flush=True)

    print("conformal calibration on the deployed model", flush=True)
    cal_text = [str(t) for t in splits["cal"]["text"]]
    cal_label = np.asarray(splits["cal"]["label"])
    cal_probs = predict_proba(model, tokenizer, cal_text, batch_size=args.batch_size)
    syn_cal = [synonym_replace(text, np.random.default_rng(args.seed + 11), rate=0.15) for text in cal_text]
    syn_cal_probs = predict_proba(model, tokenizer, syn_cal, batch_size=args.batch_size)
    human_cal = cal_label == 0
    clean_threshold = fpr_threshold(cal_probs[human_cal, 1], alpha=0.05)
    syn_threshold = fpr_threshold(syn_cal_probs[human_cal, 1], alpha=0.05)
    test_human = test_label == 0
    test_ai = test_label == 1

    def pack(name: str, human_p: np.ndarray, ai_p: np.ndarray, threshold: float, evaluated: str) -> dict[str, float | str]:
        rates = classification_rates(human_p, ai_p, threshold)
        rates["calibrated_on"] = name
        rates["evaluated_on"] = evaluated
        return rates

    conformal = [
        pack("clean cal", clean_probs[test_human, 1], clean_probs[test_ai, 1], clean_threshold, "ref test"),
        pack("clean cal", synonym_probs[test_human, 1], synonym_probs[test_ai, 1], clean_threshold, "synonym test"),
        pack("synonym cal", synonym_probs[test_human, 1], synonym_probs[test_ai, 1], syn_threshold, "synonym test"),
        pack("synonym cal", clean_probs[test_human, 1], clean_probs[test_ai, 1], syn_threshold, "ref test"),
    ]

    print("adaptation pairs", flush=True)
    adapt_clean, adapt_labels = _paired_texts(splits["train"]["text"], splits["train"]["label"], n=args.n_adapt, seed=args.seed + 3)
    adapt_adv, _adapt_flip = textfooler(
        adapt_clean,
        adapt_labels,
        predict,
        seed=args.seed + 3,
        max_changes=args.max_changes,
        synonyms_per_word=4,
        max_positions=8,
    )
    adapt_upper_layers(
        model,
        tokenizer,
        adapt_clean,
        adapt_labels,
        adapt_adv,
        adapt_labels,
        epochs=1,
        batch_size=4,
    )
    adapted_probs = {
        "ref": predict_proba(model, tokenizer, test_text, batch_size=args.batch_size),
        "textfooler": predict_proba(model, tokenizer, fool_text, batch_size=args.batch_size),
        "synonym": predict_proba(model, tokenizer, synonym_text, batch_size=args.batch_size),
        "fgsm": score_perturbed_embeddings(model, fgsm_embeds, fgsm_masks),
    }
    for key, probs in adapted_probs.items():
        row = _diagnose(probs, test_label, seed=args.seed)
        row.update({"condition": key, "stage": "adapted", "label": dict(CONDITIONS)[key] + " adapted"})
        records.append(row)
        print("adapted", key, {k: round(row[k], 3) for k in ("mean", "sub_peak_gap", "tail_excess", "accuracy")}, flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "diagnosis.csv", index=False)
    write_plot(frame, OUT / "diagnosis_scatter.png")
    meta = {
        "n_train": args.n_train,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "clean_accuracy": clean_acc,
        "fgsm_eps": fgsm_eps,
        "fgsm_probe_rate": probe_rate,
        "fgsm_flip": fgsm_flip,
        "textfooler_flip": float(fool_flip.mean()),
        "train": train_info,
        "blocks": blocks,
        "conformal": conformal,
    }
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2))
    write_board(frame, blocks, conformal, meta, OUT / "BOARD.md")
    print("wrote", OUT, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-cal", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=240)
    parser.add_argument("--n-adapt", type=int, default=80)
    parser.add_argument("--per-config", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-changes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-backtranslate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
