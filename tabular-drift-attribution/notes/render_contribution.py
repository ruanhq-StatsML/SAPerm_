"""One-page note: block contribution from a conditional counterfactual."""

from pathlib import Path

from fpdf import FPDF

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
OUT = Path(__file__).resolve().parents[1] / "contribution.pdf"
ARTIFACT = Path("/opt/cursor/artifacts/tabular_drift_contribution.pdf")


class Note(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("zh", size=8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def block(pdf, text, size, height):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("zh", size=size)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def table(pdf, headers, rows, widths):
    line_h = 6.4
    needed = (1 + len(rows)) * line_h + 4
    if pdf.get_y() + needed > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(235, 235, 235)
    pdf.set_font("zh", size=9)
    for header, width in zip(headers, widths):
        pdf.cell(width, line_h, header, border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(line_h)
    pdf.set_font("zh", size=8.5)
    for row in rows:
        pdf.set_x(pdf.l_margin)
        for value, width in zip(row, widths):
            pdf.cell(width, line_h, str(value), border=1, new_x="RIGHT", new_y="TOP")
        pdf.ln(line_h)
    pdf.ln(2)


def main():
    pdf = Note(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font("zh", "", FONT)
    pdf.add_page()
    block(pdf, "反事实贡献：每一块离开参考条件多少", 16, 8)
    block(pdf, "2026年10月28日（太平洋时间）", 9, 6)
    pdf.ln(2)
    block(
        pdf,
        "贡献是一块特征在「今天其余列，加上昨天的全量特征」下，离参考条件分布有多远，再除以全部块的距离。"
        "生成器只在第 0 批上拟合。工作行从当天观测开始，每写回一块，下一块就看到这个更新。",
        11,
        6,
    )
    pdf.ln(1)
    block(
        pdf,
        "Z(t, b)  <-  P_ref( X_b  |  X_{-b} = Z(t,-b),  X(t-1) )",
        11,
        6,
    )
    block(
        pdf,
        "c_t(b)  =  d( X_t,b , Z(t,b) )  /  sum_b' d( X_t,b' , Z(t,b') )",
        11,
        6,
    )
    pdf.ln(1)
    block(
        pdf,
        "c_t(b) 就是这一批里块 b 的贡献，输出字段 contribution。它和 block_prop 是同一个数，加总为 1。"
        "d 可以是 mean_diff、Wasserstein、MMD 或 H-divergence。特征贡献是各列绝对差占总绝对差的比例，字段 feature_contribution。",
        11,
        6,
    )
    pdf.ln(1)
    block(pdf, "默认生成器", 13, 7)
    block(
        pdf,
        "stub 不训练模型。对当前行，用今天其余列和昨天全量特征找参考批次里最近的一行，把那一行的目标块捐出来。"
        "CTGAN、TabFORGE、DiffLTM 走同一个迭代更新。DiffLTM 的每一行是 conditional sampling schema："
        "bounds、equality、ranges、weights。目标块是 free，今天其余列和昨天全量写进 equality 或 ranges。"
        "SMC、Harpoon、rejection 读同一份 schema。",
        11,
        6,
    )
    pdf.ln(1)
    block(pdf, "合成面板", 13, 7)
    block(
        pdf,
        "800 行、8 个批次、100 列，四块各 25 列。从第 3 批起，block_1 的 25 列各加 1.5。"
        "距离是 mean_diff。漂移前最大一块是 0.282。漂移后 block_1 的贡献在 0.325 到 0.329，五个批次都是第一名。",
        11,
        6,
    )
    table(
        pdf,
        ["批次", "block_0", "block_1", "block_2", "block_3", "第一名"],
        [
            ["1", "0.250", "0.250", "0.251", "0.249", "未漂移"],
            ["2", "0.282", "0.238", "0.242", "0.238", "未漂移"],
            ["3", "0.241", "0.329", "0.216", "0.215", "block_1"],
            ["4", "0.243", "0.325", "0.215", "0.218", "block_1"],
            ["5", "0.243", "0.326", "0.214", "0.217", "block_1"],
            ["6", "0.244", "0.325", "0.214", "0.217", "block_1"],
            ["7", "0.243", "0.326", "0.211", "0.219", "block_1"],
        ],
        [22, 28, 28, 28, 28, 36],
    )
    block(
        pdf,
        "第 3 批绝对差最大的五列是 f36、f29、f37、f46、f31，都落在 block_1（f25 到 f49）。"
        "有效行比例 0.983 到 0.999。mean_diff、Wasserstein、MMD、H-divergence 在漂移开始后都把 block_1 排在第一。",
        11,
        6,
    )
    pdf.ln(1)
    block(
        pdf,
        "MovieLens、IEEE-CIS、Home Credit 的加载已经接上。原始文件不在仓库里，这三份还没有贡献表。",
        11,
        6,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    ARTIFACT.write_bytes(OUT.read_bytes())
    print(f"pages={pdf.pages_count} {OUT}")


if __name__ == "__main__":
    main()
