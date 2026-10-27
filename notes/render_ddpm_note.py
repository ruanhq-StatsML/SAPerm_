"""Render the DDPM counterfactual-attribution note to PDF."""

from pathlib import Path

from fpdf import FPDF

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
OUT = Path("/workspace/notes/ddpm_counterfactual_attribution.pdf")
ARTIFACT = Path("/opt/cursor/artifacts/ddpm_counterfactual_attribution.pdf")


class Note(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("zh", size=8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _block(pdf, text, size, height):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("zh", size=size)
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def h2(pdf, text):
    pdf.ln(2)
    _block(pdf, text, 13, 7)
    pdf.ln(1)


def body(pdf, text):
    _block(pdf, text, 10.5, 6)
    pdf.ln(1)


def table(pdf, headers, rows, widths):
    line_h = 6.2
    needed = (1 + len(rows)) * line_h + 4
    if pdf.get_y() + needed > pdf.h - pdf.b_margin and needed < pdf.h - 40:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.set_font("zh", size=8.5)
    pdf.set_fill_color(235, 235, 235)
    for header, width in zip(headers, widths):
        pdf.cell(width, line_h, header, border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(line_h)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("zh", size=8)
    for row in rows:
        pdf.set_x(pdf.l_margin)
        for value, width in zip(row, widths):
            pdf.cell(width, line_h, str(value), border=1, new_x="RIGHT", new_y="TOP")
        pdf.ln(line_h)
        pdf.set_x(pdf.l_margin)
    pdf.ln(2)


def main():
    pdf = Note(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font("zh", "", FONT)
    pdf.add_page()
    _block(pdf, "条件扩散的反事实归因", 18, 9)
    _block(pdf, "一次只改一个坐标，分布差了多少", 12, 7)
    pdf.set_font("zh", size=9)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, "2026年10月27日（太平洋时间）", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    h2(pdf, "归因问的是什么")
    body(
        pdf,
        "一次归因问的是：其余条件保持这个窗口的观测值，只把条件向量里的一格换成参考路径，"
        "结果的条件分布离开当天观测多少。生成器是条件扩散模型。它在给定条件 c 时抽出一批结果，"
        "当作 P(Y | c)。分数是分布之间的距离比。这个比是条件分布的差异，不是结构因果效应，"
        "也不把风控、推荐或社交通道拆开。",
    )
    body(
        pdf,
        "这份记录里没有扩散步数、噪声日程和网络宽度，这里不补。北京东四看古城的查询在协议处停止，"
        "那一格没有训练扩散模型。",
    )

    h2(pdf, "编辑与分数")
    body(
        pdf,
        "条件向量在线性十站点验证面板上是 21 维，里面有协变量、干扰块，以及被怀疑的那一块滞后。"
        "对滞后 ℓ，反事实条件 c(ℓ) 只替换这一格，其余坐标保持观测值。",
    )
    body(
        pdf,
        "观测批是这个窗口的真实结果。反事实批是模型在 c(ℓ) 上的抽样。无干预批是模型在原条件 c 上的抽样。"
        "同一评估日的归一化分数是",
    )
    _block(
        pdf,
        "ρ(ℓ)  =  MMD²( P_obs ,  P(Y | c(ℓ)) )   /   MMD²( P_obs ,  P(Y | c) )",
        11,
        6,
    )
    pdf.ln(1)
    body(
        pdf,
        "分母是这一天不做这次编辑时，预测分布和观测已经有的距离。分子是改掉这一格之后的距离。"
        "核宽用训练期结果的边际中位距离，评估时不再另选。ρ(ℓ) > 2 是跑实验之前写死的线。",
    )
    body(
        pdf,
        "ρ 不带方向。头条句子用的是两批均值的符号差：正剂量要求观测批高于反事实批。"
        "距离 ρ、Δ^cf 和 δ² 在符号反过来的时候可以几乎不变，所以它们单独过不了方向这一步。",
    )

    h2(pdf, "一格要进头条，七步都要过")
    body(
        pdf,
        "头条窗口是线性十站点验证面板上的 episode 42、站点 0、第 10 天、dose = 1。"
        "反事实只改了 b0 的第 10 个滞后，从 1.063 放到 0.063。",
    )
    table(
        pdf,
        ["步", "问的是", "这一格"],
        [
            ["1", "哪些坐标被改了", "只有 b0 的 lag 10"],
            ["2", "编辑坐标还在盒子里吗", "z 从 +2.53 到 +0.09，线是 4"],
            ["3", "改之前是否已在外面", "异常日 z = +2.53，替换往中心走"],
            ["4", "背景是否都在盒子里", "s∞ = 2.58，落在人口统计"],
            ["5", "能否钳背景再当头条", "不能。这一格不需要钳"],
            ["6", "均值差方向是否一致", "观测减反事实，signed = +1.009"],
            ["7", "无干预那一对是否很小", "Δ_base = 0.021，Δ^cf = 0.674"],
        ],
        [14, 58, 108],
    )
    body(
        pdf,
        "写成头条的句子是：参考路径上的购买比观测低大约 1 个指数单位。"
        "Δ^cf = 0.674 和 δ² = 1.192 是距离，方向只记在 signed 里。",
    )
    body(
        pdf,
        "1920 格里有 2 格的 s∞ 大于 4，都在第 4 步被拒。观测条件和反事实条件上的越界坐标相同，"
        "拒绝它们的不是这次编辑。背景出了支持，两批之差会把这一格的编辑和模型在盒子外的误差混在一起。"
        "钳制只允许用在编辑坐标自己越过 4 的时候，钳过的数不是头条。盒子阈值是坐标标准化绝对值 4。",
    )
    table(
        pdf,
        ["窗口", "越界坐标", "观测 z", "反事实 z", "原始 Δ"],
        [
            ["ep 47，站点 5，第 6 天", "人口统计", "4.14", "4.14", "—"],
            ["ep 43，站点 7，第 10 天", "干扰块 b1 lag 10", "4.18", "4.18", "0.019"],
        ],
        [52, 42, 28, 32, 26],
    )

    h2(pdf, "五层才进入业务动作")
    body(pdf, "通过线在跑实验之前写死。真实面板的第一层只记假设检查。观测校准是必要条件。")
    table(
        pdf,
        ["层", "通过线"],
        [
            ["L1 排序", "至少三种构造过线"],
            ["L2 时间", "placebo、峰值位置、门控都过"],
            ["L3 校准", "PIT、Wasserstein、HSIC 都过"],
            ["L4 代理", "至少三个代理同向"],
            ["L5 敏感", "变异系数和 E-value 都过"],
        ],
        [36, 144],
    )
    body(pdf, "PIT 没过，或者核宽的变异系数过了 0.20，分数就停在低可信度，不进入业务动作。")

    h2(pdf, "四张面板停在哪里")
    table(
        pdf,
        ["面板", "L1", "L2", "L3", "L4", "L5", "可信度"],
        [
            ["长记忆，峰值应在第 10 天", "通过", "未过", "未过", "通过", "未过", "低"],
            ["短记忆，峰值应在第 1 天", "通过", "通过", "未过", "通过", "未过", "低"],
            ["北京 PM2.5，2014–2017", "—", "—", "—", "—", "—", "停止"],
            ["美国疫情，2020-03 至 2021-06", "假设未支持", "未过", "未过", "不确定", "未过", "低"],
        ],
        [58, 22, 16, 16, 18, 16, 20],
    )
    body(
        pdf,
        "长记忆的归一化 MMD²：第 9 天 224×，第 10 天 206×，第 15 天仍有 10.5×。"
        "声明的真值是 10 天，峰值落在第 9 天，placebo 也未过，时间这一层未过。"
        "短记忆的峰值在第 1 天，2.10×，placebo 滞后都低于 2，时间这一层通过。"
        "疫情面板整条扫描贴着 1，第 10 天是 1.00×，扫描的最大值在第 1 天。",
    )
    table(
        pdf,
        ["滞后", "长记忆", "短记忆", "疫情 南←东北"],
        [
            ["1", "0.10", "2.10", "1.43"],
            ["2", "1.84", "1.36", "1.23"],
            ["3", "0.22", "1.37", "1.29"],
            ["4", "2.71", "1.54", "1.00"],
            ["5", "4.59", "1.32", "0.96"],
            ["6", "2.63", "1.34", "0.83"],
            ["7", "45.52", "1.27", "0.64"],
            ["8", "196.6", "1.85", "1.08"],
            ["9", "224.1", "1.48", "1.05"],
            ["10", "205.73", "1.27", "1.00"],
            ["11", "0.17", "1.29", "1.02"],
            ["12", "0.31", "1.29", "1.00"],
            ["13", "0.39", "1.28", "1.00"],
            ["14", "1.68", "1.27", "1.24"],
            ["15", "10.54", "1.27", "1.12"],
        ],
        [28, 40, 40, 72],
    )
    body(
        pdf,
        "种进去的相对排序，长记忆上三组构造通过：剂量反应的一致率 0.90、Spearman 0.90；"
        "重叠度 0.83、0.80；外生隔离的效应量 −2.51，p = 0.034。"
        "单调的「越近越大」和 lag-10 峰值不相容，Spearman 0.16，这一组记在假设上。"
        "短记忆的时间距离 Spearman 是 −0.685，差 0.015 没到 −0.7，这一组记失败，另外三组通过，第一层仍然通过。",
    )
    table(
        pdf,
        ["校准与敏感", "长记忆", "短记忆"],
        [
            ["Wasserstein / 上限", "0.082 / 0.095", "0.078 / 0.093"],
            ["HSIC 比值", "0.87", "0.86"],
            ["PIT", "四维都拒绝均匀", "第 3 维 p = 5e-5"],
            ["核宽变异系数，线 0.20", "0.71", "0.25"],
            ["架构变异系数，线 0.30", "0.23", "0.00"],
            ["E-value，线 2", "19.9", "32.9"],
        ],
        [70, 55, 55],
    )

    h2(pdf, "两张真实面板")
    body(
        pdf,
        "北京，东四看古城，滞后 10 天。五段异常的排序一致率 0.30，Spearman −0.60，主查询倍数 0.30。"
        "协议在这里停止，没有训练扩散模型。10 天前的古城 PM2.5 不能按异常强弱排东四当天的小时分布。",
    )
    body(
        pdf,
        "美国疫情，南部各州的对数发病率对东北部 10 天前的区域发病率。归一化倍数 1.00，"
        "区间盖在一个没有超过基线的原始 MMD 上。业务窗口里南部确实异常（z = 2.49）。"
        "这条 10 天的跨区反事实没有解释它。观测 PIT 和 Wasserstein 都失败，"
        "这个「没有超过基线」本身也不可信。",
    )

    h2(pdf, "这套逻辑现在能说什么")
    body(
        pdf,
        "模拟面板说明协议能抓住种进去的剂量、重叠和隔离，也能在短记忆上把峰值放回第 1 天。"
        "持续性会把长滞后的峰值从第 10 天挪到第 9 天。扩散模型的 PIT 还没均匀。"
        "MMD 对核宽的变异系数超过 0.20。这三件事没过之前，分数不进入业务动作。",
    )
    body(
        pdf,
        "购买行为的归因因此落在一个窗口、一个坐标上：哪一格被改了，编辑之后还在不在训练盒子里，"
        "两批均值差的符号是否和这次编辑一致，无干预的那一对是否小到可以忽略。"
        "多个窗口的 ρ 不合成一个总量。北京那一格没有模型，就不写句子。",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    ARTIFACT.write_bytes(OUT.read_bytes())
    print(f"pages={pdf.pages_count} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
