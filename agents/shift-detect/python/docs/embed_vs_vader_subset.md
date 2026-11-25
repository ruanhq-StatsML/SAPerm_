# Embedding on top of VADER？Subset summarization 能替代 \(Z\) 吗？

结论先说：**不能把 VADER / driving subset 当成 embedding 的替代品。**  
Subset 是对**某一条已命名概念轴**的压缩总结；embedding 仍是通用 residual / concept 载体。  
两者角色不同，经验上三者对照（\(M\)-only / \(Z\)-only / \(Z{+}M\)）也分得开。

---

## 1. 两个对象不要混

| 对象 | 是什么 | 能做什么 | 不能做什么 |
| --- | --- | --- | --- |
| `vader_sent` / driving subset \(S\subset M\) | 显式、低维、可命名的情感（等）坐标 | 检测**对齐该轴**的 covariate；localize「谁在驱动」 | 代表文本的全部语义 / 未命名 concept |
| embedding \(Z\)（PCA 后并入 \(X\)） | 稠密、不可命名的语义表示 | 捕获 subset 没写进词典的 residual；heldout / concept 级 shift | 单独点名「是情感还是句法」 |

所以：

- **加入 feature** = 在 \(X\) 里多开几列（VADER 是 added features，没错）；
- **subset summarization** = 检测/归因之后，用 LOCO·carrier 把 \(M\) 压成少数驱动维；
- **≠** 用这几维替换 \(Z\)。

Subset 回答的是 *which named coords drive this window*；  
embedding 回答的是 *is there still shift orthogonal to those names*。

---

## 2. 三套对照：什么时候 \(M\) 够、什么时候必须 \(Z\)

记 \(\rho_{PO}=PO(Z{+}M)/PO(Z)\)。

### (i) 情感对齐 ≈ 落在 VADER 轴上 → \(M\)（甚至 subset）可作**该轴的**充分统计

Feature-only `sentiment_shift`：baseline \(0.40\) → meta \(1.00\)（\(\Delta{+}0.60\)）。  
此时「只用 VADER / subset 做检测」在**该 DGP** 上够用——因为 shift 被设计成极性结构。

但这是 **axis-sufficient**，不是 **representation-sufficient**：  
换一个不落在情感词典上的 shift，同一 subset 立刻失效。

### (ii) 情感窗口 + 已有 \(Z\) → \(\rho_{PO}\approx0.25\)：**交互吸收，不是替换**

Real-data `sentiment_meta`：DR power \(0{\to}0\)，\(\rho_{PO}\approx0.25\)，\(\Delta\)CF\(\approx0\)。

读法：

- \(Z\) 已经「看见」一部分情感相关变异；
- 把 VADER 加进 nuisance 后，PO-risk **塌缩**（absorb）；
- 塌缩的前提是 \(X=[Z,M]\)——是 **\(M\) 补全 \(Z\) 的 nuisance**，不是「有了 \(M\) 删掉 \(Z\)」。

若用 subset **替换** \(Z\)（只留 `vader_*`），你测到的是「极性窗是否可分」，  
丢掉的是：同一批文档上，embedding 里与极性正交的结构是否仍在 shift。  
\(\rho_{PO}\) 这个交互证据本身就**定义在** \(PO(Z{+}M)\) vs \(PO(Z)\) 上——没有 \(Z\) 就没有这条签名。

### (iii) Heldout / concept residual → \(Z\) 不可去，subset 只做 localize

`train_vs_heldout`：RF/DR 两侧 power \(=1.00\)，\(\rho_{PO}\approx1.03\)，\(\Delta\)CF \(=+0.23\)。

读法：

- PO **不吸收** → shift 不在 VADER（或整块 \(M\)）能写尽的轴上；
- 警报在 \(Z\)（concept residual）；
- meta 的增量是 **CF 点名**（\(+0.23\)），不是替代检测。

若删掉 embedding、只留 VADER subset：heldout 上你失去主信号，只剩一条窄情感轴——**会漏报 concept-level shift**。

Syntax HARD 同理：feature-only meta \(0.10\to1.00\) 说明**句法轴**可自给；  
但 embed+meta 上 \(\rho_{PO}\approx0.19\) + \(\Delta\)CF \(+0.09\) 仍是「\(Z\) 打底 + \(M\) localize」，不是「subset 换掉 \(Z\)」。

---

## 3. 形式化：subset 是 summarization，不是 span

设 \(M=(M_1,\ldots,M_p)\)，driving subset \(S=\{M_j:j\in\mathcal{J}\}\) 由 carrier / LOCO 选出。

- **Summarization（我们做的）**  
  \[
  \text{claim}:\quad
  \text{shift along sentiment}
  \;\approx\;
  \text{shift visible in }S
  \quad(\text{e.g. }\{\texttt{compound},\texttt{neg},\texttt{polar\_gap},\texttt{neu}\}).
  \]
  检验：对 \(\mathcal{J}\) 做 block/coord LOCO；对阈值 \(\mathbf{1}\{M_j>\tau\}\) 同构。

- **Substitution（不成立的）**  
  \[
  \text{false claim}:\quad
  \mathcal{L}(Y\mid Z,M)=\mathcal{L}(Y\mid S)
  \quad\text{or}\quad
  PO(S)=PO(Z).
  \]
  Heldout \(\rho_{PO}\approx1\) + power\((Z)=1\) 直接否决「\(S\) spans \(Z\)」。

因此：**embedding 仍然要 on top of / alongside VADER**——更准确说是

\[
X=[Z_{\mathrm{PCA}},\,M],\qquad
\text{然后 }S\subset M\text{ 做归因摘要}.
\]

顺序是 **fuse → test → summarize**，不是 **summarize → drop \(Z\)**。

---

## 4. 产品陈述（两句话）

1. VADER / subset：**named axis features**——加进来提高对齐轴上的 power，并给出可引用的驱动维。  
2. Embedding：**residual concept backbone**——在 \(\rho_{PO}\not\ll 1\) 的 regime 上不可删；subset 不能替代它，只能在它之上做 localization。

何时可以「几乎只用 VADER」？仅当问题**先验**就是情感窗检测、且你接受漏掉 concept residual——那是产品 scope 选择，不是统计上 \(S\succeq Z\)。
