# Cross-modal readout: PO-risk proportion + signed interactions

Delivery narrative for the two intuitive objects. Both come from the same
block leave-out on **pseudo-outcome (PO) risk**
(`higher_order_interaction_PO_risk` in `ood_vimp.py`).

Notation (fixed batch, domain indicator $T$):

| symbol | meaning |
|---|---|
| $\mathrm{PO}(\mathrm{full})$ | PO risk with all modalities kept |
| $\mathrm{PO}(-m)$ | drop modality block $m$ only |
| $\mathrm{PO}(-m_1,-m_2)$ | drop both blocks |

Higher PO risk $\Rightarrow$ worse domain / outcome calibration on the held-out fold
(more “drift signal left unexplained”).

---

## A. Modality-specific **PO-risk proportion**（很直观）

**Question:** of the detectable domain gap, how much sits on modality $m$?

Define the leave-out lift of $m$:

\[
\Delta_m \;=\; \mathrm{PO}(-m) - \mathrm{PO}(\mathrm{full}).
\]

- $\Delta_m > 0$: dropping $m$ **hurts** → $m$ carries useful domain/outcome structure.
- Share (report as proportion, optionally clipped at 0):

\[
\pi_m \;=\; \frac{\Delta_m}{\sum_{m'\in\mathcal{M}} \Delta_{m'}}
\quad\text{(or vs a fixed baseline $\mathrm{PO}(-\mathrm{all})$)}.
\]

**How we read boards today**

| Board | What $\pi_m$ looks like in practice |
|---|---|
| `mean_shift` (shift only video / only audio) | $\pi_{\text{shifted}}$ dominates; others near 0 |
| concept-drift **combo power** T×A vs T×V | not $\pi_m$ itself, but a *detectability proxy*: T×A > T×V for CF/DR $\Rightarrow$ text+audio carries more usable mass under this DGP |

**Delivery line:** “modality-specific PO-risk proportion = share of leave-out lift.”

---

## B. Cross-modal **interaction**（正 / 负）

**Question:** do $m_1$ and $m_2$ move the gap **beyond** the sum of singles?

\[
I(m_1,m_2)
\;=\;
\mathrm{PO}(-m_1)+\mathrm{PO}(-m_2)-\mathrm{PO}(-m_1,-m_2).
\]

| Sign of $I$ | Reading |
|---|---|
| **$I>0$ (正)** | synergistic: the pair does extra joint work; dropping them one-at-a-time understates the joint leave-out |
| **$I<0$ (负)** | antagonistic / redundant cancellation: singles overstate what the joint drop does |
| $I\approx 0$ | approximately additive (no 2nd-order structure) |

Inference today: `interact_PO_{a}_{b}` = permutation p-value on $|I|$
(domain-label shuffle). **Sign for the narrative comes from the point estimate of $I$**,
not from the p-value. Companion detector: `crosscorr_{a}_{b}` =
$|\mathrm{corr}_{\mathrm{new}}-\mathrm{corr}_{\mathrm{ref}}|$ on block means
(dependence change in $X$).

**DGP that isolates this:** `interaction_shift` on `interact_pair`
(`_apply_pair_interaction` + product term in $Y$). Null type-I for
`interact_PO_*` / `crosscorr_*` is the overnight `no_shift` board (≈α so far).

**Delivery line:** “cross-modal interaction = signed PO inclusion–exclusion; report $I$ and $p(|I|)$.”

---

## C. Do **not** conflate with T×A / T×V combo power

| Object | Table | Is it signed attribution? |
|---|---|---|
| Combo detectability | concept-drift T×A / T×V / TAV power | **No** — coverage only |
| PO-risk proportion $\pi_m$ | leave-out $\Delta_m$ (mean_shift + PO exports) | direction via $\Delta_m$ |
| Interaction $I(m_1,m_2)$ | `interact_PO_*` (+ `crosscorr_*`) under `interaction_shift` | **Yes** — sign of $I$ |

Under concept drift, $X$ is fixed; classical TST ≈0 is expected. CF/RF/DRPerm
read $Y\mid X$. That supports the **proportion / coverage** story, not $I$.

---

## D. Delivery checklist (收数)

| Cell | Status |
|---|---|
| CF/DR/RF T×A, T×V concept power @ B=120 | **done** |
| CF/DR/RF TAV concept power @ B=120 | **done** |
| Classical T×A / T×V concept @ B=120 | **done** (power≈0) |
| Classical TAV concept | **stacking** (~58/120) |
| Null type-I CF/DR (T×A) | done (~53); SKK13/ZWL inflated |
| Null `interact_PO` / `crosscorr` on T×A, T×V, A×V | **done** @ ≥120 (≈α) |
| Null TAV with all three pair detectors | **stacking** (~23/120) |
| `interaction_shift` power (signed $I$) | **next board after null TAV** |

Artifacts: `POLISHED_CROSS_MODAL_TA_TV.md`, `latex/cross_modal_TA_TV.tex`,
`latex/cross_modal_interpretation.tex`, this note.
