# Argument: `interaction_shift` ↔ signed PO interaction ↔ signed power

Goal: show that the overnight DGP `interaction_shift`, the estimator
$I(m_1,m_2)$, and the reported **signed power** are the *same* object —
not three unrelated numbers.

---

## 1. Definition (estimator = inclusion–exclusion on PO risk)

Fix a pair of modality blocks $(m_1,m_2)$. Pseudo-outcome risk after
block leave-out:

\[
I(m_1,m_2)
\;=\;
\mathrm{PO}(-m_1)+\mathrm{PO}(-m_2)-\mathrm{PO}(-m_1,-m_2).
\]

This is exactly `higher_order_interaction_PO_risk` → `"interaction"`.

**Why this is the interaction, not a main effect**

Rewrite as a discrete derivative:

\[
I
\;=\;
\bigl(\mathrm{PO}(-m_1)-\mathrm{PO}(-m_1,-m_2)\bigr)
\;-\;
\bigl(\mathrm{PO}(\mathrm{full}\text{ with }m_2\text{ only via }{-}m_1\text{ algebra})\bigr)
\]

More cleanly: if effects were **additive**,
$\mathrm{PO}(-m_1,-m_2)-\mathrm{PO}(\mathrm{full})
=\bigl(\mathrm{PO}(-m_1)-\mathrm{PO}(\mathrm{full})\bigr)
+\bigl(\mathrm{PO}(-m_2)-\mathrm{PO}(\mathrm{full})\bigr)$,
which rearranges to $I = \mathrm{PO}(\mathrm{full})$ under a pure-additivity
bookkeeping — the **excess** over additivity is what $I-\mathrm{PO}(\mathrm{full})$
(field `interaction_vs_full`) isolates. Empirically we report raw $I$ and its sign:

| Sign | Meaning under domain $T$ |
|---|---|
| $I>0$ | synergistic / amplifying cross-modal structure |
| $I<0$ | antagonistic / cancelling |
| $I\approx 0$ (vs null) | no second-order structure beyond noise |

Inference: permute $T$, reject on $|I|$ → **power of detecting an interaction**.
The **sign** for the narrative is $\mathrm{sign}(\hat I)$ on the unpermuted batch
→ **signed power** = fraction of reps with $p(|I|)<\alpha$ **and** $\hat I$ on the
hypothesized side (for our DGP: positive).

---

## 2. DGP (`interaction_shift`) deliberately plants a *product* structure

In `overnight_power._apply_pair_interaction` + `make_dataset_batch`:

1. **$X$-side:** shared shock $s$ into pair $(a,b)$  
   $a \leftarrow a + \lambda s$,\; $b \leftarrow b + \lambda\, s\cdot a_{:1}$  
   → dependence between blocks is **not** a mean shift of one modality alone.

2. **$Y$-side:**  
   $Y_{\mathrm{new}} \leftarrow Y + \lambda\,(a_0-\bar a_0)(b_0-\bar b_0)$  
   → the outcome itself carries a **multiplicative** cross term on the same pair.

Contrast:

| Scenario | What moves | Expected $I(a,b)$ | Expected $\Delta_m$ |
|---|---|---|---|
| `no_shift` | nothing | $\approx 0$ under null (type-I≈α) | ≈0 |
| `mean_shift` (e.g. video $+c$) | one block mean | $I\approx 0$ (additive) | $\Delta_{\mathrm{video}}$ large |
| `interaction_shift` on $(a,b)$ | product in $X$ and $Y$ | **$I(a,b)>0$** (synergy) | both $\Delta_a,\Delta_b$ can rise, but $I$ is the distinctive signal |

So the DGP’s ground-truth “interaction” and the estimator’s $I$ are matched by construction:
**the thing we inject is a second-order cross term; the thing we test is the second-order leave-out residual.**

---

## 3. Signed power = power against this matched alternative

Let $H_0$: no domain difference in the PO interaction ($T\perp$ structure).  
Let $H_1^{+}$: `interaction_shift` on pair $(a,b)$ with $\lambda>0$ (synergistic product).

- **Unsigned power:** $\mathbb{P}_{H_1}(p(|I|)<\alpha)$ — does `interact_PO_{a}_{b}` fire?
- **Signed power:** $\mathbb{P}_{H_1}\bigl(p(|I|)<\alpha \;\wedge\; \hat I>0\bigr)$ — fires **and** agrees with the planted sign.

Under our $\lambda>0$ product injection, $H_1^{+}$ predicts $\mathrm{sign}(I)=+$.
A negative $\hat I$ with small $p$ would be a *sign error* (still a detection, wrong direction).

Companion: `crosscorr_{a}_{b}` tests $|\mathrm{corr}_{\mathrm{new}}-\mathrm{corr}_{\mathrm{ref}}|$
on block means — pure $X$-dependence change from step (1), no $Y$. Together:

- `crosscorr` → “modalities became coupled in $X$”
- `interact_PO` + $\mathrm{sign}(I)$ → “that coupling (and/or the $Y$ product) is a **signed** second-order PO effect”

---

## 4. Why this is “很直观”

One sentence each:

1. **Plant** a text×video (say) product in the new domain.  
2. **Measure** whether leave-out PO risk shows a non-additive residual $I$.  
3. **Sign** tells synergy vs antagonism; **$p(|I|)$** tells detectability.  
4. **Signed power** under `interaction_shift` is the Monte Carlo rate of (2)+(3) agreeing with the plant.

Modality-specific **PO-risk proportion** $\pi_m\propto\Delta_m=\mathrm{PO}(-m)-\mathrm{PO}(\mathrm{full})$
answers a *different* question (main-effect share) and is what `mean_shift` isolates.
Interaction is what remains after those main effects are accounted for via inclusion–exclusion.

---

## 5. Delivery mapping

| Report column | Source |
|---|---|
| type-I of `interact_PO_*` | overnight `no_shift` (T×A / T×V / A×V done; TAV stacking) |
| signed power of `interact_PO_{a}_{b}` | `msrvtt_interact_{a}_{b}_top30` under `interaction_shift` |
| sign summary | mean / fraction of $\hat I>0$ on those reps |
| specificity check | same detector on a **non-planted** pair should stay near α |

Small-N sanity CSV (when present): `signed_interaction_demo.csv`.
Full B=120 board: next stack after null TAV finishes.
