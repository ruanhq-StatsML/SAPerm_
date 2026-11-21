# Empirical interpretation: ± interaction ↔ power ↔ sentiment-label decisions

Unique-insight note for the sensitivity board (CF / DR / RF only).

---

## 0. What “sentiment label” means here

In the MSR-VTT-style / MOSEI-style pipeline, $Y$ is the **outcome / sentiment (or emotion) label**
the downstream head is trying to score; $T$ is the domain (ref vs new batch).
PO risk asks: after fitting on multimodal $X$, how much **domain-conditional label structure**
is left unexplained?

So $\pm I$ is not a cosmetic sign on a correlation — it is a statement about the
**decision mechanism** that maps modalities → label.

---

## 1. Calibrated sign (important empirical correction)

Early $\lambda=-1.8$ reps already show: **raw** $I=\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-both)$
is often $>0$ even when the plant is antagonistic, because leave-out PO has a
positive baseline. Use

\[
I_{\mathrm{cal}} \;=\; I - \mathrm{PO}(\mathrm{full})
\quad\text{(field } \texttt{interaction\_vs\_full}\text{)}.
\]

| $I_{\mathrm{cal}}$ | Mechanism reading for the label head |
|---|---|
| **$>0$ (正)** | Synergy: the pair contributes **joint** structure to $Y\mid X$ / domain gap beyond what the full model’s additive pieces explain. Cue **agreement / product** drives the sentiment decision. |
| **$<0$ (负)** | Antagonism / cancellation: modalities **conflict** for the label; including both without an interaction-aware head wastes or cancels signal. |
| $\approx 0$ | Approximately additive: label decision is a sum of single-modality effects (mean-shift world). |

Matched signed power uses $\mathrm{sign}(I_{\mathrm{cal}})$ against $\mathrm{sign}(\lambda)$.

---

## 2. How ±$I$ connects to CF / DR / RF **power**

List powers on the hierarchy, then read across:

| Combo | What power means for label detection |
|---|---|
| `power_a`, `power_b` | Can **one** modality alone expose the domain change in $Y\mid X$? |
| `power_pair` | Does **concatenating** the planted pair help the perm test see the label shift? |
| `power_tav` | Does a third modality help further, or only add noise? |
| Fusion lift `pair − max(single)` | Detectability twin of synergy |

### If $I_{\mathrm{cal}}>0$ (正向协同) — expected pattern

1. Label decision uses **cross-modal product** (e.g. text×audio agreement on sentiment).
2. Singles each see only a fragment → `power_a`, `power_b` moderate.
3. Pair concat restores the product features → **`power_pair > max(power_a, power_b)`**.
4. CF/DR/RF signed story: high matched signed power **and** positive fusion lift.
5. **Sentiment implication:** fusion (or an interaction head) is the right deployment;
   dropping one modality quietly changes the decision boundary.

### If $I_{\mathrm{cal}}<0$ (负向拮抗) — expected pattern

1. Modalities push the label in **opposing** product directions (e.g. positive text, negative prosody).
2. Singles may still show some power (each channel carries a marginal cue).
3. Naive concat **does not** reliably beat the better single → fusion lift ≤ 0.
4. Matched (neg) signed power high; unsigned $|I|$ power may still fire.
5. **Sentiment implication:** forced early fusion can **hurt** calibration; better to
   gate, late-fuse, or trust the dominant modality. Decision mechanism is conflict,
   not agreement.

### Control: ±mean_shift (no interaction)

- Only `power_single(shifted)` lights up; pair/TAV ≈ single.
- $I_{\mathrm{cal}}$ signed power ≈ α.
- Label shift is a **main-effect** channel move, not a fusion story.

---

## 3. “Detection vs decision mechanism” — the unique split

| Layer | Object | Question |
|---|---|---|
| **Detection** | CF/DR/RF power on combos | Can we *tell* the batch’s label law moved? |
| **Decision mechanism** | $\mathrm{sign}(I_{\mathrm{cal}})$ + PO leave-outs | *How* do modalities combine inside that label law? |
| **Dominant modality** | $\arg\max_m \mathrm{power}_m$ | Which *single* channel carries the detectable cue? |

Positive interaction without fusion lift = mechanism claim without detectability payoff (weak for delivery).
Fusion lift without matched sign = we detect something, but do not know if the label head is synergistic or conflicting.

**Delivery quantity:** cells where matched signed power and fusion lift **co-move** with $\mathrm{sign}(\lambda)$.

### Dominant modality (extra insight under antagonism)

Empirically for $\lambda=-1.8$ on text×audio (`n_sample=300`, `n_rep=24`):

| combo | CFPerm power |
|---|---:|
| text | **0.75** ← dominant |
| audio | 0.00 |
| text_audio | 0.54 |
| TAV | 0.62 |
| fusion lift | **−0.21** |

Reading: $I_{\mathrm{cal}}≪0$ says the label head is in **conflict**; the dominant-modality table says **trust text, do not early-fuse audio**.
TAV cannot rescue that — third channel ≠ fix for antagonism.
Under $\lambda>0$ we expect the opposite pattern: no single dominates as hard, fusion lift $>0$.

---

## 4. How we list & compare (iteration protocol)

Every refresh, dump for each `(pair, λ, method∈{CF,DR,RF})`:

```
power_a | power_b | power_pair | power_tav | lift | mean_I_cal | matched_signed_pow
```

Then argue:

1. Sort by $\lambda$: does lift and matched signed-pow flip together?
2. Within $\lambda>0$: which pair (T×A / T×V / A×V) has the largest lift — that pair
   is where sentiment decisions are most cross-modally coupled under this DGP.
3. TAV − pair: if ≈0 under pair plant, third modality is not in the label mechanism.

Live tables: `POWER_LIST.md`, `SENSITIVITY_BOARD.md`, `single_vs_pair_contrast.csv`.
