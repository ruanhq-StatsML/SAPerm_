# How to characterize & evaluate multimodal interaction × modality experiments

口径 throughout: **n_sample=500**, **B = n_rep = 120**, **n_perm = 150**, α=0.05 ·
methods **CFPerm / DRPerm / RFPerm** (+ `interact_PO` as mechanism readout).

This note maps the expand board to the back-translation / random-injection logic
in `benchmark_translation.py` / `trt_back_translation_benchmark_aug.py`, and
says what each axis *proves*.

---

## 1. Two questions that must not be conflated

| Layer | Object | Question |
|---|---|---|
| **Detection** | CF / DR / RF power on a modality *combo* | Can this channel (or concat) see the domain change? |
| **Mechanism** | calibrated PO interaction $I_{\mathrm{cal}}=I-\mathrm{PO}(\mathrm{full})$ | Is the label/domain gap **additive**, **synergistic**, or **antagonistic** across modalities? |

Combo power alone ≠ interaction. Interaction alone ≠ deployment rule.
The expand board records **both**, on the same DGP grid.

---

## 2. Shift taxonomy (what we plant)

| Scenario | What moves | Text-benchmark analogue | Expect |
|---|---|---|---|
| `no_shift` | nothing | null / type-I | rate ≈ α on **text, audio, video, pairs, TAV** |
| `concept_drift` | only $Y\mid X$ ($X$ fixed) | label / rating rule change with same embeddings | classical TST≈0; CF/DR/RF powered; PO quiet |
| `covariate_backtrans` | warp **text** embedding space; keep $Y$ semantics | `emb` vs `emb_backtrans` + shared `rating` | power high on combos containing text; PO ≈ null (no planted product) |
| `covariate_inject` | random overwrite of a modality fraction | random token / feature injection | classical + CF see $X$-shift; dominant = injected mod; PO ≈ null |
| `mean_shift` | additive block mean on one mod | simple covariate mean gap | `power_single(shifted)` ≈ pair/TAV |
| `interaction_shift` ±λ | planted cross-modal product + $Y$ cross term | *no* pure text analogue — multimodal unique | sign($I_{\mathrm{cal}}$)=sign(λ); fusion lift tracks sign |

Back-translation board (uploaded): PCA(`emb`) vs PCA(`emb_backtrans`), same response column, B bootstrap, n_rows=500.
Our `covariate_backtrans` is the **multimodal lift** of that logic: only the text block is rotated+noised; audio/video stay; outcome semantics preserved.

---

## 3. Modality ladder (how we evaluate)

For every scenario, report CF/DR/RF on:

```
text | audio | video | T×A | T×V | A×V | TAV
```

| Contrast | Formula | Insight |
|---|---|---|
| Dominant modality | $\arg\max_m \mathrm{Power}_m$ | which channel carries the shift |
| Fusion lift (pair) | $\mathrm{Power}_{ab}-\max(\mathrm{Power}_a,\mathrm{Power}_b)$ | does concat help? |
| TAV discipline | $\mathrm{Power}_{\mathrm{TAV}}-\mathrm{Power}_{\mathrm{best\ pair}}$ | is the third channel free lunch? |
| Type-I ladder | rate under `no_shift` for each rung | no inflation when stacking modalities |

**Decision rule (empirical):**

- $I_{\mathrm{cal}}>0$ **and** fusion lift $>0$ → early fusion / interaction head justified  
- $I_{\mathrm{cal}}<0$ **or** fusion lift $<0$ → prefer dominant single (or late fusion / gating)  
- covariate_backtrans / inject with PO≈α → mechanism is **modality-specific covariate**, not cross-modal synergy

---

## 4. Type-I: why singles + TAV matter

Pairwise `interact_PO` already calibrates ≈α (delivery). Missing piece was:

1. **Single-modality** CF/DR/RF under null — baseline false-positive when only one channel is tested  
2. **TAV concat** CF/DR/RF under null — does stacking three blocks inflate type-I?

If singles ≈α and TAV ≈α, the multimodal test is **size-correct** across the ladder; then power differences under concept/covariate/interaction are attributable to signal, not dimension inflation.

---

## 5. What “full experiment” means for delivery

| Block | Datasets | Combos | Status target |
|---|---|---|---|
| Type-I | `msrvtt_null_top30` | all 7 | B≥100 |
| Concept | `msrvtt_concept_top30` | all 7 (esp. singles new) | B≥100 |
| Covariate | backtrans_text, inject_text, inject_audio | combos containing shifted mod | B≥100 |
| Interaction | ±λ on T×A / T×V | pairs + TAV + singles of pair | B≥100 + signed $I_{\mathrm{cal}}$ |

Artifacts: `results/expand/power_raw.csv`, `EXPAND_BOARD.md`, this file.

---

## 6. One-sentence claim structure

> Under concept drift, combo power ranks **coverage** of $Y\mid X$ change; under back-translation / inject, power ranks **which modality’s $X$ moved**; under ±interaction_shift, **$I_{\mathrm{cal}}$ + fusion lift** jointly decide whether multimodal fusion helps the sentiment/domain decision — and type-I on singles→TAV shows that ranking is not an artifact of stacking channels.
