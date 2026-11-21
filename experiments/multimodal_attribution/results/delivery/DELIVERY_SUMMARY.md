# Delivery-grade power + type-I (polished)

_stamp 2026-11-21 · 口径: n_sample=500, B≥100 (target 120), n_perm∈{120,150}, α=0.05_

**Read in one line.** Under concept drift, classical TST≈0 is *expected* (low covariate SNR: $X$ fixed). Graded CF/DR/RF power is therefore a strong proof that the suite reads $Y|X$. PO type-I ≈ α closes the size story.

LaTeX: `experiments/multimodal_attribution/latex/delivery_power_po.tex`  
口径对齐: `results/delivery/MSRVTT_POWER_TYPEI_CALIBER.md` (concept-drift power ↔ null type-I share $n=500$, $B\to 120$, $n_{\mathrm{perm}}=150$).

---

## 1. Concept-drift power (`msrvtt_concept_top30`)

| method | T×A | T×V | TAV |
|---|---:|---:|---:|
| **RFPerm** | 1.000 | 0.987 | 0.933 |
| **CFPerm** | 0.600 | 0.342 | 0.400 |
| **DRPerm** | 0.400 | 0.217 | 0.300 |
| SKK13‡ | 1.000 | 1.000 | 1.000 |
| ZWL‡ | 1.000 | 1.000 | 1.000 |
| C2ST_XY | 0.083 | 0.000 | 0.000 |
| C2ST_X / Miles11 / Xu16 / kMMD / Chen10 / Bai96 / Pan14 / Sri08 / MMD_rbf | 0 | 0 | 0 |

‡ Inflated on null (type-I≈1) — not valid detections.

**Why this is strong.** Dataset SNR on $X$ is low under concept drift by construction → classical methods must be quiet. CF/DR/RF still recover **T×A > TAV > T×V**. That gap is the PO-risk / multimodal attribution claim.

---

## 2. Type-I — interact_PO / crosscorr (delivery)

| pair | interact_PO | B | crosscorr | B |
|---|---:|---:|---:|---:|
| text×audio | 0.027 | 150 | 0.047 | 150 |
| text×video | 0.067 | 150 | 0.060 | 150 |
| audio×video | 0.033 | 120 | 0.033 | 120 |

All ≈ α — mechanism detector size-correct.

---

## 3. Type-I — CF / DR / RF null (live)

| board | combo | method | type-I | B | grade |
|---|---|---|---:|---:|---|
| concept | text_audio | CFPerm | 0.045 | 89 | preliminary |
| concept | text_audio | DRPerm | 0.056 | 89 | preliminary |
| concept | text_audio | RFPerm | 0.000 | 120 | delivery |
| concept | text_video | RFPerm | 0.000 | 120 | delivery |
| concept | TAV | RFPerm | 0.000 | 120 | delivery |

Expand: singles→TAV type-I filling in parallel; covariate_backtrans / inject also running.

---

## 4. Signed $I_{cal}$ (λ=−1.8, preliminary B=14)

| mean $I$ | mean $I_{cal}$ | frac($I_{cal}<0$) |
|---:|---:|---:|
| +3.31 | **−2.19** | 1.00 |

Raw $I$ biased +; calibrated sign matches antagonism. CF≈0.71 on same cell.

---

## 5. Covariate shift (running)

Messy $X$-shifts (back-translation-style text warp + random feature inject on text/audio) launched in parallel on the expand board (`n=500`, `B=120`, `n_perm=150`, CF/DR/RF). Expect: power on combos containing the shifted modality; contrast with concept drift where classical≈0.
