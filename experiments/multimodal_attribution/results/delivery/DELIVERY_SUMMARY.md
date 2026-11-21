# Delivery-grade power + PO summary

_stamp 2026-11-21 11:26 UTC · 口径: n_sample=500, B≥100 (target 120), n_perm∈{120,150}, α=0.05_

## 1. Concept drift — CF / DR / RF

`msrvtt_concept_top30`

| method | T×A | T×V | TAV |
|---|---:|---:|---:|
| **rfperm** | 1.000 | 0.987 | 0.933 |
| **cfperm** | 0.600 | 0.342 | 0.400 |
| **drperm** | 0.400 | 0.217 | 0.300 |

Ordering (CF): **T×A > TAV > T×V**. RF saturates. TAV ≠ free lunch.

## 2. Type-I — interact_PO / crosscorr (delivery)

| pair | interact_PO | B | crosscorr | B |
|---|---:|---:|---:|---:|
| text×audio | 0.026 | 38 | 0.026 | 38 |
| text×video | 0.067 | 150 | 0.060 | 150 |
| audio×video | 0.079 | 38 | 0.105 | 38 |

All ≈ α — **PO interaction type-I calibrated**.

## 3. Type-I — CF/DR/RF null (live)

| board | combo | method | type-I | n_rep | grade |
|---|---|---|---:|---:|---|
| concept | text_audio | cfperm | 0.045 | 89 | preliminary |
| concept | text_audio | drperm | 0.056 | 89 | preliminary |
| concept | text_audio | rfperm | 0.000 | 120 | delivery |
| concept | text_audio_video | rfperm | 0.000 | 120 | delivery |
| concept | text_video | rfperm | 0.000 | 120 | delivery |
| overnight | text_audio | cfperm | 0.000 | 13 | preliminary |
| overnight | text_audio | drperm | 0.000 | 13 | preliminary |
| overnight | text_audio | rfperm | 0.000 | 13 | preliminary |
| expand | text | cfperm | 0.000 | 8 | preliminary |
| expand | text | drperm | 0.000 | 8 | preliminary |
| expand | text | rfperm | 0.000 | 8 | preliminary |

## 4. Sensitivity signed $I_{cal}$ (preliminary)

| dataset | combo | λ | mean $I$ | mean $I_{cal}$ | frac($I_{cal}<0$) | n |
|---|---|---:|---:|---:|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | -1.8 | 3.31 | -2.19 | 1.00 | 14 |

### CF/DR/RF on same cell

| dataset | combo | method | power | n_rep |
|---|---|---|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | cfperm | 0.71 | 14 |
| sens_interact_text_audio_neg1p8 | text_audio | drperm | 0.00 | 14 |
| sens_interact_text_audio_neg1p8 | text_audio | rfperm | 0.00 | 14 |

## 5. Expand board progress (type-I singles→TAV)

| combo | method | type-I | n_rep | grade |
|---|---|---:|---:|---|
| text | cfperm | 0.000 | 8 | preliminary |
| text | drperm | 0.000 | 8 | preliminary |
| text | rfperm | 0.000 | 8 | preliminary |

## 6. Read

- Concept: T×A strongest pairwise coverage for CF/DR; RF saturates; TAV not free lunch.
- interact_PO null ≈ α on all three pairs — mechanism detector size-correct.
- λ=−1.8 sensitivity: raw $I>0$ but $I_{cal}<0$ (antagonism); need B→120 for delivery.
- Expand: filling single-modality + TAV type-I at full cores.

