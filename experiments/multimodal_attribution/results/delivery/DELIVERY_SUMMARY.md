# Delivery-grade power + PO summary

_stamp 2026-11-21 11:28 UTC · 口径: n_sample=500, B≥100 (target 120), n_perm∈{120,150}, α=0.05_

## 1. Concept drift — CF / DR / RF (`msrvtt_concept_top30`)

| method | T×A | T×V | TAV |
|---|---:|---:|---:|
| **rfperm** | 1.000 | 0.987 | 0.933 |
| **cfperm** | 0.600 | 0.342 | 0.400 |
| **drperm** | 0.400 | 0.217 | 0.300 |

Ordering (CF): **T×A > TAV > T×V**. RF saturates. TAV ≠ free lunch.

## 2. Type-I — interact_PO / crosscorr (pairwise null, delivery)

| pair | interact_PO | B | crosscorr | B |
|---|---:|---:|---:|---:|
| text×audio | 0.027 | 150 | 0.047 | 150 |
| text×video | 0.067 | 150 | 0.060 | 150 |
| audio×video | 0.033 | 120 | 0.033 | 120 |

All ≈ α — **PO interaction type-I calibrated**.

## 3. Type-I — CF/DR/RF null (live)

| board | combo | method | type-I | n_rep | grade |
|---|---|---|---:|---:|---|
| concept | text_audio | cfperm | 0.045 | 89 | preliminary |
| concept | text_audio | drperm | 0.056 | 89 | preliminary |
| concept | text_audio | rfperm | 0.000 | 120 | delivery |
| concept | text_audio_video | rfperm | 0.000 | 120 | delivery |
| concept | text_video | rfperm | 0.000 | 120 | delivery |
| expand | text | cfperm | 0.000 | 10 | preliminary |
| expand | text | drperm | 0.000 | 10 | preliminary |
| expand | text | rfperm | 0.000 | 10 | preliminary |

## 4. Sensitivity signed $I_{cal}$ (preliminary)

| cell | λ | mean $I$ | mean $I_{cal}$ | frac($I_{cal}<0$) | n |
|---|---:|---:|---:|---:|---:|
| text_audio | −1.8 | 3.31 | -2.19 | 1.00 | 14 |

CF/DR/RF power: cf=0.71, dr=0.00, rf=0.00 (B=14)

## 5. Read

- Concept: T×A > TAV > T×V for CF/DR; RF≈1.
- interact_PO null ≈ α (T×A 0.027/150, T×V 0.067/150, A×V 0.033/120).
- λ=−1.8: $I_{cal}<0$ always so far (antagonism); CF≈0.71 preliminary.
- Expand @ **n_jobs=4** (all 4 cores) filling singles→TAV type-I.

