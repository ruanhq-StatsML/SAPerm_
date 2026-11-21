# Delivery-grade power summary

_stamp 2026-11-21 09:47 UTC · 口径: n_sample=500, B≥100 (target 120), n_perm∈{120,150}, α=0.05_

## 1. Concept drift — CF / DR / RF by modality combo

`msrvtt_concept_top30`, n_perm=150, B=120 (RF B=150).

| method | T×A | T×V | TAV |
|---|---:|---:|---:|
| **RFPerm** | 1.000 | 0.987 | 0.933 |
| **CFPerm** | 0.600 | 0.342 | 0.400 |
| **DRPerm** | 0.400 | 0.217 | 0.300 |

### Read
- **T×A > T×V > (roughly) for CF/DR**: text+audio carries more usable concept-drift signal than text+video.
- **RF saturates** all three combos on top30.
- **TAV** does not dominate T×A for CF/DR (0.40 / 0.30 vs T×A 0.60 / 0.40) — third modality is not free lunch under this DGP.

## 2. Concept drift — classical TST (same board, B=120)

| method | T×A | T×V | TAV |
|---|---:|---:|---:|
| miles11 | 0.000 | 0.000 | 0.000 |
| xu16 | 0.000 | 0.000 | 0.000 |
| kmmd | 0.000 | 0.000 | 0.000 |
| chen10 | 0.000 | 0.000 | 0.000 |
| skk13 | 1.000 | 1.000 | 1.000 |
| sri08 | 0.000 | 0.000 | 0.000 |
| bai96 | 0.000 | 0.000 | 0.000 |
| zwl | 1.000 | 1.000 | 1.000 |
| pan14 | 0.000 | 0.000 | 0.000 |
| c2st_X | 0.000 | 0.000 | 0.000 |
| c2st_XY | 0.083 | 0.000 | 0.000 |
| mmd_rbf | 0.000 | 0.000 | 0.000 |

Classical ≈0 except SKK13/ZWL (=1, inflated). Expected: $X$ fixed under concept drift.

## 3. Type-I — interact_PO / crosscorr (overnight null, delivery B)

| pair | interact_PO | n_rep | crosscorr | n_rep |
|---|---:|---:|---:|---:|
| text×audio | 0.027 | 150 | 0.047 | 150 |
| text×video | 0.067 | 150 | 0.060 | 150 |
| audio×video | 0.033 | 120 | 0.033 | 120 |

All ≈α — **PO interaction type-I calibrated** on pairwise nulls.

## 4. Type-I — CF/DR/RF null (thin → stacking)

| board | combo | method | power | n_rep | grade |
|---|---|---|---:|---:|---|
| concept | text_audio | cfperm | 0.056 | 54 | THIN→stack |
| concept | text_audio | drperm | 0.037 | 54 | THIN→stack |
| overnight | text_audio | cfperm | 0.000 | 13 | THIN→stack |
| overnight | text_audio | drperm | 0.000 | 13 | THIN→stack |
| overnight | text_audio | rfperm | 0.000 | 13 | THIN→stack |

## 5. Not yet delivery-grade

| item | status |
|---|---|
| Sensitivity ±λ / dominant-modality / fusion lift | relaunched n=500,B=120,n_perm=150 — filling |
| Overnight null CF/DR/RF | B≈13 — stacking to 120 |
| Concept null CF/DR | B≈54 — stacking to 120 |
| interact_PO under `interaction_shift` (signed power) | awaiting sensitivity B=120 |
| Overnight null TAV PO | thin |

## 6. Multimodal hierarchy (delivery concept powers, CFPerm)

```
text      — (single not on concept board)
audio     —
video     —
T×A       0.600
T×V       0.342
TAV       0.400
```
CFPerm ordering under concept drift: **T×A > TAV > T×V**.
