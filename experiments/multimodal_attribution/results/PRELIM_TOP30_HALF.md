# Preliminary half-results (top30 focus)

_Updated 2026-11-21T00:22:49Z. Target B=120._

## `msrvtt_concept_top30` / `text_audio`

### concept_drift
| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 1.000 | 0.007 | 150 |
| skk13 | 1.000 | 0.000 | 71 |
| zwl | 1.000 | 0.000 | 71 |
| **cfperm** | 0.704 | 0.207 | 71 |
| **drperm** | 0.577 | 0.176 | 71 |
| c2st_XY | 0.141 | 0.753 | 71 |
| bai96 | 0.000 | 1.000 | 71 |
| c2st_X | 0.000 | 1.000 | 71 |
| chen10 | 0.000 | 1.000 | 71 |
| kmmd | 0.000 | 1.000 | 71 |
| miles11 | 0.000 | 1.000 | 71 |
| mmd_rbf | 0.000 | 1.000 | 71 |
| pan14 | 0.000 | 1.000 | 71 |
| sri08 | 0.000 | 1.000 | 71 |
| xu16 | 0.000 | 1.000 | 71 |

### no_shift
| method | power | mean p | n_rep |
|---|---:|---:|---:|
| skk13 | 1.000 | 0.000 | 53 |
| zwl | 0.132 | 0.432 | 53 |
| c2st_XY | 0.075 | 0.533 | 53 |
| **cfperm** | 0.057 | 0.513 | 53 |
| kmmd | 0.057 | 0.598 | 53 |
| xu16 | 0.057 | 0.564 | 53 |
| c2st_X | 0.038 | 0.540 | 53 |
| **drperm** | 0.038 | 0.509 | 53 |
| pan14 | 0.038 | 0.550 | 53 |
| bai96 | 0.019 | 0.543 | 53 |
| miles11 | 0.019 | 0.503 | 53 |
| mmd_rbf | 0.019 | 0.570 | 53 |
| sri08 | 0.019 | 0.552 | 53 |
| chen10 | 0.000 | 0.887 | 53 |

## Status
- Concept TA ≈ 70/120
- MSR AV ≈ 103/120 → then cores to concept n_jobs=4
- Takeaway: rfperm=1.0; cf/dr ≫ classical under drift; null type-I OK for cf/dr.
