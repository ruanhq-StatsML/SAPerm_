# Preliminary half-results (top30 focus)

_Updated 2026-11-21T00:49:03Z. Accelerated cf∥dr → B=120._

## `msrvtt_concept_top30` / `text_audio` / concept_drift
| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 1.000 | 0.007 | 150 |
| skk13 | 1.000 | 0.000 | 73 |
| zwl | 1.000 | 0.000 | 73 |
| **cfperm** | 0.675 | 0.226 | 80 |
| **drperm** | 0.566 | 0.183 | 76 |
| c2st_XY | 0.137 | 0.760 | 73 |
| bai96 | 0.000 | 1.000 | 76 |
| c2st_X | 0.000 | 1.000 | 73 |
| chen10 | 0.000 | 1.000 | 76 |
| kmmd | 0.000 | 1.000 | 76 |
| miles11 | 0.000 | 1.000 | 73 |
| mmd_rbf | 0.000 | 1.000 | 76 |
| pan14 | 0.000 | 1.000 | 76 |
| sri08 | 0.000 | 1.000 | 76 |
| xu16 | 0.000 | 1.000 | 73 |

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
- cfperm 79/120, drperm 75/120, rfperm 150 (done)
- MSR paused at AV~102 to free cores for concept
- ETA cf/dr →120: ~1–2h at current rate
