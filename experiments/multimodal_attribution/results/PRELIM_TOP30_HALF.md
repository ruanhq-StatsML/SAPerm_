# Preliminary half-results (top30 focus)

_For noon summarize. Target B=120; numbers below are live partial._

## Concept drift vs null — `msrvtt_concept_top30` / `text_audio`

### concept_drift
| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 1.000 | 0.007 | 150 |
| skk13 | 1.000 | 0.000 | 66 |
| zwl | 1.000 | 0.000 | 66 |
| **cfperm** | 0.712 | 0.204 | 66 |
| **drperm** | 0.606 | 0.181 | 66 |
| c2st_XY | 0.152 | 0.734 | 66 |
| bai96 | 0.000 | 1.000 | 66 |
| c2st_X | 0.000 | 1.000 | 66 |
| chen10 | 0.000 | 1.000 | 66 |
| kmmd | 0.000 | 1.000 | 66 |
| miles11 | 0.000 | 1.000 | 66 |
| mmd_rbf | 0.000 | 1.000 | 66 |
| pan14 | 0.000 | 1.000 | 66 |
| sri08 | 0.000 | 1.000 | 66 |
| xu16 | 0.000 | 1.000 | 66 |

### no_shift (type-I)
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

## Takeaway (符合预期)
- Under concept drift: **rfperm≈1.0**, **cfperm/drperm ≫ classical TST** (most classical power≈0).
- Under null: cf/dr type-I ≈α; skk13 still over-rejects.
- Continuing to B=120 on top30; MSR AV nearly done then cores → concept.
