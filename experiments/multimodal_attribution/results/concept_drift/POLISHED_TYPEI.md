# Type-I error board (polished) — 2026-11-21 07:36 UTC

Null DGP: `msrvtt_null_top30` (no shift). Target nominal $\alpha=0.05$.
Cell = reject rate; prefer overnight board when $B$ is larger.

## Main type-I table

| method | T×A | T×V | A×V | B (T×A / T×V / A×V) |
|---|---:|---:|---:|---:|
| **rfperm** | 0.00† | — | — | 13 / — / — |
| **cfperm** | 0.057 | — | — | 53 / — / — |
| **drperm** | 0.038 | — | — | 53 / — / — |
| skk13 | 1.000‡ | 1.000‡ | 1.000‡ | 150 / 150 / 120 |
| zwl | 0.153‡ | 0.207‡ | 0.217‡ | 150 / 150 / 120 |
| c2st_XY | 0.033 | 0.053 | 0.025 | 150 / 150 / 120 |
| c2st_X | 0.040 | 0.033 | 0.050 | 150 / 150 / 120 |
| miles11 | 0.033 | 0.067 | 0.067 | 150 / 150 / 120 |
| xu16 | 0.033 | 0.033 | 0.025 | 150 / 150 / 120 |
| kmmd | 0.067 | 0.080 | 0.033 | 150 / 150 / 120 |
| chen10 | 0.000 | 0.000 | 0.000 | 150 / 150 / 120 |
| sri08 | 0.060 | 0.073 | 0.092 | 150 / 150 / 120 |
| bai96 | 0.060 | 0.073 | 0.092 | 150 / 150 / 120 |
| pan14 | 0.060 | 0.093 | 0.092 | 150 / 150 / 120 |
| mmd_rbf | 0.060 | 0.053 | 0.100‡ | 150 / 150 / 120 |

† preliminary ($B<30$).  ‡ inflated (reject rate $\ge 0.10$, above $\alpha=0.05$).

## Primary suite @ T×A (spotlight)

| method | B | type-I | source | vs α=0.05 |
|---|---:|---:|---|---|
| **rfperm** | 13 | 0.000 | overnight | OK (~nominal) |
| **cfperm** | 53 | 0.057 | concept | OK (~nominal) |
| **drperm** | 53 | 0.038 | concept | OK (~nominal) |

## Readout

- Most classical TST stay near $\alpha$ on T×A/T×V/A×V with $B\ge 120$.
- **SKK13** rejects ~100% under null → unusable as a calibrated level-α test here.
- **ZWL** is mildly–clearly inflated (~0.15–0.22).
- **CFPerm / DRPerm** (concept-board null, $B=53$) track $\alpha$ closely (0.057 / 0.038).
- RFPerm null still thin on this merge (fill from overnight when available).
