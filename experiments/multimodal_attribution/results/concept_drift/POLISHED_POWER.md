# Concept-drift power board (polished) — 2026-11-21 07:36 UTC

Setting: `msrvtt_concept_top30`, PCA `top30`, $n=500$/batch, $\alpha=0.05$.
X fixed across domains; only $Y\mid X$ drifts (`concept_shift=0.35`).

## Power under concept drift

| method | T×A | T×V | T×A×V | B (T×A / T×V / T×A×V) |
|---|---:|---:|---:|---:|
| **rfperm** | 1.000 | 0.987 | 0.933 | 150 / 150 / 150 |
| **cfperm** | 0.600 | 0.342 | 0.400 | 120 / 120 / 120 |
| **drperm** | 0.400 | 0.217 | 0.300 | 120 / 120 / 120 |
| skk13 | 1.000 | 1.000 | 1.00† | 120 / 120 / 12 |
| zwl | 1.000 | 1.000 | 1.00† | 120 / 120 / 12 |
| c2st_XY | 0.083 | 0.000 | 0.00† | 120 / 120 / 12 |
| c2st_X | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| miles11 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| xu16 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| kmmd | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| chen10 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| sri08 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| bai96 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| pan14 | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |
| mmd_rbf | 0.000 | 0.000 | 0.00† | 120 / 120 / 12 |

† preliminary ($B<30$).

## Type-I (see also `POLISHED_TYPEI.md`)

| method | T×A | T×V | A×V |
|---|---:|---:|---:|
| **rfperm** | 0.00† | — | — |
| **cfperm** | 0.057 | — | — |
| **drperm** | 0.038 | — | — |
| skk13 | 1.000‡ | 1.000‡ | 1.000‡ |
| zwl | 0.153‡ | 0.207‡ | 0.217‡ |
| c2st_XY | 0.033 | 0.053 | 0.025 |
| c2st_X | 0.040 | 0.033 | 0.050 |
| miles11 | 0.033 | 0.067 | 0.067 |
| xu16 | 0.033 | 0.033 | 0.025 |
| kmmd | 0.067 | 0.080 | 0.033 |
| chen10 | 0.000 | 0.000 | 0.000 |
| sri08 | 0.060 | 0.073 | 0.092 |
| bai96 | 0.060 | 0.073 | 0.092 |
| pan14 | 0.060 | 0.093 | 0.092 |
| mmd_rbf | 0.060 | 0.053 | 0.100‡ |

† $B<30$.  ‡ reject rate $\ge 0.10$ (inflated vs $\alpha=0.05$).

## Dataset schema (standardized)

| field | values | role |
|---|---|---|
| `scenario` | `no_shift` / `mean_shift` / `interaction_shift` / `concept_drift` | DGP family |
| `band` | `top30` / `pc30_100` / `pc31_100` | PCA band per modality |
| `combo` | T×A / T×V / A×V / T×A×V | modality subset |
| `concept_shift` | scalar (board: 0.35) | strength of $Y\mid X$ drift |
| checkpoint | `(dataset, combo, scenario, method, rep)` | stackable $B$ |
