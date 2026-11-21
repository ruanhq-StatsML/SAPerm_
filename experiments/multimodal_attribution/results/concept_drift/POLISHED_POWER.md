# Concept-drift power board (polished) — 2026-11-21 07:34 UTC

Setting: `msrvtt_concept_top30`, PCA `top30`, $n=500$/batch, $\alpha=0.05$.
X fixed across domains; only $Y\mid X$ drifts (`concept_shift=0.35`).

## Power under concept drift

| method | T×A | T×V | T×A×V | B (T×A / T×V / T×A×V) |
|---|---:|---:|---:|---:|
| **rfperm** | 1.000 | 0.987 | 0.933 | 150 / 150 / 150 |
| **cfperm** | 0.600 | 0.342 | 0.400 | 120 / 120 / 120 |
| **drperm** | 0.400 | 0.217 | 0.300 | 120 / 120 / 120 |
| skk13 | 1.000 | 1.000 | 1.00$^\dagger$ | 120 / 120 / 10 |
| zwl | 1.000 | 1.000 | 1.00$^\dagger$ | 120 / 120 / 10 |
| c2st_XY | 0.083 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| c2st_X | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| miles11 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| xu16 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| kmmd | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| chen10 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| sri08 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| bai96 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| pan14 | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |
| mmd_rbf | 0.000 | 0.000 | 0.00$^\dagger$ | 120 / 120 / 10 |

$^\dagger$ preliminary ($B<30$).

## Type-I (null `msrvtt_null_top30`, text_audio)

| method | B | reject rate |
|---|---:|---:|
| **rfperm** | — | — |
| **cfperm** | 53 | 0.057 |
| **drperm** | 53 | 0.038 |
| skk13 | 53 | 1.000 |
| zwl | 53 | 0.132 |
| c2st_XY | 53 | 0.075 |
| c2st_X | 53 | 0.038 |
| miles11 | 53 | 0.019 |
| xu16 | 53 | 0.057 |
| kmmd | 53 | 0.057 |
| chen10 | 53 | 0.000 |
| sri08 | 53 | 0.019 |
| bai96 | 53 | 0.019 |
| pan14 | 53 | 0.038 |
| mmd_rbf | 53 | 0.019 |

## Dataset schema (standardized)

| field | values | role |
|---|---|---|
| `scenario` | `no_shift` / `mean_shift` / `interaction_shift` / `concept_drift` | DGP family |
| `band` | `top30` / `pc30_100` / `pc31_100` | PCA band per modality |
| `combo` | T×A / T×V / A×V / T×A×V | modality subset |
| `concept_shift` | scalar (board: 0.35) | strength of $Y\mid X$ drift |
| checkpoint | `(dataset, combo, scenario, method, rep)` | stackable $B$ |

## Readout

- **RFPerm** saturates near 1 on all finished combos (outcome-model sensitive to $Y\mid X$).
- **CFPerm / DRPerm** retain non-trivial power; T×A strongest among the three combos.
- Most classical two-sample tests ≈0 under concept drift (X unchanged) — as expected.
- **SKK13 / ZWL** hit power 1 but SKK13 also rejects under null → not calibrated here.
