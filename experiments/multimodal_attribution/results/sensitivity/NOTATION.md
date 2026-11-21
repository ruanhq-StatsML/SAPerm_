# Notation / 口径（aligned to main boards）

| Symbol | Meaning | This sensitivity run |
|---|---|---|
| **n / n_sample** | samples **per domain** | **500** |
| **n_rep / B** | Monte Carlo reps for power | **120** |
| **n_perm** | permutations inside CF/DR/RF | **150** |
| **n_perm_po** | permutations for signed PO \|I\| | 50 |
| **α** | reject level | 0.05 |
| **power** | fraction of B reps with p<α | — |

**Delivery rule:** do not treat a cell as final unless **B ≥ 100** (target 120).
Pilot `sensitivity_b24_nperm32` (n=300,B=24,n_perm=32) is exploratory only.
