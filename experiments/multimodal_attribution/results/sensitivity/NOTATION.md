# Notation / 口径对齐

| Symbol | Meaning | Sensitivity run default |
|---|---|---|
| **n** / **n_sample** | Observations **per domain** (ref batch and new batch each have size $n$) | 300 |
| **n_rep** / **B** | Independent Monte Carlo replications used to estimate **power** | 24 (target) |
| **n_perm** | Label / domain permutations inside one CF/DR/RF (or PO) test | 32 (PO: 12) |
| **α** | Rejection level | 0.05 |
| **power** | $\widehat{\mathbb{P}}(p<\alpha)$ over `n_rep` | — |
| **$I$** | raw $\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)$ | often $+$-biased |
| **$I_{cal}$** | $I-\mathrm{PO}(\mathrm{full})$ | signed mechanism |
| **fusion lift** | `power_pair − max(power_a, power_b)` | — |
| **dominant modality** | $\arg\max_{m\in\{a,b\}}$ `power_m` | — |

**Do not** write bare `n` in power tables for the replication count — that was ambiguous.
Use **`n_rep`** for Monte Carlo size and **`n_sample`** (or header `n=…`) for sample size.
