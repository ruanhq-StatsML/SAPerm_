# Power 口径（delivery-grade）

| knob | delivery value | notes |
|---|---|---|
| **n / n_sample** | **500** | per domain |
| **B / n_rep** | **120** | Monte Carlo size for power |
| **n_perm** | **150** (concept / sensitivity CF–DR–RF) or **120** (overnight classical+PO board) | both acceptable; do not mix within one board |
| **α** | 0.05 | — |

## Rule

- Cite a power number only if that cell has **B ≥ 100** (target **120**).
- If a conclusion was drawn at **B ≪ 120** (e.g. pilot B=24, null CF at B=13/53), **stack to B=120 and recompute** before delivery.
- Pilot `results/sensitivity_b24_nperm32/` (n=300, B=24, n_perm=32) is **exploratory only** — qualitative insights OK, not final powers.

## Live status

| board | n_perm | B target | action |
|---|---|---|---|
| concept_drift (CF/DR/RF power) | 150 | 120 | main cells OK; null CF/DR stacking 53→120 |
| overnight (classical + interact_PO) | 120 | 120 | null CF/DR/RF stacking 13→120; TAV null stacking |
| sensitivity (CF/DR/RF ±λ) | **150** | **120** | relaunched; replaces B=24 pilot |
