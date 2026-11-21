# PO-risk 2nd/3rd-order (RF outcome) + LOMO power cross-check

Outcome / τ / ν models: `rf_regressor`.

Definitions (leave-out PO risk $R(S)$ = held-out PO after dropping block set $S$):

- $I^{(2)}(m_1,m_2)=R(\{m_1\})+R(\{m_2\})-R(\{m_1,m_2\})$; $I_{\mathrm{cal}}^{(2)}=I^{(2)}-\mathrm{PO}(\mathrm{full})$.
- $I^{(3)}=R(\{t\})+R(\{a\})+R(\{v\})-R(\{t,a\})-R(\{t,v\})-R(\{a,v\})+R(\{t,a,v\})$.
- LOMO power loss $=\mathrm{Power}(\mathrm{TAV})-\mathrm{Power}(\mathrm{TAV}\setminus\{m\})$.
  Loss $>0$: dropped modality helps the triple. Loss $<0$: over-complete / conflicting (dropping raises power) — cross-check for $I^{(3)}<0$.

## 1. Pairwise $I^{(2)}$ / $I^{(2)}_{\mathrm{cal}}$ (RF)

_No I2 rows yet._

## 2. Triple $I^{(3)}$ (RF)

_No I3 rows yet._

## 3. LOMO power loss (cross-check)

| dataset | method | dropped | Power(TAV) | Power(LOO) | loss (TAV−LOO) | Δ% | B_TAV | B_LOO |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `msrvtt_concept_pc31_100` | rfperm | audio | 0.573 | 0.693 | -0.120 | -20.9% | 150 | 150 |
| `msrvtt_concept_pc31_100` | rfperm | video | 0.573 | 0.647 | -0.073 | -12.8% | 150 | 150 |
| `msrvtt_concept_pc31_100` | rfperm | vs_best_pair | 0.573 | 0.693 | -0.120 | -20.9% | 150 | 150 |
| `msrvtt_concept_top30` | cfperm | audio | 0.400 | 0.342 | +0.058 | +14.6% | 120 | 120 |
| `msrvtt_concept_top30` | cfperm | video | 0.400 | 0.600 | -0.200 | -50.0% | 120 | 120 |
| `msrvtt_concept_top30` | cfperm | vs_best_pair | 0.400 | 0.600 | -0.200 | -50.0% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | audio | 0.300 | 0.217 | +0.083 | +27.8% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | video | 0.300 | 0.400 | -0.100 | -33.3% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | vs_best_pair | 0.300 | 0.400 | -0.100 | -33.3% | 120 | 120 |
| `msrvtt_concept_top30` | rfperm | audio | 0.933 | 0.987 | -0.053 | -5.7% | 150 | 150 |
| `msrvtt_concept_top30` | rfperm | video | 0.933 | 1.000 | -0.067 | -7.1% | 150 | 150 |
| `msrvtt_concept_top30` | rfperm | vs_best_pair | 0.933 | 1.000 | -0.067 | -7.1% | 150 | 150 |
| `msrvtt_null_top30` | rfperm | audio | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 120 |
| `msrvtt_null_top30` | rfperm | video | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 133 |
| `msrvtt_null_top30` | rfperm | vs_best_pair | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 133 |

**Read rule.** If TAV loses to best pair (`dropped=vs_best_pair`, loss $<0$), triple fusion is inefficient — consistent with $I^{(3)}\le 0$ / over-complete. Per-modality loss $<0$ means that channel is antagonistic or pure noise in the concat.
