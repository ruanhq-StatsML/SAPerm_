# Synthetic MSR-VTT: concept-drift power ↔ type-I 口径对齐

**Unified caliber (delivery):**

| knob | value |
|---|---|
| `n_sample` | **500** |
| `B` / `n_rep` | **120** (RFPerm power may be 150; type-I target still ≥100) |
| `n_perm` | **150** for CF / DR / RF |
| `α` | **0.05** |
| DGP family | synthetic multimodal MSR-VTT-style embeddings |
| boards | `msrvtt_concept_*` (power) · `msrvtt_null_*` (type-I) |

Same generator (`DatasetSpec` / `make_dataset_batch`), same modality combos (T×A / T×V / TAV), same methods.
Only the **scenario** flips: `concept_drift` vs `no_shift`.

---

## Logic (must stay paired)

| Axis | Scenario | What moves | What we report |
|---|---|---|---|
| **Power** | `concept_drift` | $X$ fixed; $Y\mid X$ shifts | reject rate under alternative |
| **Type-I** | `no_shift` | nothing | reject rate under null |

**Interpretation lock:**

1. Classical TST power ≈ 0 under concept drift → **low covariate SNR** (expected; $P(X)$ unchanged).
2. CF/DR/RF graded power on the same cells → suite reads **$Y\mid X$**, not an $X$-artifact.
3. Type-I on `msrvtt_null_*` at the **same** $(n,B,n_{\mathrm{perm}},α)$ → those powers are not size inflation.
4. `interact_PO` type-I ≈ α on pairwise null → signed $I_{\mathrm{cal}}$ under interaction_shift is interpretable.

Do **not** mix overnight classical type-I at $n_{\mathrm{perm}}=120$ into CF/DR/RF claims without noting the perm budget; primary CF/DR/RF type-I is the concept-board null at $n_{\mathrm{perm}}=150$.

---

## Dataset split (keep separate tables)

| Dataset | band | Power status | Type-I status |
|---|---|---|---|
| **A** `msrvtt_concept_top30` | top30 | CF/DR/RF + classical **delivery** | RF delivery; CF/DR stacking →120; classical overnight OK for TST |
| **B** `msrvtt_concept_pc31_100` | pc31–100 | RF delivery; CF/DR/classical TBD | use shared `msrvtt_null_top30` RF until pc31 null filled |

---

## Live alignment checklist

| cell | target | status |
|---|---|---|
| Power A: CF/DR/RF × T×A/T×V/TAV | B≥120, n_perm=150 | **done** |
| Type-I: RF × T×A/T×V/TAV | B≥120, n_perm=150 | **done** (0.000) |
| Type-I: CF/DR × T×A | B→120, n_perm=150 | stacking (B≈89) |
| Type-I: CF/DR × T×V / TAV | B→120, n_perm=150 | **queued** (`null_align_topup`) |
| PO type-I pairwise | B≥120 | **done** (overnight; n_perm=120) |
| Power B: CF/DR/classical | B→120 | not yet |

Top-up command (same out-dir, checkpoint-safe):

```bash
python -m experiments.multimodal_attribution.concept_drift_overnight \
  --n 500 --n-rep 120 --n-perm 150 \
  --scenarios no_shift --datasets msrvtt_null_top30 \
  --methods cfperm drperm rfperm --n-jobs 2 \
  --out-dir experiments/multimodal_attribution/results/concept_drift
```
