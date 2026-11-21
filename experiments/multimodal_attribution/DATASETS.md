# Standardized multimodal board schema

Every overnight / concept-drift cell is a `DatasetSpec`:

| field | role |
|---|---|
| `name` | stable id, e.g. `msrvtt_concept_top30` |
| `scenario` | `no_shift` \| `mean_shift` \| `interaction_shift` \| `concept_drift` |
| `band` | PCA band: `top30` / `pc30_100` / `pc31_100` |
| `combo` | modality subset at eval time (not in spec; from `MODALITY_COMBOS`) |
| shift knobs | `mean_shift`, `interaction_shift`, `concept_shift`, `shift_modality`, `interact_pair` |

**Concept drift (this board):** keep multimodal `X` identical across domains; only change `Y|X` via `apply_concept_drift` (`concept_shift=0.35`). Classical two-sample tests on `X` should be powerless; CF/RF/DRPerm target the outcome shift.

**Checkpoint key:** `(dataset, combo, scenario, method, rep)` — raising `B` only fills missing reps.

Export polished tables:

```bash
python -m experiments.multimodal_attribution.export_polished_tables
```
