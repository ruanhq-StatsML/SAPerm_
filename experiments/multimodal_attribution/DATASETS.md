# Standardized multimodal board schema

Every overnight / concept-drift cell is a `DatasetSpec`:

| field | role |
|---|---|
| `name` | stable id, e.g. `msrvtt_concept_top30` |
| `scenario` | `no_shift` \| `mean_shift` \| `interaction_shift` \| `concept_drift` \| `covariate_backtrans` \| `covariate_inject` |
| `band` | PCA band: `top30` / `pc30_100` / `pc31_100` |
| `combo` | modality subset at eval time (from `MODALITY_COMBOS` / expand `FULL_COMBOS` incl. singles) |
| shift knobs | `mean_shift`, `interaction_shift`, `concept_shift`, `covariate_amount`, `inject_frac`, `shift_modality`, `interact_pair` |

**Concept drift (this board):** keep multimodal `X` identical across domains; only change `Y|X` via `apply_concept_drift` (`concept_shift=0.35`). Classical two-sample tests on `X` should be powerless; CF/RF/DRPerm target the outcome shift.

**Covariate back-translation:** warp one modality’s embedding space (default text) while keeping $Y$ semantics — multimodal analogue of `emb` vs `emb_backtrans` + shared rating (`reference/benchmark_translation.py`).

**Covariate inject:** overwrite a random fraction of a modality’s coordinates (random token / feature pollution).

**Expand board** (`expand_shift_board.py`): singles → pairs → TAV under type-I / concept / covariate / ±interaction. See `results/expand/EVALUATION_FRAMEWORK.md`.

**Checkpoint key:** `(dataset, combo, scenario, method, rep)` — raising `B` only fills missing reps.

Export polished tables:

```bash
python -m experiments.multimodal_attribution.export_polished_tables
```
