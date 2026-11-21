---
description: How to attribute covariate vs concept drift to blocks and features
---

# Attribution playbook

## Pathway (default for covariate)

1. Split features into blocks.
2. For each block `b`, rewrite today's block from a reference neighbor conditioned on `X_{-b}`.
3. Contribution share = block distance / sum of block distances.
4. Report top block + share. This mirrors `tabular-drift-attribution`.

## LOCO PO-risk (concept)

1. Fit reference and local outcome models.
2. Drop one feature at a time; measure drop in PO-risk / delta_cf.
3. Rank features by risk drop. Sibling research: CFPerm loco, OOD_variable_importance.

## Decision table

| Shift type | Attribution mode | Next action |
| --- | --- | --- |
| none | skip | monitor |
| covariate | pathway | reweight / refresh drifted blocks |
| concept | loco | retrain concept; CFPerm drivers |
| mixed | pathway then loco | fix covariates, then residual concept |
| label | taxonomy first | check label pipeline / noise |

## Quality bars

- Attribution top-hit on planted block fixtures ≥ 0.7 before Eval Gate pass.
- Never claim a driver without a share or score.
