---
description: Offline/online eval loop — Dataset, Rubric, Evaluator, Gate, Regression
---

# Evaluation loop

Map the Agent Evaluation framework onto this attribution agent:

1. **Dataset** — fixtures (`null_shift` … `label_noise`) + promoted badcases.
2. **Rubric** — Type-I ≤ 0.1; covariate power ≥ 0.7; concept power ≥ 0.6; attribution top-hit ≥ 0.7; type accuracy ≥ 0.6.
3. **Evaluator** — `run_fixture_suite` (deterministic stats) + optional LLM-as-Judge on report wording.
4. **Human Evaluation** — taxonomize disagreements; promote high-severity misses.
5. **Regression Evaluation** — re-run suite after every method/prompt change; `eval_gate` must pass.
6. **Eval Gate** — hard stop before shipping architecture / model / threshold changes.

## Quality dimensions

| Dimension | How we measure it here |
| --- | --- |
| Task completion | Suite finishes; report has verdict + drivers |
| Correctness | Shift type matches ground truth |
| Completeness | Detect + classify + attribute present |
| Groundedness | Shares/p-values cited; no invented blocks |
| Tool-use quality | Right mode (pathway vs loco) for shift type |
| Process quality | Gate before promote; badcases taxonomized |

## Closed loop

`Evaluation → Data → Optimization → Verification`

Miss → `taxonomize_badcase` → `promote_to_dataset` → improve threshold/method/prompt → `run_fixture_suite` + `eval_gate`.
