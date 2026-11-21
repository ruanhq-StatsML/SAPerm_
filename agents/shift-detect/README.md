# shift-detect — Distribution-Shift Attribution Agent

Agent SDK agent that wraps the **ruanhq-StatsML / SAPerm** monitoring stack into an
**attribution-first** workflow:

**Detect → Classify → Attribute → Gate → Promote**

It operationalizes the Agent Evaluation framework (Dataset / Rubric / Evaluator /
LLM-as-Judge hooks / Human Evaluation / Regression / Eval Gate / Badcase taxonomy /
Evaluation→Data→Optimization→Verification closed loop) on distribution-shift problems.

## Why attribution

Detection alone answers “did something move?”. This agent answers:

1. **What kind** — covariate / concept / label / mixed
2. **What drove it** — pathway block shares or LOCO PO-risk drivers
3. **What to do** — monitor, reweight, retrain concept, or promote a badcase

Portable TypeScript implementations ship here. Heavy siblings stay in the research repos:

| Agent capability | Research sibling |
| --- | --- |
| MMD / C2ST / mean tests | SAPerm benchmarks |
| PO-risk / delta_cf | OnlineMonitoring_Out_of_Distribution, primary_driver_10site |
| Pathway block attribution | `tabular-drift-attribution` |
| LOCO concept drivers | CFPerm, OOD_variable_importance |
| Permute-then-refit | RFPerm, OnlineRFPerm, XGBPerm |
| Feature selection under shift | FSDS-FeatureSelection_for_DistributionShift |

## Quick start

```bash
cd agents/shift-detect
npx @grok/agent-sdk validate --dir .
npx @grok/agent-sdk tool call attribute_shift --dir . --input '{"fixture":"covariate_shift","mode":"pathway"}'
npx @grok/agent-sdk tool call score_report --dir . --input '{"fixture":"covariate_shift"}'
npx @grok/agent-sdk tool call run_fixture_suite --dir . --input '{}'
npx @grok/agent-sdk dev --dir .
```

## Tools

| Tool | Role |
| --- | --- |
| `attribute_shift` | Pathway / block-mean / LOCO attribution |
| `detect_shift` | MMD, C2ST, mean_diff, PO-risk, delta_cf |
| `classify_shift_type` | Shift taxonomy |
| `score_report` | One-shot attribution report |
| `run_fixture_suite` | Offline batch metrics |
| `eval_gate` | Release gate on metrics |
| `taxonomize_badcase` | Error-factor taxonomy |
| `promote_to_dataset` | Closed-loop dataset candidate |
| `compare_methods` | Method A/B on a fixture |
| `list_method_catalog` | Method ↔ repo map + fixtures |

## Fixtures

`null_shift`, `covariate_shift` (planted `block_1`), `concept_drift`, `mixed_shift`, `label_noise`.

## Eval Gate defaults

- Type-I ≤ 0.1
- Covariate power ≥ 0.7
- Concept power ≥ 0.6
- Attribution top-hit ≥ 0.7
- Shift-type accuracy ≥ 0.6
