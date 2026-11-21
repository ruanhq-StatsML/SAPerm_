# shift-detect — Trustworthy Eval & Attribution Agent

Agent SDK agent for the **ruanhq-StatsML / SAPerm** stack. Two linked missions:

1. **Distribution-shift attribution** — detect / classify / pathway-LOCO attribute / Eval Gate  
2. **Trustworthy text & multimodal LLM evaluation** — versioned datasets, rubrics, cross-model对照, same-model对齐, eval-side interpretability, Trust Eval Gate

This maps directly onto building a credible eval system: cross-model comparison, same-model alignment, behavior explanation from the eval lens, rigorous data/engineering, and systematized daily eval work.

## Quick start

```bash
cd agents/shift-detect
npx @grok/agent-sdk validate --dir .
npx @grok/agent-sdk tool call run_trust_suite --dir . --input '{}'
npx @grok/agent-sdk tool call align_llm_checkpoints --dir . --input '{"baseModelId":"base-text-v1","candidateModelId":"candidate-text-v2"}'
npx @grok/agent-sdk tool call explain_llm_behavior --dir . --input '{"modelId":"weak-vl-v0","caseId":"vl_ocr_sign_v1"}'
npx @grok/agent-sdk tool call score_report --dir . --input '{"fixture":"covariate_shift"}'
npx @grok/agent-sdk dev --dir .
```

## Trustworthy LLM eval tools

| Tool | Role |
| --- | --- |
| `list_trust_catalog` | Cases, rubrics, fixture models |
| `evaluate_llm_output` | Rubric scoring for one output |
| `compare_llm_models` | Cross-model对照 |
| `align_llm_checkpoints` | Same-model对齐 + regressions |
| `explain_llm_behavior` | Eval-side behavior attribution |
| `run_trust_suite` | Offline batch + Trust Eval Gate |
| `design_eval_case` | Rigorous new-case draft |

## Shift attribution tools

| Tool | Role |
| --- | --- |
| `attribute_shift` / `score_report` | Pathway / LOCO drivers |
| `detect_shift` / `classify_shift_type` | Shift evidence + type |
| `run_fixture_suite` / `eval_gate` | Drift offline gate |
| `taxonomize_badcase` / `promote_to_dataset` | Closed loop |

## Design principles

- **Clarity over size** — small versioned cases with checkable references  
- **Shared rubrics** — factuality, groundedness, safety, multimodal alignment, …  
- **对照 & 对齐** — per-axis deltas, not only overall scores  
- **Interpretability** — attribute failures to operational factors (prompt/RAG/modality/…)  
- **Gates** — hard stops before promoting checkpoints or drift methods  

Research siblings: RFPerm, CFPerm, OnlineMonitoring, FSDS, `tabular-drift-attribution`.
