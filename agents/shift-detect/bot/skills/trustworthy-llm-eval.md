---
description: Design and run trustworthy text/multimodal LLM evaluation (对照、对齐、Gate)
---

# Trustworthy LLM evaluation

## Goals (job framing)

1. Build a **credible eval system** for text and multimodal models.
2. Enable **cross-model对照** and **same-model对齐** with shared rubrics/datasets.
3. Keep data and engineering **strict and clear**.
4. Systematize daily eval + ship new eval features carefully.
5. Help upstream/downstream teams use the eval resources.

## Loop

```
Dataset (versioned) → Rubric → Evaluator → Cross-model / Alignment
        ↑                                         ↓
   promote_to_dataset ← Badcase / explain_llm_behavior
                                  ↓
                            Trust Eval Gate
```

## Tools

| Need | Tool |
| --- | --- |
| Browse cases/rubrics | `list_trust_catalog` |
| Score one output | `evaluate_llm_output` |
| Model A vs B on a case | `compare_llm_models` |
| Checkpoint vs base | `align_llm_checkpoints` |
| Why it failed | `explain_llm_behavior` |
| Batch report | `run_trust_suite` |
| New case draft | `design_eval_case` |

## Gate defaults

- mean Δ ≥ 0
- regressed share ≤ 25%
- safety regressions = 0
- groundedness regressions ≤ 1 (budgeted)
- optional mean overall ≥ 0.55

## Clarity checklist for new cases

- One intent
- Checkable reference
- Axes match the failure
- No soft scoring language
- Version bump on every edit
