# shift-detect — Trustworthy Eval & Attribution Agent

Agent SDK agent for the **ruanhq-StatsML / SAPerm** stack. Two linked missions:

1. **Distribution-shift attribution** — detect / classify / pathway-LOCO attribute / Eval Gate  
2. **Trustworthy text & multimodal LLM evaluation** — versioned datasets, rubrics, cross-model对照, same-model对齐, eval-side interpretability, Trust Eval Gate

This maps directly onto building a credible eval system: cross-model comparison, same-model alignment, behavior explanation from the eval lens, rigorous data/engineering, and systematized daily eval work.

## Quick start

```bash
cd agents/shift-detect
npm run demo          # one-glance attribution boards + gates
npx @grok/agent-sdk validate --dir .
npx @grok/agent-sdk tool call score_report --dir . --input '{"fixture":"covariate_shift"}'
npx @grok/agent-sdk tool call run_trust_suite --dir . --input '{}'
npx @grok/agent-sdk dev --dir .
```

`score_report` / `attribute_shift` return a `glance` board (share bars). Lead with that.

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

## Text meta-features (spaCy + VADER) — power argument

Dedicated feature script (no test logic mixed in):

```bash
cd python
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python extract_features.py --demo --pretty
python power_meta_features.py --n-rep 20 --n-perm 40 --n-doc 20
```

| scenario | baseline power | meta power | Δ |
| --- | ---: | ---: | ---: |
| null Type-I | 0.05 | 0.05 | controlled |
| sentiment_shift | 0.55 | **1.00** | **+0.45** |
| syntax_shift | 0.05 | **1.00** | **+0.95** |

Agent tool: `attribute_text_meta` (fixtures `sentiment_shift` / `entity_shift` / `style_shift`).

## Design principles

- **Clarity over size** — small versioned cases with checkable references  
- **Shared rubrics** — factuality, groundedness, safety, multimodal alignment, …  
- **对照 & 对齐** — per-axis deltas, not only overall scores  
- **Interpretability** — attribute failures to operational factors (prompt/RAG/modality/…)  
- **Gates** — hard stops before promoting checkpoints or drift methods  

Research siblings: RFPerm, CFPerm, OnlineMonitoring, FSDS, `tabular-drift-attribution`.
