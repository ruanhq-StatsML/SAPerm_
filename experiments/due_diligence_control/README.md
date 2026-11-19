# Due-diligence control (Phase 1) + FinanceGPT hooks

**Yes — the bilateral exposure method can empower due diligence.**  
The scarce resources are the same shape; only the nouns change.

| Recruiting control | Due diligence |
| --- | --- |
| Candidate attention | Analyst attention / alert fatigue |
| Review + interview capacity | Associate/partner review + IC slots |
| Marketing traffic | Research / data-room coverage |
| Job slots in one email | Docs in one digest |
| Backlog + pacing | Unread data-room + IC calendar pacing |
| Supply shortage vs matching failure | Not enough signal vs noisy flags / bad retrieval |

## Design rule

FinanceGPT / FinGPT / FinBERT **score documents**.  
Rule-based control **allocates scarce attention**.  
Model scores never bypass mute, frequency caps, or `backlog_hard`.

## What is implemented

- Analyst: frequency, cooldown, sector cap, pipeline, mute / ignore
- Desk: review-room cap, **backlog control**, pacing, sector dedup
- Staged DD: screen → red flag → deep dive → IC (dynamic)
- Bilateral score: \(\Delta V - C^{fatigue} - C^{congestion} - C^{redundancy}\)
- Model hooks: `heuristic` (default), `finbert` (optional), `PromptedRiskProvider` for any FinanceGPT callable
- 2×2 factorial simulator

## Run

```bash
python3 -m pytest experiments/due_diligence_control/test_smoke.py -q
python3 -m experiments.due_diligence_control
```

Plug a fine-tuned FinanceGPT:

```python
from experiments.due_diligence_control.model_hooks import PromptedRiskProvider, score_documents

def finance_gpt(text: str) -> float:
    # your LoRA / API → risk in [0, 1]
    ...

provider = PromptedRiskProvider(finance_gpt, name="FinanceGPT-DD")
```

## Model shortlist (Phase 2)

| Model | Role |
| --- | --- |
| `ProsusAI/finbert` | Fast first-pass tone/risk on excerpts |
| `AdaptLLM/finance-LLM` | Domain generative backbone for memos |
| FinGPT LoRA family | Filing QA / sentiment multi-task |
| **Your FinanceGPT LoRA** | SFT/DPO on labeled red flags + IC outcomes |

Avoid treating random Hub "FinanceGPT" forks as production DD brains without your labeled corpus.

## Opportunity roadmap

| Phase | Work |
| --- | --- |
| 1 | This package — rules + staged release + heuristic/FinanceGPT hook |
| 2 | FinBERT filter → FinanceGPT red-flag QA; capacity quantile forecasts |
| 3 | Causal / uplift model for marginal doc value |
| 4 | Min-cost flow over analysts × deals under the same hard caps; FSDS even-effort on desks |

Sister packages: `experiments/bilateral_exposure_control` (recruiting), `fsds-review-agent` (strategy board 审核).
