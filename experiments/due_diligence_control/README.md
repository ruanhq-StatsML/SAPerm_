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

## Can it substitute for traditional DD — especially with missing / sparse data?

Run the planted-risk stress test:

```bash
python3 -m experiments.due_diligence_control --mode substitution
# or
python3 -m experiments.due_diligence_control.substitution_study
```

Read `SUBSTITUTION_ARGUMENT.md` (checked in) and `results/SUBSTITUTION.md` (regenerated).

**Verdict in one line:** partial substitute for triage under observable sparse/dense rooms; **not** a substitute when critical documents or fields are missing — `recall_all` is bounded by availability, and FinanceGPT cannot invent redacted evidence.

### v2 → v8 iteration

- v2 `integrity.py`: refuse IC when availability / unread critical / redaction fail gates
- v3 `escalation.py`: request missing docs / unredact, soft-halt filler, recover via alt-data
- v4: **critical lane** under soft-halt (paused ≠ zero desk budget) + thin-stream absorb + live-gated unfound halt
- v5: **same-day dual-look** on critical docs that miss first confirm (no open-pool starvation)
- v6: **reopen abstain** after recovery / integrity clear; IC retry on reopened mandates
- v7: **adaptive escalation** — repeated pulses raise alt-data / unredact odds (capped)
- v8: **same-day recovery micro-pass** — read recovered/unredacted critical docs immediately
- Methods: `controlled_v1` → `controlled_v2` → `controlled` (v8)
- CI: Python experiment workflow + R 4.x

| Regime | What happens | Substitute? |
| --- | --- | --- |
| dense | Controlled ≈ flood recall, less coverage waste | Triage yes / IC no |
| sparse | Rules keep attention for late critical docs | Often better co-pilot |
| missing_docs | `recall_all` collapses for every method | **No** — escalate data |
| missing_fields | Model goes blind on redactionsctions | **No** without primary source |
| sparse_missing | Low completeness; staged shortage/failure fires | Co-pilot only |

## Opportunity roadmap

| Phase | Work |
| --- | --- |
| 1 | This package — rules + staged release + heuristic/FinanceGPT hook |
| 1b | Missing/sparse substitution board (`substitution_study`) |
| 2 | FinBERT filter → FinanceGPT red-flag QA; capacity quantile forecasts |
| 3 | Causal / uplift model for marginal doc value |
| 4 | Min-cost flow over analysts × deals under the same hard caps; FSDS even-effort on desks |

Sister packages: `experiments/bilateral_exposure_control` (recruiting), `fsds-review-agent` (strategy board 审核).
