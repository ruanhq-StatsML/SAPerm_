# RecSys Landing Roadmap — OnlineRFPerm × CFPerm × RFPerm

Map the three papers onto a deployed recommender: **when** to refresh
(OnlineRFPerm), **what** drifted (CFPerm attribution), and **how** text /
review signals behave under benign shifts (Sentiment RFPerm / RriskPerm).

## Mapping

| Paper | RecSys role | Decision it owns |
| --- | --- | --- |
| OnlineRFPerm | Sequential monitor on rating / CTR / engagement loss of a frozen scorer | *When* the deployed model needs refresh (FDR-controlled) |
| CFPerm (meta-learner VIMP) | Treat exist vs new as control vs treatment; rank user / item / context / review blocks | *What* to retrain, re-embed, or re-segment |
| Sentiment RFPerm / RriskPerm | Review / title / prompt text as modality; separate concept drift from benign label-preserving noise | *Whether* a text shift is refresh-worthy |

## Phase 0 — Rating monitor (this PR)

Synthetic (and optional MovieLens) interaction stream with user / item / context
blocks and a continuous rating.

1. Fit `f_ref` on a reference window (RandomForest / Ridge).
2. Burn-in batches build the historical degradation pool `T_j = MSE_j − E_ref`.
3. Incoming batches produce conformal-style empirical p-values; ADDIS / SAFFRON /
   fixed-α decide rejections under online FDR.
4. Optional EWMA on p-values for earlier alerts (practical, not super-uniform).
5. On reject → emit a **refresh pre-signal** and reset state (wealth, history, burn-in).

Success: controlled false alarms under stationary DGP; detection delay small after
planted concept drift or covariate shift.

## Phase 1 — Block attribution (this PR, lightweight)

When OnlineRFPerm rejects (or on a rolling schedule):

1. Run CFPerm-style PO-risk / LOCO (and optional MMD) on feature blocks:
   `user`, `item`, `context`, optional `review`.
2. Rank blocks; surface top drivers for the refresh ticket.
3. Align with `tabular-drift-attribution` MovieLens blocks for a shared vocabulary.

Success: planted concept drift on item taste ranks `item` (or the drifted block)
first; pure covariate shift on user demographics ranks `user`.

## Phase 2 — Localization & cohort refresh

1. Partition users along CFPerm top features (age / occupation / genre affinity).
2. Re-run the two-batch test inside each cohort (post-hoc subset analysis from CFPerm).
3. Prefer **partial refresh** (cohort-specific re-ranker / bias term) when only a
   subset rejects.

## Phase 3 — Online experiment platform

From OnlineRFPerm Appendix A.9:

1. Trigger CUPED / CUPAC re-fit of the pre-experiment covariate model at each
   OnlineRFPerm onset.
2. Use the same monitor on A/B arms so variance reduction stays valid under drift.
3. Tie into the existing `experiments/uplift_monitor` Thompson loop as the
   allocation policy while the monitor stays on the randomized slice.

## Phase 4 — Production workflow

Follow OnlineRFPerm Appendix A.4–A.7 state machine:

```
load f_ref → burn-in → monitor → reject?
  no  → append T, continue
  yes → retrain → deploy → full state reset
```

Ops requirements called out in the paper: model registry (`fit`/`predict` only),
JSON metrics (`p_t`, `α_t`, reject, latency), durable checkpoints, graceful skip
if a batch exceeds the latency budget, wealth reset if ADDIS discards too often.

## Phase 5 — Multimodal / LLM recsys

1. Encode reviews / titles with the Sentiment paper pipeline (BERT / LLM
   embeddings); run RFPerm vs RriskPerm to tell refresh-worthy concept drift
   from benign paraphrases.
2. Cross-modal attribution: text block vs tabular user/item (CFPerm + pathway
   attribution already in `tabular-drift-attribution`).
3. Shadow LLM ranker: OnlineRFPerm watches a cheap RF/XGBoost scorer as the
   OOD characterizer while the production deep model retrains offline.

## Deliverables in this package

| Artifact | Role |
| --- | --- |
| `dgp.py` | Synthetic recsys panel with planted concept / covariate / both |
| `online_rfperm.py` + `online_fdr.py` | Phase 0 monitor |
| `cfperm_attr.py` | Phase 1 block ranking |
| `pipeline.py` / `run.py` | Detect → attribute → refresh board |
| `results/BOARD.md` | Offline reading of power / delay / attribution |
