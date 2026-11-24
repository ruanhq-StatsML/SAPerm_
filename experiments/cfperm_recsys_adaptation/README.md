# CFPerm-guided update selection

Downstream benchmark for a fixed adaptation budget. It does not change the CFPerm, RFPerm, or PO-risk implementations already in this repository, and it does not add a cross-modal alignment model or a backbone fine-tune.

`ruanhq-StatsML` has no recommender codebase (no ranking, click, or collaborative-filtering trainer). The concrete pilot is this synthetic three-block classification task. The recommendation mapping is an application note, not a production system:

| This benchmark | Recommendation reading |
| --- | --- |
| text / audio / visual block | a feature block (item text, a second content or context block, item metadata or visual features) |
| domain 0 / 1 | reference serving slice versus shifted slice; a batch label, not a treatment |
| target Brier | Brier risk of the observed click or conversion label on the shifted slice |
| adapter / fusion / head groups | which block encoder, interaction layer, or head to update |
| tertile bin | a user or item subset defined from reference data |

MSR-VTT attribution in the CFPerm repository is the closest multimodal artifact (video, audio, and text embeddings). SAPerm is sentiment classification on embeddings. Neither is a recommender.

## Smoke test

```bash
python experiments/cfperm_recsys_adaptation/run_experiment.py --dataset synthetic --seeds 3 --permutations 99 --quick
```

`--quick` uses the spec's starting sample sizes, three seeds, and one label budget (100), with shorter Adam runs than the budget sweep. It is not the 499-permutation, multi-budget grid. `--dataset modasense` stops without writing metrics when no verified embedding table is passed via `--input`. Read the CSVs with `keep_default_na=False` so the scenario name `null` is preserved.

Outputs land in `experiments/cfperm_recsys_adaptation/results/`.

## What the selection rules use

- MMD-only: smallest modality-wise permutation p-value below 0.05, otherwise no update. Distances are not treated as percentage shares.
- Predictive-risk-only: Brier gap of frozen logistic subset models. No update when the full-model gap is at most 0.01.
- CFPerm-guided: joint-null max statistic of the block PO-distance on the task label. PO(S) is mean(tau_S^2) on held-out rows inside the full-block overlap mask. Propensity clip is [0.05, 0.95]. The same contrast on frozen-model squared loss is written out and is not used to pick the arm.
- The installed `po_risk` residual, mean((psi - tau)^2), is written as `po_squared_prediction_error` and is not the selection score.
- CFPerm0 and CFPerm1 are not separate functions in the inspected CFPerm entry points, so this runner does not report them.
