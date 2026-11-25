# LLM-eval PO-risk prototype

Prototype (not a paper claim): attribute distribution shift in **mock judge scores**
across eval batches using FSDS-style random-forest nuisances and

\[
\mathrm{PO}(S)=\mathrm{mean}(\tau_S(X_S)^2).
\]

Primary endpoint: **attribution hit** on planted concept-drift blocks.
Does not edit CFPerm / RFPerm / `ood_vimp`.

## Run

```bash
# from repo root
python -m experiments.llm_eval_prototype --mode synthetic --seeds 202601,202602,202603 --permutations 49 --quick
python -m experiments.llm_eval_prototype --mode cross --seeds 202601,202602,202603 --quick
python -m experiments.llm_eval_prototype --mode fixture --fixture experiments/llm_eval_prototype/fixtures/mock_judge_batch.csv
```

Decision mapping:

1. Degraded? — permutation max-test on PO + `delta_data` / `delta_model` cross grid.
2. Which block? — `shift_score[m] = PO(all) - PO(all without m)`, `PO=mean(tau^2)`.
3. What to update next? — prototype stops at attribution priority; interventional \(G_j\) / ROI is out of scope.

Outputs in `results/`: attribution CSVs + `cross_eval.csv` + `plots/attribution.png`.

## Design note

User-facing argument (Chinese): project store `docs/llm-eval-prototype.md`.
