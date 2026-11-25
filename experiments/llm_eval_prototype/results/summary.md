# LLM-eval PO-risk smoke summary

Seeds: [202601, 202602, 202603]. Permutations: 49.
Primary endpoint: attribution hit (concept → planted block; null → p≥0.05; benign covariate → not content_signal).
Score: PO(S)=mean(tau_S^2) with random-forest nuisances. Residual mean((psi-tau)^2) is logged only.

## By mechanism

| mechanism        |   n |   hit_rate |   mean_p |   reject_rate_p05 |   mean_po_all |   mean_brier_gap | top_mode       |
|:-----------------|----:|-----------:|---------:|------------------:|--------------:|-----------------:|:---------------|
| benign_covariate |   3 |   0.666667 | 0.293333 |                 0 |     0.0140811 |       0.00787534 | prompt_meta    |
| concept_content  |   3 |   1        | 0.02     |                 1 |     0.0857559 |       0.12215    | content_signal |
| concept_style    |   3 |   1        | 0.02     |                 1 |     0.0687148 |       0.0647418  | response_style |
| no_shift         |   3 |   1        | 0.633333 |                 0 |     0.0137275 |       0.00921453 | prompt_meta    |

## Diagnostics head

| mechanism        |   seed | top_block      |   attribution_hit |   p_value |    po_all |   brier_gap |
|:-----------------|-------:|:---------------|------------------:|----------:|----------:|------------:|
| no_shift         | 202601 | prompt_meta    |                 1 |      0.24 | 0.0120968 | -0.00720667 |
| benign_covariate | 202601 | prompt_meta    |                 1 |      0.1  | 0.0131799 | -0.0057578  |
| concept_content  | 202601 | content_signal |                 1 |      0.02 | 0.0789199 |  0.0901781  |
| concept_style    | 202601 | response_style |                 1 |      0.02 | 0.071545  |  0.0841427  |
| no_shift         | 202602 | response_style |                 1 |      0.7  | 0.0090615 |  0.0425595  |
| benign_covariate | 202602 | prompt_meta    |                 1 |      0.64 | 0.0109009 |  0.0414874  |
| concept_content  | 202602 | content_signal |                 1 |      0.02 | 0.0879392 |  0.15089    |
| concept_style    | 202602 | response_style |                 1 |      0.02 | 0.0588868 |  0.0693369  |
| no_shift         | 202603 | prompt_meta    |                 1 |      0.96 | 0.0200241 | -0.0077092  |
| benign_covariate | 202603 | content_signal |                 0 |      0.14 | 0.0181626 | -0.0121036  |
| concept_content  | 202603 | content_signal |                 1 |      0.02 | 0.0904086 |  0.125382   |
| concept_style    | 202603 | response_style |                 1 |      0.02 | 0.0757127 |  0.040746   |

## Block MMD reject rate (p<0.05)

| mechanism        | block          |   reject |
|:-----------------|:---------------|---------:|
| benign_covariate | content_signal | 0        |
| benign_covariate | prompt_meta    | 1        |
| benign_covariate | response_style | 0        |
| concept_content  | content_signal | 0        |
| concept_content  | prompt_meta    | 0.333333 |
| concept_content  | response_style | 0        |
| concept_style    | content_signal | 0        |
| concept_style    | prompt_meta    | 0.333333 |
| concept_style    | response_style | 0        |
| no_shift         | content_signal | 0        |
| no_shift         | prompt_meta    | 0.333333 |
| no_shift         | response_style | 0        |
