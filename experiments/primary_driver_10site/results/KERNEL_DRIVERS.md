# Per-site MMD, HSIC, and R-risk on the synthetic designs

Each site is scored on its own reference window and its own new window.
MMD is the covariate score. R-risk is the concept score. HSIC is the dependence between the reference residual and the environment indicator.
Skill is out-of-fold predictability of the new-window outcome. Gated scores multiply HSIC and R-risk by that skill.
The pooled row stacks every site and recomputes the same three numbers. That overall number does not name a site.
The concept score is the larger of skill-gated R-risk and the largest coordinate HSIC gap times the same skill.
A principal driver is named only when that score beats the lower half of the federation by 0.03 for a concept call, or by 0.006 for an MMD call.

## Heterogeneous federation, mean score by role

| role | mmd | hsic | r_risk | skill | hsic_gated | r_gated |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.001 | 0.007 | 0.009 | 0.797 | 0.006 | 0.007 |
| concept | 0.001 | 0.028 | 0.069 | 0.796 | 0.023 | 0.059 |
| covariate | 0.047 | 0.009 | 0.011 | 0.845 | 0.007 | 0.009 |
| mixed | 0.025 | 0.017 | 0.022 | 0.822 | 0.014 | 0.018 |
| noise | 0.002 | 0.122 | 0.388 | 0.000 | 0.000 | 0.000 |

## Concordance against clean sites

A cell is the fraction of (driver, clean) pairs with a strict inequality, averaged over seeds.
Covariate drivers are scored by MMD. Concept drivers are scored by raw R-risk and by skill-gated R-risk.
MMD on a concept-versus-clean contrast is a negative control.

| design | mmd_vs_clean | mmd_concept_vs_clean | r_vs_clean | r_gated_vs_clean | concept_vs_clean | hsic_gated_vs_clean | r_raw_vs_noise | r_gated_vs_noise | concept_vs_noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_10site |  | 0.412 | 0.950 | 0.950 | 0.912 | 1.000 | 0.800 | 0.900 | 1.000 |
| hetero_10 | 1.000 | 0.500 | 0.781 | 0.781 | 0.906 | 0.875 | 0.000 | 0.938 | 1.000 |
| homo_cov_0.6 | 0.611 |  |  |  |  |  |  |  |  |
| homo_cubic |  | 0.500 | 1.000 | 1.000 | 1.000 | 0.917 |  |  |  |
| homo_interaction |  | 0.306 | 0.500 | 0.500 | 0.583 | 0.833 |  |  |  |
| homo_linear |  | 0.167 | 1.000 | 1.000 | 1.000 | 1.000 |  |  |  |
| homo_mean_0.5 | 1.000 |  |  |  |  |  |  |  |  |
| homo_mean_1.5 | 1.000 |  |  |  |  |  |  |  |  |
| homo_mean_interaction | 1.000 | 1.000 | 0.639 | 0.639 | 1.000 | 0.944 |  |  |  |
| homo_mean_sine | 1.000 | 1.000 | 0.889 | 0.889 | 1.000 | 0.889 |  |  |  |
| homo_mixture_0.8 | 1.000 |  |  |  |  |  |  |  |  |
| homo_none |  |  |  |  |  |  |  |  |  |
| homo_sine |  | 0.333 | 0.861 | 0.861 | 1.000 | 0.833 |  |  |  |
| homo_threshold |  | 0.444 | 0.944 | 0.944 | 0.917 | 1.000 |  |  |  |
| mixed_block | 1.000 | 0.417 | 1.000 | 1.000 | 1.000 | 0.583 |  |  |  |

## Principal-driver calls

Hit is 1 when the named site has the right kind of shift, or when a design with no shift of that kind names nobody.

| design | arm | covariate_hit | concept_hit | covariate_precision | covariate_recall | concept_precision | concept_recall | noise_called_concept | site_coordinate_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_10site | concept | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.400 | 0.000 | 0.900 |
| hetero_10 | both | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.625 | 0.000 | 1.000 |
| homo_cov_0.6 | covariate | 0.000 | 0.750 | 0.000 | 0.000 | 0.750 | 1.000 | 0.000 | 0.750 |
| homo_cubic | concept | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 |
| homo_interaction | concept | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.833 |
| homo_linear | concept | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.000 | 1.000 |
| homo_mean_0.5 | covariate | 0.750 | 0.750 | 0.750 | 0.583 | 0.750 | 1.000 | 0.000 | 0.917 |
| homo_mean_1.5 | covariate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| homo_mean_interaction | both | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.000 | 1.000 |
| homo_mean_sine | both | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.000 | 1.000 |
| homo_mixture_0.8 | covariate | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 1.000 |
| homo_none | none | 1.000 | 0.750 | 1.000 | 1.000 | 0.750 | 1.000 | 0.000 |  |
| homo_sine | concept | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| homo_threshold | concept | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.000 | 0.917 |
| mixed_block | both | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.917 | 0.000 | 1.000 |

## Pooled statistic against the strongest site

The pooled column is the overall score. The max column is the strongest site.
A driver that is one site among many is visible in the max and diluted in the pool.

| design | pooled_mmd | max_mmd | pooled_r_risk | max_r_risk | pooled_r_gated | max_r_gated |
| --- | --- | --- | --- | --- | --- | --- |
| binary_10site | 0.000 | 0.002 | 0.007 | 0.203 | 0.002 | 0.176 |
| hetero_10 | 0.003 | 0.075 | 0.005 | 0.388 | 0.003 | 0.115 |
| homo_cov_0.6 | 0.001 | 0.004 | 0.000 | 0.021 | 0.000 | 0.017 |
| homo_cubic | 0.001 | 0.002 | 0.070 | 0.368 | 0.055 | 0.298 |
| homo_interaction | 0.000 | 0.002 | 0.003 | 0.041 | 0.002 | 0.032 |
| homo_linear | 0.002 | 0.003 | 0.049 | 0.203 | 0.041 | 0.178 |
| homo_mean_0.5 | 0.001 | 0.008 | 0.002 | 0.027 | 0.001 | 0.021 |
| homo_mean_1.5 | 0.013 | 0.078 | 0.002 | 0.014 | 0.002 | 0.011 |
| homo_mean_interaction | 0.005 | 0.030 | 0.020 | 0.020 | 0.015 | 0.015 |
| homo_mean_sine | 0.005 | 0.034 | 0.012 | 0.065 | 0.010 | 0.054 |
| homo_mixture_0.8 | 0.019 | 0.070 | 0.001 | 0.022 | 0.001 | 0.018 |
| homo_none | 0.000 | 0.002 | 0.002 | 0.026 | 0.002 | 0.021 |
| homo_sine | 0.001 | 0.001 | 0.012 | 0.076 | 0.010 | 0.062 |
| homo_threshold | 0.000 | 0.002 | 0.014 | 0.084 | 0.012 | 0.070 |
| mixed_block | 0.013 | 0.075 | 0.022 | 0.191 | 0.019 | 0.167 |

## Coordinate recovery after aggregation

The target of an MMD vector is the coordinates carried by covariate and mixed sites.
The target of an HSIC or R-risk vector is the coordinates carried by concept and mixed sites.
Weights are the site MMD or the skill-gated R-risk. The pooled aggregator recomputes leave-one-coordinate on the stacked sample.

| design | aggregator | spearman | auroc | precision_at_k | top_in_support |
| --- | --- | --- | --- | --- | --- |
| binary_10site | mmd_weighted_mmd |  |  |  | 0.000 |
| binary_10site | pooled_hsic | 0.461 | 0.725 | 0.583 | 0.750 |
| binary_10site | pooled_mmd |  |  |  | 0.000 |
| binary_10site | pooled_r | -0.093 | 0.392 | 0.250 | 0.000 |
| binary_10site | rgated_weighted_hsic | 0.821 | 0.983 | 0.917 | 1.000 |
| binary_10site | rgated_weighted_r | 0.641 | 0.867 | 0.667 | 1.000 |
| binary_10site | uniform_hsic | 0.873 | 1.000 | 1.000 | 1.000 |
| binary_10site | uniform_mmd |  |  |  | 0.000 |
| binary_10site | uniform_r | 0.655 | 0.833 | 0.667 | 1.000 |
| hetero_10 | mmd_weighted_mmd | 0.589 | 0.828 | 0.750 | 1.000 |
| hetero_10 | pooled_hsic | 0.049 | 0.531 | 0.625 | 0.250 |
| hetero_10 | pooled_mmd | 0.595 | 0.776 | 0.625 | 1.000 |
| hetero_10 | pooled_r | -0.270 | 0.365 | 0.458 | 0.250 |
| hetero_10 | rgated_weighted_hsic | 0.497 | 0.792 | 0.792 | 1.000 |
| hetero_10 | rgated_weighted_r | 0.462 | 0.771 | 0.750 | 1.000 |
| hetero_10 | uniform_hsic | 0.391 | 0.729 | 0.708 | 1.000 |
| hetero_10 | uniform_mmd | 0.626 | 0.839 | 0.812 | 1.000 |
| hetero_10 | uniform_r | 0.231 | 0.635 | 0.667 | 1.000 |
| homo_cov_0.6 | mmd_weighted_mmd | 0.494 | 0.972 | 0.750 | 0.750 |
| homo_cov_0.6 | pooled_hsic |  |  |  | 0.000 |
| homo_cov_0.6 | pooled_mmd | 0.350 | 0.736 | 0.250 | 0.500 |
| homo_cov_0.6 | pooled_r |  |  |  | 0.000 |
| homo_cov_0.6 | rgated_weighted_hsic |  |  |  | 0.000 |
| homo_cov_0.6 | rgated_weighted_r |  |  |  | 0.000 |
| homo_cov_0.6 | uniform_hsic |  |  |  | 0.000 |
| homo_cov_0.6 | uniform_mmd | 0.523 | 1.000 | 1.000 | 1.000 |
| homo_cov_0.6 | uniform_r |  |  |  | 0.000 |
| homo_cubic | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_cubic | pooled_hsic | 0.545 | 1.000 | 1.000 | 1.000 |
| homo_cubic | pooled_mmd |  |  |  | 0.000 |
| homo_cubic | pooled_r | 0.594 | 1.000 | 1.000 | 1.000 |
| homo_cubic | rgated_weighted_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_cubic | rgated_weighted_r | 0.526 | 1.000 | 1.000 | 1.000 |
| homo_cubic | uniform_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_cubic | uniform_mmd |  |  |  | 0.000 |
| homo_cubic | uniform_r | 0.526 | 1.000 | 1.000 | 1.000 |
| homo_interaction | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_interaction | pooled_hsic | 0.564 | 0.883 | 0.750 | 0.500 |
| homo_interaction | pooled_mmd |  |  |  | 0.000 |
| homo_interaction | pooled_r | 0.242 | 0.625 | 0.375 | 0.500 |
| homo_interaction | rgated_weighted_hsic | 0.631 | 0.953 | 0.875 | 1.000 |
| homo_interaction | rgated_weighted_r | 0.098 | 0.570 | 0.500 | 0.750 |
| homo_interaction | uniform_hsic | 0.696 | 1.000 | 1.000 | 1.000 |
| homo_interaction | uniform_mmd |  |  |  | 0.000 |
| homo_interaction | uniform_r | 0.054 | 0.539 | 0.375 | 0.750 |
| homo_linear | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_linear | pooled_hsic | 0.538 | 1.000 | 1.000 | 1.000 |
| homo_linear | pooled_mmd |  |  |  | 0.000 |
| homo_linear | pooled_r | 0.625 | 1.000 | 1.000 | 1.000 |
| homo_linear | rgated_weighted_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_linear | rgated_weighted_r | 0.523 | 1.000 | 1.000 | 1.000 |
| homo_linear | uniform_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_linear | uniform_mmd |  |  |  | 0.000 |
| homo_linear | uniform_r | 0.523 | 1.000 | 1.000 | 1.000 |
| homo_mean_0.5 | mmd_weighted_mmd | 0.549 | 1.000 | 1.000 | 1.000 |
| homo_mean_0.5 | pooled_hsic |  |  |  | 0.000 |
| homo_mean_0.5 | pooled_mmd | 0.332 | 0.722 | 0.250 | 0.750 |
| homo_mean_0.5 | pooled_r |  |  |  | 0.000 |
| homo_mean_0.5 | rgated_weighted_hsic |  |  |  | 0.000 |
| homo_mean_0.5 | rgated_weighted_r |  |  |  | 0.000 |
| homo_mean_0.5 | uniform_hsic |  |  |  | 0.000 |
| homo_mean_0.5 | uniform_mmd | 0.549 | 1.000 | 1.000 | 1.000 |
| homo_mean_0.5 | uniform_r |  |  |  | 0.000 |
| homo_mean_1.5 | mmd_weighted_mmd | 0.573 | 1.000 | 1.000 | 1.000 |
| homo_mean_1.5 | pooled_hsic |  |  |  | 0.000 |
| homo_mean_1.5 | pooled_mmd | 0.873 | 1.000 | 1.000 | 1.000 |
| homo_mean_1.5 | pooled_r |  |  |  | 0.000 |
| homo_mean_1.5 | rgated_weighted_hsic |  |  |  | 0.000 |
| homo_mean_1.5 | rgated_weighted_r |  |  |  | 0.000 |
| homo_mean_1.5 | uniform_hsic |  |  |  | 0.000 |
| homo_mean_1.5 | uniform_mmd | 0.573 | 1.000 | 1.000 | 1.000 |
| homo_mean_1.5 | uniform_r |  |  |  | 0.000 |
| homo_mean_interaction | mmd_weighted_mmd | 0.378 | 0.703 | 0.500 | 1.000 |
| homo_mean_interaction | pooled_hsic | 0.722 | 1.000 | 1.000 | 1.000 |
| homo_mean_interaction | pooled_mmd | 0.543 | 0.773 | 0.500 | 1.000 |
| homo_mean_interaction | pooled_r | 0.623 | 0.906 | 0.875 | 1.000 |
| homo_mean_interaction | rgated_weighted_hsic | 0.696 | 1.000 | 1.000 | 1.000 |
| homo_mean_interaction | rgated_weighted_r | 0.239 | 0.672 | 0.500 | 0.750 |
| homo_mean_interaction | uniform_hsic | 0.696 | 1.000 | 1.000 | 1.000 |
| homo_mean_interaction | uniform_mmd | 0.334 | 0.672 | 0.500 | 1.000 |
| homo_mean_interaction | uniform_r | 0.326 | 0.734 | 0.500 | 0.750 |
| homo_mean_sine | mmd_weighted_mmd | 0.440 | 0.742 | 0.625 | 1.000 |
| homo_mean_sine | pooled_hsic | 0.327 | 0.727 | 0.500 | 0.750 |
| homo_mean_sine | pooled_mmd | 0.580 | 0.797 | 0.625 | 1.000 |
| homo_mean_sine | pooled_r | 0.475 | 0.711 | 0.500 | 1.000 |
| homo_mean_sine | rgated_weighted_hsic | 0.305 | 0.719 | 0.500 | 1.000 |
| homo_mean_sine | rgated_weighted_r | 0.485 | 0.844 | 0.500 | 1.000 |
| homo_mean_sine | uniform_hsic | 0.261 | 0.688 | 0.500 | 1.000 |
| homo_mean_sine | uniform_mmd | 0.440 | 0.742 | 0.625 | 1.000 |
| homo_mean_sine | uniform_r | 0.485 | 0.844 | 0.500 | 1.000 |
| homo_mixture_0.8 | mmd_weighted_mmd | 0.698 | 1.000 | 1.000 | 1.000 |
| homo_mixture_0.8 | pooled_hsic |  |  |  | 0.000 |
| homo_mixture_0.8 | pooled_mmd | 0.873 | 1.000 | 1.000 | 1.000 |
| homo_mixture_0.8 | pooled_r |  |  |  | 0.000 |
| homo_mixture_0.8 | rgated_weighted_hsic |  |  |  | 0.000 |
| homo_mixture_0.8 | rgated_weighted_r |  |  |  | 0.000 |
| homo_mixture_0.8 | uniform_hsic |  |  |  | 0.000 |
| homo_mixture_0.8 | uniform_mmd | 0.698 | 1.000 | 1.000 | 1.000 |
| homo_mixture_0.8 | uniform_r |  |  |  | 0.000 |
| homo_none | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_none | pooled_hsic |  |  |  | 0.000 |
| homo_none | pooled_mmd |  |  |  | 0.000 |
| homo_none | pooled_r |  |  |  | 0.000 |
| homo_none | rgated_weighted_hsic |  |  |  | 0.000 |
| homo_none | rgated_weighted_r |  |  |  | 0.000 |
| homo_none | uniform_hsic |  |  |  | 0.000 |
| homo_none | uniform_mmd |  |  |  | 0.000 |
| homo_none | uniform_r |  |  |  | 0.000 |
| homo_sine | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_sine | pooled_hsic | 0.533 | 1.000 | 1.000 | 1.000 |
| homo_sine | pooled_mmd |  |  |  | 0.000 |
| homo_sine | pooled_r | 0.645 | 0.972 | 0.750 | 0.750 |
| homo_sine | rgated_weighted_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_sine | rgated_weighted_r | 0.525 | 1.000 | 1.000 | 1.000 |
| homo_sine | uniform_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_sine | uniform_mmd |  |  |  | 0.000 |
| homo_sine | uniform_r | 0.525 | 1.000 | 1.000 | 1.000 |
| homo_threshold | mmd_weighted_mmd |  |  |  | 0.000 |
| homo_threshold | pooled_hsic | 0.391 | 0.806 | 0.750 | 0.750 |
| homo_threshold | pooled_mmd |  |  |  | 0.000 |
| homo_threshold | pooled_r | 0.103 | 0.597 | 0.250 | 0.250 |
| homo_threshold | rgated_weighted_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_threshold | rgated_weighted_r | 0.348 | 0.833 | 0.500 | 0.500 |
| homo_threshold | uniform_hsic | 0.522 | 1.000 | 1.000 | 1.000 |
| homo_threshold | uniform_mmd |  |  |  | 0.000 |
| homo_threshold | uniform_r | 0.348 | 0.833 | 0.500 | 0.500 |
| mixed_block | mmd_weighted_mmd | 0.602 | 0.820 | 0.625 | 1.000 |
| mixed_block | pooled_hsic | 0.405 | 0.696 | 0.583 | 0.750 |
| mixed_block | pooled_mmd | 0.689 | 0.789 | 0.500 | 1.000 |
| mixed_block | pooled_r | 0.652 | 0.762 | 0.583 | 1.000 |
| mixed_block | rgated_weighted_hsic | 0.556 | 0.821 | 0.750 | 1.000 |
| mixed_block | rgated_weighted_r | 0.749 | 0.952 | 0.750 | 1.000 |
| mixed_block | uniform_hsic | 0.556 | 0.821 | 0.750 | 1.000 |
| mixed_block | uniform_mmd | 0.582 | 0.805 | 0.625 | 1.000 |
| mixed_block | uniform_r | 0.648 | 0.881 | 0.667 | 1.000 |
