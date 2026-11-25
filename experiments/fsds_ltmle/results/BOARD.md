# FSDS-guided longitudinal TMLE

Target is the always-treat minus never-treat contrast. Truth is a Monte Carlo g-formula under the intervened covariate law. The frozen estimator reuses Super Learner weights from the pre-shift terminal regression. Refit chooses weights by cross-validation on the evaluation batch and keeps truncation at 0.02. FSDS uses the same cross-validation, then applies the concept-arm ridge penalty and the covariate-arm truncation. No feature is removed from the adjustment set.

Configuration: n=480, T=3, diagnostic trees=40, Q trees=40, permutations=5, seeds=[0, 1, 2, 3].

## Absolute error

| Regime | Truth | |frozen| (sd) | |refit| (sd) | |FSDS| (sd) | Level | Forest weight frozen / refit / FSDS |
| --- | --- | --- | --- | --- | --- | --- |
| stable | 0.954 | 0.181 (0.063) | 0.158 (0.060) | 0.158 (0.060) | GREEN | 0.21 / 0.14 / 0.14 |
| covariate | 0.949 | 0.288 (0.179) | 0.271 (0.197) | 0.254 (0.185) | ORANGE_COVARIATE | 0.21 / 0.14 / 0.14 |
| concept | -0.754 | 0.139 (0.070) | 0.129 (0.070) | 0.129 (0.069) | ORANGE_CONCEPT | 0.21 / 0.99 / 1.00 |
| joint | -0.683 | 0.158 (0.077) | 0.173 (0.110) | 0.167 (0.079) | RED | 0.21 / 1.00 / 1.00 |

## Where the arms pointed

- `stable`: covariate top-3 hit n/a, concept top-3 hit n/a. Per seed the top covariate/concept features were severity_last/noise_extra_6, noise_extra_5/redemption_last, redemption_last/noise1_mean, noise_extra_1/treat_rate. Mean overlap 0.959. Mean truncation 0.020. Mean ridge penalty 0.000.
- `covariate`: covariate top-3 hit 1.00, concept top-3 hit n/a. Per seed the top covariate/concept features were channel_mean/treat_rate, channel_last/channel_last, channel_mean/noise1_mean, channel_mean/channel_mean. Mean overlap 0.797. Mean truncation 0.100. Mean ridge penalty 0.000.
- `concept`: covariate top-3 hit n/a, concept top-3 hit 1.00. Per seed the top covariate/concept features were severity_mean/redemption_mean, severity_mean/redemption_mean, channel_last/redemption_mean, severity_mean/redemption_mean. Mean overlap 0.942. Mean truncation 0.020. Mean ridge penalty 3.000.
- `joint`: covariate top-3 hit 1.00, concept top-3 hit 1.00. Per seed the top covariate/concept features were channel_mean/redemption_mean, channel_mean/redemption_mean, channel_mean/redemption_mean, channel_last/redemption_mean. Mean overlap 0.753. Mean truncation 0.120. Mean ridge penalty 3.000.

A top-3 hit is reported only when that regime plants a driver. `channel_*` is the planted covariate driver. `redemption_*` is the planted concept driver. `treat_rate` ranks with redemption when the blip changes, because it is the exposure whose payoff changed, and it can rank under channel when a mix shift changes who is offered the incentive. Neither ranking is a reason to delete the feature.

Ranks are a review order. Two stable batches already move the VIMP numbers, so a peak is read against that null rather than against zero. The adjustment set on every row is severity, channel, redemption, and the dropped-feature list is empty.

The absolute-error gaps sit inside the seed standard deviation. What the diagnostic changes reliably is the routing: on the quadratic blip the forest weight moves from the frozen value up to the refit value, and on a channel shift the truncation moves from 0.02 to 0.10 while the forest weight stays on ridge. A joint batch raises both knobs and sets the pause flag.
