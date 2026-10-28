# Quadrant monitor

A streaming coupon policy, a PO-risk gate, and one rule per uplift segment.

The reference window fits two outcome forests, Q(0, x) and Q(1, x), and a logistic propensity on the historical logging policy. The logging policy rarely coupons high-activity users, so a later shift of that coordinate makes the inverse-propensity factor explode. The deployed score is the plugin CATE times `0.5 / clip(g, ε, 1-ε)` with `ε = 0.05`. A coupon is sent when the score exceeds 0.30.

Each micro-batch randomizes 5% of users. That slice is the only place the gate is computed. PO-risk is the squared error of the deployed score against the doubly robust pseudo-outcome at the known propensity 1/2. The p-value is the right tail of that mean against the reference variance. SAFFRON (`α = 0.10`, the `onlineFDR` wealth update) turns the p-values into rejections.

A rejection runs two attributions.

- RF-domain: a random forest classifies the reference window against the new slice. Permutation importance is the covariate arm.
- PermuCATE: the outcome models and the conditional models `E[X_j | X_{-j}]` are `RandomForestRegressor`, and so is the pseudo-outcome regression. This is the conditional-permutation VIMP in `permuCATE_vimp.py`. Propensity on this slice stays at the design value 1/2.

The rule is chosen from those two arms.

| Arm | Feature the arm names | Action |
| --- | --- | --- |
| covariate | fingerprint | hard-drop above the reference 99th percentile |
| covariate | activity | tighten truncation from 0.05 to 0.15 on that upper tail |
| concept | sandbox uplift below −0.20 | blacklist that upper quartile |
| concept | anything else | refit Q on the recent sandbox; from the next batch, use it only in the upper quartile of the named feature |

The rule stays in force until ten batches in a row are not rejected. Ten quiet batches also cut the sandbox share from 5% to 2%. A new rejection puts it back to 5%.

```bash
python3 -m experiments.quadrant_monitor
```

`results/BOARD.md` is the reading of the run. Net value uses both potential outcomes: the coupon earns `Y(1) * 6 - 1`, no coupon earns `Y(0) * 6`.

The coupon rule compares an assigned cut, 0.30, on the IPW-reweighted plugin `τ̂ × (1/2) / g_ε`. The loss supplies a different cut, on τ itself: `cost / margin = 1/6`. At g = 1/2 the assigned cut is 0.30. At the propensity floor g = 0.05 it falls to 0.03 on the τ scale, which is how near-zero contrasts become false treats. Tightening the floor to 0.15 only raises that tail cut to 0.09.

A collapsed persuadable is a change in the shape of the blip. A structural nested mean model projects that new blip onto a parametric class. A longitudinal super learner keeps the parametric blip as one candidate and can leave it; on the sandbox, where g = 1/2 is known, the doubly robust pseudo-outcome stays unbiased for the new τ. The shadow refit in this run is two random forests, not that library.

The value of the gate is a false-positive bill. Turning a coupon off gains 1 on a `(Y(0), Y(1)) = (0, 0)` draw, gains 7 on a `(1, 0)` draw, and loses 5 on a `(0, 1)` draw. Sleeping dogs dominate per person because one poison atom is worth seven lost-cause false positives. Lost causes dominate headcount. PO-risk only opens the gate; the bill is `1 - 6τ` on the region the rule removes, estimated on the sandbox by the doubly robust mean. The atom table in the board is the oracle audit of that estimand. The atoms are not the latent segment labels.
