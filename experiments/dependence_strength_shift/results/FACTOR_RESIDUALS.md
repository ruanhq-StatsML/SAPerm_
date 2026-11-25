# Factor residual dependence shift — demo

Common factor removed; DSS / PO-risk / pairwise residual corr on ε.

Read **both** rankings: `A` = residual redundancy regime; `po_gap` = residual
map/hedge change. Ranking recovery: strength keeps sign agree on weaken/strengthen;
PO-risk lights up the flip. See `INSIGHTS.md`.

| Factor | kind | A (strength) | PO-risk gap | mean |Δρ| |
| --- | --- | ---: | ---: | ---: |
| f0 | weaken | -0.316 | 0.288 | 0.190 |
| f1 | strengthen | +0.137 | 2.629 | 0.260 |
| f2 | flip | +0.189 | 1.899 | 0.297 |
| f3 | none | -0.021 | -0.058 | 0.035 |
| f4 | none | -0.021 | -0.101 | 0.039 |
| f5 | none | -0.019 | -0.213 | 0.031 |

Active-block Spearman(A) strength: **0.500**, sign agree: **1.000**
Active-block Spearman(A) PO-risk: **1.000**, sign agree: **0.500**

Incremental residual R² shift `V1-V0` (rows = target residual, cols = source):

```
[[ 0.    -0.083 -0.075  0.001  0.008  0.002]
 [-0.09   0.     0.294 -0.003  0.003 -0.001]
 [-0.078  0.299  0.    -0.    -0.    -0.001]
 [ 0.001 -0.005 -0.001  0.    -0.001 -0.   ]
 [ 0.008  0.004 -0.    -0.001  0.    -0.001]
 [ 0.002 -0.001 -0.001 -0.    -0.001  0.   ]]
```
