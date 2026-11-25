# Dependence Strength Shift — Ranking Recovery

Estimand: for each feature `j`,

```text
η²_{t,j} = Var_t(E_t[X_j | X_{-j}]) / Var_t(X_j)
A_j      = η²_{1,j} - η²_{0,j}
```

`A_j > 0` means the remaining features explain more of `X_j` on the target batch;
`A_j < 0` means dependence weakened. Relationship change (`conditional_mean_shift`,
`po_gap`, pairwise `|Δρ|`) answers a different question and ranks a pure sign flip
as large even when strength is unchanged.

## Synthetic (known population η²)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.682 | 0.682 | 0.750 | 1.000 |
| `conditional_mean_shift` | 0.218 | 0.592 | 0.500 | 0.500 |
| `pairwise_corr_shift` | 0.109 | 0.483 | 0.500 | 0.500 |
| `shape_change_q` | 0.218 | 0.592 | 0.500 | 0.500 |
| `po_gap` | 0.218 | 0.557 | 0.500 | 0.500 |
| `rf_domain` | 0.109 | 0.733 | 1.000 | 0.500 |

### Flip separation (relationship reverses, strength fixed)

- mean `|strength_shift|` on flip coords: **0.088**
- mean `conditional_mean_shift` on flip coords: **2.702**
- mean Q-fixed `shape_change` on flip coords: **2.426**
- mean `Corr_Q(g0, g1)` on flip coords: **-0.871** (near `-1` = pure flip)

## UCI covariates + controlled factor responses

Each UCI table is split into reference / target. A PCA factor of the real features
drives three appended responses with known population η² under weaken / strengthen / flip.
Real columns have oracle `A_j = 0`. Precision@k uses the strength-changing responses only
(flip is excluded because its oracle strength shift is zero).

### wine (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 0.500 | 0.750 | 1.000 |
| `conditional_mean_shift` | 0.000 | -0.750 | 0.500 | 0.500 |
| `pairwise_corr_shift` | -0.500 | -0.500 | 0.500 | 0.500 |
| `shape_change_q` | 0.000 | -0.750 | 0.500 | 0.500 |
| `po_gap` | -0.250 | -0.625 | 0.500 | 0.625 |
| `rf_domain` | 0.750 | 0.000 | 0.750 | 0.500 |

### breast_cancer (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 1.000 | 1.000 | 1.000 |
| `conditional_mean_shift` | 0.250 | -0.875 | 0.500 | 0.500 |
| `pairwise_corr_shift` | -0.500 | -0.500 | 0.500 | 0.500 |
| `shape_change_q` | 0.000 | -0.750 | 0.500 | 0.500 |
| `po_gap` | 0.500 | -0.875 | 0.500 | 0.875 |
| `rf_domain` | 0.500 | -1.000 | 0.500 | 0.500 |

### digits (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 0.875 | 1.000 | 1.000 |
| `conditional_mean_shift` | 0.500 | -1.000 | 0.500 | 0.500 |
| `pairwise_corr_shift` | -0.500 | -0.500 | 0.500 | 0.500 |
| `shape_change_q` | -0.500 | -0.500 | 0.500 | 0.500 |
| `po_gap` | -0.250 | -0.500 | 0.500 | 0.375 |
| `rf_domain` | 0.500 | -1.000 | 0.500 | 0.500 |

### pendigits (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 0.875 | 1.000 | 1.000 |
| `conditional_mean_shift` | 0.500 | -1.000 | 0.500 | 0.500 |
| `pairwise_corr_shift` | -0.500 | -0.500 | 0.500 | 0.500 |
| `shape_change_q` | -0.250 | -0.625 | 0.500 | 0.500 |
| `po_gap` | 0.500 | -1.000 | 0.500 | 0.500 |
| `rf_domain` | 0.875 | -0.625 | 0.500 | 0.500 |

### satimage (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 0.875 | 1.000 | 1.000 |
| `conditional_mean_shift` | 0.250 | -0.875 | 0.500 | 0.500 |
| `pairwise_corr_shift` | -0.500 | -0.500 | 0.500 | 0.500 |
| `shape_change_q` | -0.250 | -0.625 | 0.500 | 0.500 |
| `po_gap` | 0.500 | -1.000 | 0.500 | 0.500 |
| `rf_domain` | 0.625 | -0.875 | 0.500 | 0.500 |

## Reading

- **Strength shift** is the only score that is *signed* for redundancy:
  strengthen vs weaken is recoverable from `sign(A_j)`.
- **CMS / PO-gap / pairwise corr** recover *that a relationship moved*, including flips,
  so they over-rank flip coordinates when the question is dependence intensity.
- **Q-fixed shape term** isolates the flip / mechanism component under a common design `Q`,
  complementary to `Var_Q(m1)-Var_Q(m0)`.

