# Dependence Strength Shift — Ranking Recovery

Estimand: for each feature `j`,

```text
η²_{t,j} ≈ 1 - MSE_t(X_j, m_{t,j}) / Var_t(X_j)   # held-out correlation ratio
A_j      = η²_{1,j} - η²_{0,j}
```

This matches `dependence_strength_shift` in `OOD_vimp_wrapper` / `ood_vimp.py`:
per-feature T-learner for `m_0, m_1`, held-out η², plus `po_gap` as a
*relationship-change* companion (not a strength score).

## What Spearman is measuring

Spearman ρ is the Pearson correlation of **ranks**:

```text
Spearman(A)     = corr( rank(oracle A_j), rank(estimated A_j) )
Spearman(|A|)   = corr( rank(|oracle A_j|), rank(|estimated A_j|) )
```

- **Spearman(A)** answers: did we recover the *signed* order
  (most-weakened → … → most-strengthened)? This is the DSS question.
- **Spearman(|A|)** answers: did we recover *which* features moved most,
  ignoring strengthen vs weaken. Flip has oracle |A|=0, so |A|-ranking
  is a different task and is closer to what CMS / PO-gap optimize.
- With only 3 responses, Spearman is coarse (few distinct ranks).
  Here we use **6 graded oracle levels** so ρ has resolution:

| Response | target coef | oracle A = Δη² |
| --- | ---: | ---: |
| `weaken_hard` | 0.20 | -0.462 |
| `weaken_soft` | 0.55 | -0.268 |
| `near_null` | 0.90 | -0.052 |
| `strengthen_soft` | 1.40 | +0.162 |
| `strengthen_hard` | 2.20 | +0.329 |
| `flip` | -1.00 | +0.000 |

Population values assume `X = c·F + ε` with unit-variance F,ε.
Flip keeps `|c|=1` so oracle `A=0` while the map reverses.

## Synthetic graded (response-block ranking)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.971 | 0.814 | 0.950 | 1.000 |
| `conditional_mean_shift` | 0.357 | -0.086 | 0.800 | 0.400 |
| `pairwise_corr_shift` | 0.157 | 0.000 | 0.800 | 0.400 |
| `shape_change_q` | 0.714 | -0.443 | 0.800 | 0.400 |
| `po_gap` | 0.329 | -0.100 | 0.800 | 0.400 |
| `rf_domain` | 0.200 | 0.443 | 0.800 | 0.400 |

Full-p Spearman (includes inert noise features with oracle A=0):

| Method | Spearman(A) full | Spearman(|A|) full |
| --- | ---: | ---: |
| `strength_shift` | 0.931 | 0.889 |
| `conditional_mean_shift` | 0.042 | 0.562 |
| `pairwise_corr_shift` | -0.029 | 0.582 |
| `shape_change_q` | 0.204 | 0.459 |
| `po_gap` | 0.032 | 0.524 |
| `rf_domain` | -0.019 | 0.721 |

### Flip separation (relationship reverses, strength fixed)

- mean `|strength_shift|` on flip coords: **0.043**
- mean `conditional_mean_shift` on flip coords: **3.172**
- mean Q-fixed `shape_change` on flip coords: **3.355**
- mean `Corr_Q(g0, g1)` on flip coords: **-0.957** (near `-1` = pure flip)

## UCI covariates + graded factor responses

Each table is split into reference / target. A PCA factor of the real features
drives six appended responses with the oracle ladder above. Ranking metrics below
are computed on that **response block** (the scientifically relevant coordinates).

### Cross-dataset summary (`strength_shift`)

| Dataset | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `wine` | 0.729 | 0.543 | 0.800 | 0.700 |
| `breast_cancer` | 0.929 | 0.771 | 1.000 | 1.000 |
| `diabetes` | 0.929 | 0.757 | 0.900 | 1.000 |
| `digits` | 0.914 | 0.600 | 0.900 | 0.900 |
| `pendigits` | 0.986 | 0.957 | 0.950 | 0.950 |
| `satimage` | 0.957 | 0.843 | 0.900 | 0.950 |
| `ionosphere` | 0.686 | 0.343 | 0.850 | 0.850 |
| `spambase` | 0.800 | 0.786 | 1.000 | 0.800 |
| `waveform` | 1.000 | 0.929 | 0.900 | 0.950 |
| `optdigits` | 0.857 | 0.543 | 0.850 | 0.850 |

### Per-dataset method comparison

#### wine (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.729 | 0.543 | 0.800 | 0.700 |
| `conditional_mean_shift` | 0.343 | -0.200 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.129 | 0.043 | 0.800 | 0.400 |
| `shape_change_q` | 0.186 | -0.286 | 0.800 | 0.400 |
| `po_gap` | 0.300 | -0.243 | 0.800 | 0.700 |
| `rf_domain` | 0.371 | 0.271 | 0.800 | 0.400 |

#### breast_cancer (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.929 | 0.771 | 1.000 | 1.000 |
| `conditional_mean_shift` | 0.586 | -0.257 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.386 | 0.029 | 0.800 | 0.400 |
| `shape_change_q` | 0.129 | -0.429 | 0.800 | 0.400 |
| `po_gap` | 0.529 | -0.271 | 0.800 | 0.800 |
| `rf_domain` | 0.457 | 0.086 | 0.800 | 0.400 |

#### diabetes (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.929 | 0.757 | 0.900 | 1.000 |
| `conditional_mean_shift` | 0.257 | 0.043 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.143 | 0.129 | 0.800 | 0.400 |
| `shape_change_q` | -0.043 | -0.300 | 0.800 | 0.400 |
| `po_gap` | 0.300 | -0.143 | 0.800 | 0.450 |
| `rf_domain` | 0.386 | 0.200 | 0.800 | 0.400 |

#### digits (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.914 | 0.600 | 0.900 | 0.900 |
| `conditional_mean_shift` | 0.371 | -0.043 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.271 | 0.043 | 0.800 | 0.400 |
| `shape_change_q` | 0.057 | -0.143 | 0.800 | 0.400 |
| `po_gap` | 0.257 | 0.229 | 0.850 | 0.700 |
| `rf_domain` | 0.457 | 0.200 | 0.800 | 0.400 |

#### pendigits (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.986 | 0.957 | 0.950 | 0.950 |
| `conditional_mean_shift` | 0.214 | 0.057 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.257 | 0.100 | 0.800 | 0.400 |
| `shape_change_q` | 0.114 | -0.157 | 0.800 | 0.400 |
| `po_gap` | 0.229 | -0.129 | 0.800 | 0.600 |
| `rf_domain` | 0.314 | 0.371 | 0.800 | 0.400 |

#### satimage (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.957 | 0.843 | 0.900 | 0.950 |
| `conditional_mean_shift` | 0.257 | 0.043 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.157 | 0.100 | 0.800 | 0.400 |
| `shape_change_q` | -0.157 | -0.186 | 0.800 | 0.400 |
| `po_gap` | 0.243 | -0.286 | 0.800 | 0.600 |
| `rf_domain` | 0.229 | 0.143 | 0.800 | 0.400 |

#### ionosphere (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.686 | 0.343 | 0.850 | 0.850 |
| `conditional_mean_shift` | 0.157 | -0.343 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.171 | 0.057 | 0.800 | 0.400 |
| `shape_change_q` | 0.243 | -0.314 | 0.800 | 0.400 |
| `po_gap` | 0.186 | -0.186 | 0.800 | 0.650 |
| `rf_domain` | 0.343 | 0.100 | 0.800 | 0.400 |

#### spambase (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.800 | 0.786 | 1.000 | 0.800 |
| `conditional_mean_shift` | 0.600 | -0.186 | 0.800 | 0.400 |
| `pairwise_corr_shift` | 0.329 | -0.200 | 0.800 | 0.400 |
| `shape_change_q` | 0.443 | -0.300 | 0.800 | 0.400 |
| `po_gap` | 0.229 | -0.114 | 0.850 | 0.750 |
| `rf_domain` | 0.657 | 0.100 | 0.800 | 0.400 |

#### waveform (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 1.000 | 0.929 | 0.900 | 0.950 |
| `conditional_mean_shift` | 0.243 | 0.071 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.143 | 0.129 | 0.800 | 0.400 |
| `shape_change_q` | 0.371 | -0.186 | 0.800 | 0.400 |
| `po_gap` | 0.257 | -0.443 | 0.800 | 0.700 |
| `rf_domain` | 0.314 | 0.371 | 0.800 | 0.400 |

#### optdigits (mean over 4 reps)

| Method | Spearman(A) | Spearman(|A|) | Precision@k | Sign agree |
| --- | ---: | ---: | ---: | ---: |
| `strength_shift` | 0.857 | 0.543 | 0.850 | 0.850 |
| `conditional_mean_shift` | 0.257 | 0.043 | 0.800 | 0.400 |
| `pairwise_corr_shift` | -0.100 | 0.114 | 0.800 | 0.400 |
| `shape_change_q` | -0.029 | -0.100 | 0.800 | 0.400 |
| `po_gap` | 0.357 | -0.114 | 0.800 | 0.750 |
| `rf_domain` | 0.357 | 0.300 | 0.800 | 0.400 |

## Argument: why these Spearman numbers support DSS

1. **Signed order is the DSS estimand.** Spearman(A) rewards putting
   `weaken_hard` below `weaken_soft` below `near_null` below strengthens,
   and keeping flip near zero. CMS / pairwise / PO-gap are unsigned gap
   magnitudes, so they systematically mis-order flip and cannot achieve
   high sign agreement (chance ≈ 0.5 on binary strengthen/weaken).
2. **High Spearman(A) with graded levels is not automatic.** With 6 oracle
   ranks, a random permutation has E[ρ]=0. Recovering ρ ≳ 0.8 means the
   held-out η² differences track the population ladder, not just signs.
3. **Spearman(|A|) can lag Spearman(A).** Magnitude ranking confuses flip
   (oracle |A|=0 but large CMS) and soft near-null shifts; DSS is primarily
   a *signed* monitor. Prefer Spearman(A) + sign agreement as the headline.
4. **Full-p Spearman is a stress test.** Inert UCI/noise columns have oracle
   A=0 but finite-sample η² noise; ρ drops relative to the response block.
   That is expected and does not invalidate signed recovery on the moved coords.
5. **Complementarity.** Use CMS / `po_gap` / Q-shape to detect *that* a
   conditional map moved (including flips); use `A_j` to say whether
   reconstructibility / redundancy got stronger or weaker.

