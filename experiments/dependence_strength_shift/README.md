# Dependence Strength Shift

Label-free monitor for whether `X_j | X_{-j}` got **stronger or weaker** across batches.

```text
m_{t,j}(X_{-j}) = E_t[X_j | X_{-j}]
η²_{t,j}        = Var_t(m_{t,j}) / Var_t(X_j)
A_j             = η²_{1,j} - η²_{0,j}
```

`A_j` is signed. Relationship change (`E[(m_1-m_0)^2]`) is complementary: a pure sign flip has large CMS and near-zero `A_j`.

## Run

```bash
python -m experiments.dependence_strength_shift --reps 4
python -m experiments.dependence_strength_shift --uci-protocol residual --uci wine breast_cancer
python -m experiments.dependence_strength_shift --skip-uci --reps 5
python -m pytest experiments/dependence_strength_shift/test_dependence_strength_shift.py -q
```

Default UCI protocol appends **six graded** factor-driven responses
(hard/soft weaken, near-null, soft/hard strengthen, flip) with known population
`η²`, then scores Spearman ranking recovery on that response block.
`--uci-protocol residual` instead rewrites real columns via `m̂ + γ·residual`.

Headline metric: **Spearman(A)** = rank correlation between oracle signed
`Δη²` and estimated `strength_shift`, compared to **PO-risk ranking**
(`po_gap = R(0)-R(τ)`).

```bash
python -m experiments.dependence_strength_shift.write_latex
```

Writes `latex/dependence_strength_shift.tex` (also copied under `results/`).
See `results/BOARD.md` for the markdown board.

## API

```python
from OOD_vimp_wrapper import dependence_strength_shift, dependence_strength_shift_q

out = dependence_strength_shift(X_ref, X_new)   # η²_0, η²_1, A_j, CMS, po_gap
q   = dependence_strength_shift_q(X_ref, X_new) # Q-fixed strength vs shape split
```

## Quant factor residuals

In multi-factor panels the relevant object is usually the **residual** after a
common component (market / PCA / style) is removed:

```text
F_j = β_j' G + ε_j
η²_j(ε) = Var(E[ε_j | ε_{-j}]) / Var(ε_j)
A_j     = η²_1(ε) - η²_0(ε)     # residual redundancy up/down across windows
```

Pairwise residual correlation is the linear special case; DSS is the
multivariate residual-interaction strength; `po_gap` ranks residual
*relationship* changes (including sign flips).

```python
from experiments.dependence_strength_shift.factor_residuals import (
    factor_residual_dependence_shift,
)

out = factor_residual_dependence_shift(F_ref, F_new, n_common=1)
# out["strength_shift"], out["po_gap"], out["residual_corr_shift"],
# out["incremental_shift"]  # Δ V_{j←k} unique residual information
```

```bash
python -m experiments.dependence_strength_shift.run_factor_demo
```
