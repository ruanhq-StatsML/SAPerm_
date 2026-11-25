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

Default UCI protocol appends three factor-driven responses (weaken / strengthen / flip)
with known population `η²`, then scores ranking recovery on that response block.
`--uci-protocol residual` instead rewrites real columns via `m̂ + γ·residual`.

Results land in `results/BOARD.md` and `results/ranking_recovery.json`.

## API

```python
from OOD_vimp_wrapper import dependence_strength_shift, dependence_strength_shift_q

out = dependence_strength_shift(X_ref, X_new)   # η²_0, η²_1, A_j, CMS, po_gap
q   = dependence_strength_shift_q(X_ref, X_new) # Q-fixed strength vs shape split
```
