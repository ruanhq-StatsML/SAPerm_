# FSDS Thompson sampling on TabReD

Applies block-wise FSDS penalties inside Thompson sampling on industrial TabReD
tables (Yandex / ICLR 2025). Tasks are run as **classification**:

| Dataset | Native task | Blocks | Notes |
| --- | --- | --- | --- |
| `weather` | regression (temperature) | 4 contiguous quarters | median-binarized |
| `sberbank-housing` | regression (price) | 4 VIMP clusters | median-binarized |
| `ecom-offers` | classification | 4 contiguous quarters | native label |
| `homesite-insurance` | classification | 4 VIMP clusters | native label |

Rows are sorted by `timestamp` so the stream follows the paper's temporal
order. Data come from the public Hugging Face mirror (no Kaggle token).

## Run

```bash
# recommended first pass
python3 -m experiments.fsds_tabred weather ecom-offers --max-rows 8000

# full suite
python3 -m experiments.fsds_tabred
```

Policies compared per dataset:

- `fsds_vimp` — Gaussian TS + `sum_VIMP` block share (`λ=1`)
- `fsds_logo` — Gaussian TS + leave-one-group PO risk share
- `ts_flat` — Gaussian TS with flat weights (`λ=0`)
- `boot_fsds_vimp` — Bootstrap TS + `sum_VIMP`

Each arm is a feature block. The online reward model is an RF classifier fit on
the reference window **using only the selected block** (`keep_selected`). FSDS
subtracts `λ · w_block` from the Thompson draw, so high-shift blocks are
avoided. Reward is negative Brier.

Results land in `results/BOARD.md` and `results/<dataset>/`.
