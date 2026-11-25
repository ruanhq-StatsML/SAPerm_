# RecSys monitor — OnlineRFPerm × CFPerm landing

Land the three papers on a recommender interaction stream:

- **OnlineRFPerm** — sequential rating-MSE monitor with online FDR → *when* to refresh
- **CFPerm** — user / item / context / review block attribution → *what* drifted
- **Sentiment RFPerm** — review block as the text modality (benign vs refresh-worthy)

Full phased plan: [`ROADMAP.md`](ROADMAP.md).

## Quick start

```bash
# from repo root
python -m experiments.recsys_monitor --quick
python -m experiments.recsys_monitor --n-rep 4
python -m pytest experiments/recsys_monitor -q
```

`--quick` uses 1 replicate and Ridge for a fast smoke path. The default board
uses RandomForest + ADDIS and writes `results/BOARD.md`.

## Pipeline

```
D_ref ──fit f_ref──► E_ref
          │
burn-in batches ──► history {T_j = MSE_j − E_ref}
          │
incoming batch ──► T, empirical p, SAFFRON/LORD/fixed-α
          │
       reject? ──no──► append T, continue
          │yes
          ├── CFPerm block ranking (user/item/context/review)
          ├── emit RefreshTicket
          └── retrain on recent window + full state reset
```

## Scenarios

| Name | Planted shift after batch 8 |
| --- | --- |
| `none` | stationary (false-alarm check) |
| `concept` | item-taste coefficients flip (+ mild review shift) |
| `covariate` | user demographics mean-shift |
| `both` | concept + covariate |

## Relation to the rest of the repo

| Sibling | Shared idea |
| --- | --- |
| `experiments/uplift_monitor` | batch monitor + refresh; here the outcome is rating not uplift |
| `tabular-drift-attribution` | MovieLens user/item/context blocks — same vocabulary |
| `agents/shift-detect` | method catalog points RFPerm / OnlineRFPerm / CFPerm here next |
