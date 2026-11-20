# Substitution under missing / sparse data

# v19 board (prefer critical release)

When unfound available critical still sits outside live, Poisson release
prefers those docs before filler. Cuts coverage further (sparse 216→124)
while recall_all / false-IC stay at 1.00 / 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 247 | 0.81 | 0.81 | 0.64 | 0 | 0 | 0 | 234.8 | 23.3 |
| dense | controlled_v1 | 322 | 0.81 | 0.81 | 0.68 | 0 | 0 | 0 | 180.1 | 20.8 |
| dense | controlled_v2 | 322 | 0.85 | 0.85 | 0.00 | 48 | 0 | 0 | 189.4 | 41.7 |
| dense | controlled | 324 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 222.2 | 57.3 |
| dense | rules_blind | 324 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 222.2 | 57.3 |
| dense | model_only | 247 | 0.81 | 0.81 | 0.64 | 0 | 0 | 0 | 234.8 | 23.3 |
| sparse | flood | 145 | 0.75 | 0.75 | 0.53 | 0 | 0 | 0 | 372.4 | 26.6 |
| sparse | controlled_v1 | 139 | 0.76 | 0.76 | 0.55 | 0 | 0 | 0 | 395.7 | 26.3 |
| sparse | controlled_v2 | 219 | 0.72 | 0.72 | 0.00 | 39 | 0 | 0 | 237.4 | 35.8 |
| sparse | controlled | 124 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 580.6 | 59.0 |
| sparse | rules_blind | 124 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 580.6 | 59.0 |
| sparse | model_only | 145 | 0.75 | 0.75 | 0.53 | 0 | 0 | 0 | 372.4 | 26.6 |
| missing_docs | flood | 460 | 0.78 | 0.39 | 0.43 | 0 | 0 | 0 | 60.9 | 11.0 |
| missing_docs | controlled_v1 | 274 | 0.83 | 0.42 | 0.18 | 0 | 0 | 0 | 109.5 | 19.7 |
| missing_docs | controlled_v2 | 277 | 0.86 | 0.43 | 0.00 | 40 | 0 | 0 | 111.9 | 16.7 |
| missing_docs | controlled | 269 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 267.7 | 66.7 |
| missing_docs | rules_blind | 268 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 268.7 | 66.7 |
| missing_docs | model_only | 460 | 0.78 | 0.39 | 0.43 | 0 | 0 | 0 | 60.9 | 11.0 |
| missing_fields | flood | 575 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 34.8 | 1.1 |
| missing_fields | controlled_v1 | 267 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 74.9 | 4.9 |
| missing_fields | controlled_v2 | 268 | 0.29 | 0.29 | 0.00 | 11 | 0 | 0 | 78.4 | 8.4 |
| missing_fields | controlled | 334 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 215.6 | 69.2 |
| missing_fields | rules_blind | 332 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 216.9 | 69.7 |
| missing_fields | model_only | 575 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 34.8 | 1.1 |
| sparse_missing | flood | 186 | 0.73 | 0.53 | 0.43 | 0 | 0 | 0 | 204.3 | 20.3 |
| sparse_missing | controlled_v1 | 201 | 0.69 | 0.50 | 0.36 | 0 | 0 | 0 | 179.1 | 21.0 |
| sparse_missing | controlled_v2 | 278 | 0.69 | 0.50 | 0.00 | 40 | 0 | 0 | 129.5 | 19.2 |
| sparse_missing | controlled | 138 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 521.7 | 67.2 |
| sparse_missing | rules_blind | 138 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 521.7 | 67.2 |
| sparse_missing | model_only | 186 | 0.73 | 0.53 | 0.43 | 0 | 0 | 0 | 204.3 | 20.3 |
| hostile | flood | 158 | 0.27 | 0.11 | 0.00 | 0 | 0 | 0 | 50.6 | 3.4 |
| hostile | controlled_v1 | 158 | 0.27 | 0.11 | 0.00 | 0 | 0 | 0 | 50.6 | 3.4 |
| hostile | controlled_v2 | 155 | 0.30 | 0.12 | 0.00 | 2 | 0 | 0 | 58.1 | 3.0 |
| hostile | controlled | 136 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 529.4 | 73.2 |
| hostile | rules_blind | 136 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 529.4 | 73.2 |
| hostile | model_only | 158 | 0.27 | 0.11 | 0.00 | 0 | 0 | 0 | 50.6 | 3.4 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled)

| Regime | recall | flood | coverage | conf/1k | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | **1.00** | 0.81 | 324 | 222.2 | 57.3 |
| sparse | **1.00** | 0.75 | 124 | 580.6 | 59.0 |
| missing_docs | **1.00** | 0.39 | 269 | 267.7 | 66.7 |
| missing_fields | **1.00** | 0.28 | 334 | 215.6 | 69.2 |
| sparse_missing | **1.00** | 0.53 | 138 | 521.7 | 67.2 |
| hostile | **1.00** | 0.11 | 136 | 529.4 | 73.2 |

### Operating rule

Score → allocate → integrity-gate → escalate → dual-look → lean deadline_pull → last-mile → clearance IC → stop filler after IC → **prefer critical release**.

