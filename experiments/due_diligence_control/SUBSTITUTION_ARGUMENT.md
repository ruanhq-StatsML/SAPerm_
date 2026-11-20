# Substitution under missing / sparse data

# v18 board (stop filler after IC-ready)

Once a deal is IC-ready, stop releasing filler into the live room
(critical-only if anything remains). Coverage drops sharply on sparse /
hostile while recall_all and false-IC stay at 1.00 / 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 684 | 0.76 | 0.76 | 1.00 | 0 | 0 | 0 | 80.4 | 2.3 |
| dense | controlled_v1 | 330 | 0.74 | 0.74 | 0.65 | 0 | 0 | 0 | 160.6 | 19.8 |
| dense | controlled_v2 | 323 | 0.82 | 0.82 | 0.00 | 41 | 0 | 0 | 182.7 | 41.3 |
| dense | controlled | 353 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 204.0 | 57.4 |
| dense | rules_blind | 353 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 204.0 | 57.4 |
| dense | model_only | 684 | 0.76 | 0.76 | 1.00 | 0 | 0 | 0 | 80.4 | 2.3 |
| sparse | flood | 233 | 0.49 | 0.49 | 0.86 | 0 | 0 | 0 | 150.2 | 11.4 |
| sparse | controlled_v1 | 230 | 0.53 | 0.53 | 0.78 | 0 | 0 | 0 | 165.2 | 13.1 |
| sparse | controlled_v2 | 234 | 0.53 | 0.53 | 0.00 | 18 | 0 | 0 | 162.4 | 22.6 |
| sparse | controlled | 216 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 333.3 | 58.3 |
| sparse | rules_blind | 216 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 333.3 | 58.3 |
| sparse | model_only | 233 | 0.49 | 0.49 | 0.86 | 0 | 0 | 0 | 150.2 | 11.4 |
| missing_docs | flood | 537 | 0.75 | 0.38 | 0.38 | 0 | 0 | 0 | 50.3 | 10.7 |
| missing_docs | controlled_v1 | 305 | 0.75 | 0.38 | 0.50 | 0 | 0 | 0 | 88.5 | 10.9 |
| missing_docs | controlled_v2 | 314 | 0.78 | 0.39 | 0.00 | 16 | 0 | 0 | 89.2 | 13.7 |
| missing_docs | controlled | 364 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 197.8 | 67.3 |
| missing_docs | rules_blind | 373 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 193.0 | 67.6 |
| missing_docs | model_only | 537 | 0.75 | 0.38 | 0.38 | 0 | 0 | 0 | 50.3 | 10.7 |
| missing_fields | flood | 619 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 32.3 | 0.6 |
| missing_fields | controlled_v1 | 287 | 0.26 | 0.26 | 1.00 | 0 | 0 | 0 | 66.2 | 4.2 |
| missing_fields | controlled_v2 | 285 | 0.26 | 0.26 | 0.00 | 4 | 0 | 0 | 66.7 | 7.9 |
| missing_fields | controlled | 382 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 188.5 | 70.6 |
| missing_fields | rules_blind | 393 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 183.2 | 70.3 |
| missing_fields | model_only | 619 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 32.3 | 0.6 |
| sparse_missing | flood | 314 | 0.42 | 0.31 | 1.00 | 0 | 0 | 0 | 70.1 | 5.5 |
| sparse_missing | controlled_v1 | 277 | 0.38 | 0.28 | 1.00 | 0 | 0 | 0 | 72.2 | 5.6 |
| sparse_missing | controlled_v2 | 279 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.9 | 5.9 |
| sparse_missing | controlled | 245 | 1.00 | 1.00 | 0.00 | 1 | 20 | 13 | 293.9 | 66.9 |
| sparse_missing | rules_blind | 232 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 310.3 | 68.2 |
| sparse_missing | model_only | 314 | 0.42 | 0.31 | 1.00 | 0 | 0 | 0 | 70.1 | 5.5 |
| hostile | flood | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled_v1 | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled_v2 | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled | 157 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 458.6 | 73.5 |
| hostile | rules_blind | 157 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 458.6 | 73.5 |
| hostile | model_only | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled)

| Regime | recall | flood | false-IC | coverage | conf/1k | net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | **1.00** | 0.76 | 0 | 353 | 204.0 | 57.4 |
| sparse | **1.00** | 0.49 | 0 | 216 | 333.3 | 58.3 |
| missing_docs | **1.00** | 0.38 | 0 | 364 | 197.8 | 67.3 |
| missing_fields | **1.00** | 0.28 | 0 | 382 | 188.5 | 70.6 |
| sparse_missing | **1.00** | 0.31 | 0 | 245 | 293.9 | 66.9 |
| hostile | **1.00** | 0.01 | 0 | 157 | 458.6 | 73.5 |

### Operating rule

Score → allocate → integrity-gate → escalate → dual-look → lean deadline_pull → last-mile → clearance IC → **stop filler once IC-ready**.

