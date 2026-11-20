# Substitution under missing / sparse data

# v15 board (v14 last-mile + clearance IC)

After last-mile clears available critical evidence, promote allow_ic deals
to IC and reconcile residual abstain stock. False-IC stays at 0; IC slots
fill instead of rotting as incomplete abstains.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | -0.4 |
| dense | controlled_v1 | 331 | 0.82 | 0.82 | 0.81 | 0 | 0 | 0 | 178.2 | 16.6 |
| dense | controlled_v2 | 326 | 0.83 | 0.83 | 0.00 | 50 | 0 | 0 | 184.0 | 42.2 |
| dense | controlled | 358 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 201.1 | 57.0 |
| dense | rules_blind | 358 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 201.1 | 57.0 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | -0.4 |
| sparse | flood | 255 | 0.49 | 0.49 | 0.73 | 0 | 0 | 0 | 137.3 | 11.6 |
| sparse | controlled_v1 | 245 | 0.49 | 0.49 | 0.64 | 0 | 0 | 0 | 142.9 | 13.5 |
| sparse | controlled_v2 | 245 | 0.57 | 0.57 | 0.00 | 13 | 0 | 0 | 167.3 | 27.7 |
| sparse | controlled | 305 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 236.1 | 57.7 |
| sparse | rules_blind | 305 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 236.1 | 57.7 |
| sparse | model_only | 255 | 0.49 | 0.49 | 0.73 | 0 | 0 | 0 | 137.3 | 11.6 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 325 | 0.89 | 0.44 | 0.20 | 0 | 0 | 0 | 98.5 | 16.0 |
| missing_docs | controlled_v2 | 314 | 0.78 | 0.39 | 0.00 | 16 | 0 | 0 | 89.2 | 13.7 |
| missing_docs | controlled | 371 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 194.1 | 66.7 |
| missing_docs | rules_blind | 368 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 195.7 | 67.1 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.29 | 0.29 | 1.00 | 0 | 0 | 0 | 32.4 | 0.8 |
| missing_fields | controlled_v1 | 288 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 69.4 | 4.7 |
| missing_fields | controlled_v2 | 293 | 0.28 | 0.28 | 0.00 | 4 | 0 | 0 | 68.3 | 8.3 |
| missing_fields | controlled | 375 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 192.0 | 70.8 |
| missing_fields | rules_blind | 386 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 186.5 | 70.3 |
| missing_fields | model_only | 648 | 0.29 | 0.29 | 1.00 | 0 | 0 | 0 | 32.4 | 0.8 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 290 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 69.0 | 5.7 |
| sparse_missing | controlled_v2 | 279 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.9 | 5.9 |
| sparse_missing | controlled | 386 | 1.00 | 1.00 | 0.00 | 1 | 20 | 13 | 186.5 | 66.8 |
| sparse_missing | rules_blind | 381 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 189.0 | 67.7 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled recall_all)

| Regime | v3 | v12 | v14 | **v15** | flood | false-IC | abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.79 | 0.97 | 1.00 | **1.00** | 0.78 | 0 | 0 |
| sparse | 0.36 | 0.79 | 1.00 | **1.00** | 0.49 | 0 | 0 |
| missing_docs | 0.61 | 0.99 | 1.00 | **1.00** | 0.40 | 0 | 0 |
| missing_fields | 0.60 | 0.99 | 1.00 | **1.00** | 0.29 | 0 | 0 |
| sparse_missing | 0.50 | 0.82 | 1.00 | **1.00** | 0.32 | 0 | 1 |

### Operating rule

Score → allocate → integrity-gate → escalate → same-day micro-pass → dual-look → reopen → last-mile confirm → **clearance IC** (promote cleared deals; reconcile abstain stock).

