# Substitution under missing / sparse data

# v16 board (v15 + deadline_pull + hostile regime)

When days_to_ic ≤ 6, force-release stranded available critical and take
partner looks before the calendar end window. Adds a hostile stress regime
(very thin + high missing/redact). False-IC stays at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | -0.4 |
| dense | controlled_v1 | 331 | 0.82 | 0.82 | 0.81 | 0 | 0 | 0 | 178.2 | 16.6 |
| dense | controlled_v2 | 326 | 0.83 | 0.83 | 0.00 | 50 | 0 | 0 | 184.0 | 42.2 |
| dense | controlled | 358 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 201.1 | 57.6 |
| dense | rules_blind | 357 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 201.7 | 57.6 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | -0.4 |
| sparse | flood | 255 | 0.49 | 0.49 | 0.73 | 0 | 0 | 0 | 137.3 | 11.6 |
| sparse | controlled_v1 | 245 | 0.49 | 0.49 | 0.64 | 0 | 0 | 0 | 142.9 | 13.5 |
| sparse | controlled_v2 | 245 | 0.57 | 0.57 | 0.00 | 13 | 0 | 0 | 167.3 | 27.7 |
| sparse | controlled | 284 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 253.5 | 57.8 |
| sparse | rules_blind | 284 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 253.5 | 57.8 |
| sparse | model_only | 255 | 0.49 | 0.49 | 0.73 | 0 | 0 | 0 | 137.3 | 11.6 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 325 | 0.89 | 0.44 | 0.20 | 0 | 0 | 0 | 98.5 | 16.0 |
| missing_docs | controlled_v2 | 314 | 0.78 | 0.39 | 0.00 | 16 | 0 | 0 | 89.2 | 13.7 |
| missing_docs | controlled | 370 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 194.6 | 66.7 |
| missing_docs | rules_blind | 367 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 196.2 | 67.1 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.29 | 0.29 | 1.00 | 0 | 0 | 0 | 32.4 | 0.8 |
| missing_fields | controlled_v1 | 288 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 69.4 | 4.7 |
| missing_fields | controlled_v2 | 293 | 0.28 | 0.28 | 0.00 | 4 | 0 | 0 | 68.3 | 8.3 |
| missing_fields | controlled | 383 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 188.0 | 70.8 |
| missing_fields | rules_blind | 393 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 183.2 | 70.5 |
| missing_fields | model_only | 648 | 0.29 | 0.29 | 1.00 | 0 | 0 | 0 | 32.4 | 0.8 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 290 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 69.0 | 5.7 |
| sparse_missing | controlled_v2 | 279 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.9 | 5.9 |
| sparse_missing | controlled | 376 | 1.00 | 1.00 | 0.00 | 1 | 20 | 13 | 191.5 | 66.0 |
| sparse_missing | rules_blind | 372 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 193.5 | 66.8 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| hostile | flood | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled_v1 | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled_v2 | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |
| hostile | controlled | 245 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 293.9 | 72.7 |
| hostile | rules_blind | 249 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 289.2 | 72.5 |
| hostile | model_only | 160 | 0.03 | 0.01 | 0.00 | 0 | 0 | 0 | 6.2 | -1.1 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled recall_all)

| Regime | v14 | v15 | **v16** | flood | false-IC | abstain | deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1.00 | 1.00 | **1.00** | 0.78 | 0 | 0 | 7/11 |
| sparse | 1.00 | 1.00 | **1.00** | 0.49 | 0 | 0 | 38/78 |
| missing_docs | 1.00 | 1.00 | **1.00** | 0.40 | 0 | 0 | 2/3 |
| missing_fields | 1.00 | 1.00 | **1.00** | 0.29 | 0 | 0 | 3/7 |
| sparse_missing | 1.00 | 1.00 | **1.00** | 0.32 | 0 | 1 | 27/57 |
| hostile | — | — | **1.00** | 0.01 | 0 | 0 | 14/31 |

### Operating rule

Score → allocate → integrity-gate → escalate → dual-look → **deadline_pull** (days_to_ic≤6) → last-mile → clearance IC.

