# Substitution under missing / sparse data

# v11 board (budget transfer + soft critical-first + v10 stack)

IC-ready / closed deals donate spare coverage slots to deals with unread
critical. Soft critical-first in dense rooms. False-IC stays at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 344 | 0.97 | 0.97 | 0.00 | 9 | 0 | 0 | 203.5 | 51.2 |
| dense | rules_blind | 344 | 0.97 | 0.97 | 0.00 | 9 | 0 | 0 | 203.5 | 51.2 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 274 | 0.79 | 0.79 | 0.00 | 13 | 0 | 0 | 208.0 | 38.8 |
| sparse | rules_blind | 267 | 0.74 | 0.74 | 0.00 | 13 | 0 | 0 | 198.5 | 36.1 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 377 | 1.00 | 1.00 | 0.00 | 15 | 36 | 0 | 191.0 | 63.1 |
| missing_docs | rules_blind | 384 | 1.00 | 1.00 | 0.00 | 19 | 36 | 0 | 187.5 | 63.8 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 412 | 0.99 | 0.99 | 0.00 | 21 | 0 | 45 | 172.3 | 61.7 |
| missing_fields | rules_blind | 418 | 0.97 | 0.97 | 0.00 | 18 | 0 | 46 | 167.5 | 62.3 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 341 | 0.81 | 0.81 | 0.00 | 23 | 20 | 14 | 170.1 | 47.8 |
| sparse_missing | rules_blind | 341 | 0.81 | 0.81 | 0.00 | 23 | 20 | 14 | 170.1 | 47.8 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled recall_all)

| Regime | v3 | **v11** | flood | false-IC |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.79 | **0.97** | 0.78 | 0 |
| sparse | 0.36 | **0.79** | 0.49 | 0 |
| missing_docs | 0.61 | **1.00** | 0.40 | 0 |
| missing_fields | 0.60 | **0.99** | 0.33 | 0 |
| sparse_missing | 0.50 | **0.81** | 0.32 | 0 |

### Operating rule

Score → allocate (critical-first, even-effort, budget transfer) → integrity-gate → escalate → same-day micro-pass → dual-look → reopen.
