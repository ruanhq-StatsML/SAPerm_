# Substitution under missing / sparse data

# v10 board (soft critical-first in dense rooms + even-effort + v9 stack)

In crowded rooms (≥8 open) with unread critical, boost critical / penalize filler.
Even-effort score-bumps under-covered deals. Sparse thin streams unchanged.
False-IC stays at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 344 | 0.97 | 0.97 | 0.00 | 10 | 0 | 0 | 203.5 | 53.2 |
| dense | rules_blind | 343 | 0.96 | 0.96 | 0.00 | 13 | 0 | 0 | 201.2 | 50.9 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 275 | 0.79 | 0.79 | 0.00 | 13 | 0 | 0 | 207.3 | 38.8 |
| sparse | rules_blind | 267 | 0.74 | 0.74 | 0.00 | 13 | 0 | 0 | 198.5 | 36.1 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 376 | 1.00 | 1.00 | 0.00 | 13 | 36 | 0 | 191.5 | 63.9 |
| missing_docs | rules_blind | 376 | 1.00 | 1.00 | 0.00 | 14 | 36 | 0 | 191.5 | 64.9 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 416 | 0.99 | 0.99 | 0.00 | 22 | 0 | 45 | 170.7 | 63.8 |
| missing_fields | rules_blind | 417 | 0.97 | 0.97 | 0.00 | 19 | 0 | 46 | 167.9 | 63.2 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 343 | 0.81 | 0.81 | 0.00 | 22 | 20 | 14 | 169.1 | 47.9 |
| sparse_missing | rules_blind | 343 | 0.81 | 0.81 | 0.00 | 22 | 20 | 14 | 169.1 | 47.9 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**
Not a full substitute for partner accountability when escalation fails.

### Ladder (controlled recall_all)

| Regime | v3 | **v10** | flood | false-IC |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.79 | **0.97** | 0.78 | 0 |
| sparse | 0.36 | **0.79** | 0.49 | 0 |
| missing_docs | 0.61 | **1.00** | 0.40 | 0 |
| missing_fields | 0.60 | **0.99** | 0.33 | 0 |
| sparse_missing | 0.50 | **0.81** | 0.32 | 0 |

### Operating rule

Score → allocate (critical-first in dense rooms, even-effort) → integrity-gate (delayed abstain) → escalate (persist) → same-day micro-pass → dual-look → reopen.
