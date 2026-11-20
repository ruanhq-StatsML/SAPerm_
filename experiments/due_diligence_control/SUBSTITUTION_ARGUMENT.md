# Substitution under missing / sparse data

# v9 board (delayed abstain + same-day recovery + adaptive escalate)

Keep hunting after first integrity refusal (strikes before hard abstain).
Same-day recovery micro-pass, adaptive escalation, dual-look, critical lane.
False-IC stays at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 345 | 0.97 | 0.97 | 0.00 | 11 | 0 | 0 | 202.9 | 54.3 |
| dense | rules_blind | 307 | 0.99 | 0.99 | 0.00 | 3 | 0 | 0 | 231.3 | 44.8 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 275 | 0.79 | 0.79 | 0.00 | 13 | 0 | 0 | 207.3 | 38.8 |
| sparse | rules_blind | 266 | 0.76 | 0.76 | 0.00 | 10 | 0 | 0 | 206.8 | 37.0 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 375 | 0.99 | 0.99 | 0.00 | 12 | 36 | 0 | 190.7 | 63.3 |
| missing_docs | rules_blind | 393 | 0.97 | 0.97 | 0.00 | 15 | 36 | 0 | 178.1 | 59.1 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 399 | 0.97 | 0.97 | 0.00 | 21 | 0 | 44 | 175.4 | 62.0 |
| missing_fields | rules_blind | 426 | 1.00 | 1.00 | 0.00 | 6 | 0 | 46 | 169.0 | 62.8 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 341 | 0.81 | 0.81 | 0.00 | 21 | 20 | 14 | 170.1 | 48.8 |
| sparse_missing | rules_blind | 343 | 0.82 | 0.82 | 0.00 | 24 | 20 | 14 | 172.0 | 48.3 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**
Not a full substitute for partner accountability when escalation fails to recover evidence.

### Ladder wins (controlled vs flood / early versions)

| Regime | v3 freeze | v9 controlled | flood | false-IC |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.79 | **0.97** | 0.78 | 0 |
| sparse | 0.36 | **0.79** | 0.49 | 0 |
| missing_docs | 0.61 | **0.99** | 0.40 | 0 |
| missing_fields | 0.60 | **0.97** | 0.33 | 0 |
| sparse_missing | 0.50 | **0.81** | 0.32 | 0 |

### Operating rule

Score → allocate → integrity-gate (delayed abstain) → escalate (persist) → same-day critical micro-pass → dual-look → reopen. Soft-halt never zeros the critical lane.
