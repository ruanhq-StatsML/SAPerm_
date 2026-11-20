# Substitution under missing / sparse data

# v8 board (same-day recovery micro-pass + adaptive escalate + dual-look)

Freshly recovered / unredacted critical docs are read the same day (no
next-day lag). Adaptive escalation, reopen, dual-look, critical lane retained.
False-IC stays at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 345 | 0.97 | 0.97 | 0.00 | 20 | 0 | 0 | 202.9 | 55.3 |
| dense | rules_blind | 389 | 0.99 | 0.99 | 0.00 | 7 | 0 | 0 | 182.5 | 50.8 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 284 | 0.81 | 0.81 | 0.00 | 26 | 0 | 0 | 204.2 | 41.0 |
| sparse | rules_blind | 272 | 0.75 | 0.75 | 0.00 | 25 | 0 | 0 | 198.5 | 36.2 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 377 | 1.00 | 1.00 | 0.00 | 23 | 36 | 0 | 191.0 | 67.0 |
| missing_docs | rules_blind | 405 | 0.99 | 0.99 | 0.00 | 40 | 36 | 0 | 175.3 | 66.9 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 399 | 0.97 | 0.97 | 0.00 | 32 | 0 | 44 | 175.4 | 62.7 |
| missing_fields | rules_blind | 434 | 0.99 | 0.99 | 0.00 | 28 | 0 | 46 | 163.6 | 65.5 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 341 | 0.81 | 0.81 | 0.00 | 46 | 20 | 14 | 170.1 | 49.1 |
| sparse_missing | rules_blind | 333 | 0.72 | 0.72 | 0.00 | 46 | 20 | 14 | 156.2 | 42.1 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Strong partial substitute for triage + evidence chase under capacity constraints — still not a full substitute for primary-source accountability when escalation fails.**

### Evidence from this board

1. **Dense.** Controlled recall=0.97 vs flood=0.78 at ~1/3 coverage; confirmed/1k 203 vs 52; false-IC=0.

2. **Sparse.** Controlled 0.81 beats flood 0.49 (v3 freeze was 0.36).

3. **Missing docs.** Controlled recall_all=1.00 after recovered=36 + same-day reads; false-IC=0. (Ceiling is recovery success — failed alt-data still cannot be invented.)

4. **Missing fields.** Controlled 0.97 after 44 unredacts + same-day re-read; model-only was 0.33.

5. **sparse_missing.** Controlled 0.81 (v7 was 0.69) via same-day recovery micro-pass; false-IC=0.

### Operating rule

Score → allocate → integrity-gate → escalate (persist) → **same-day critical micro-pass** → dual-look → reopen. Soft-halt never zeros the critical lane. `recall_all` under missingness remains the integrity metric.
