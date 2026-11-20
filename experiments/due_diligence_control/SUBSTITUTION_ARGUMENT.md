# Substitution under missing / sparse data

# v20 board (stop filler once readable critical found)

Once every available non-redacted planted risk is found, stop releasing
filler even before IC clearance. Coverage collapses further
(missing_fields 334→149) with recall/false-IC still 1.00 / 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 241 | 0.82 | 0.82 | 0.70 | 0 | 0 | 0 | 244.8 | 21.9 |
| dense | controlled_v1 | 318 | 0.78 | 0.78 | 0.80 | 0 | 0 | 0 | 176.1 | 16.0 |
| dense | controlled_v2 | 316 | 0.82 | 0.82 | 0.00 | 58 | 0 | 0 | 186.7 | 40.2 |
| dense | controlled | 305 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 236.1 | 57.5 |
| dense | rules_blind | 305 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 236.1 | 57.5 |
| dense | model_only | 241 | 0.82 | 0.82 | 0.70 | 0 | 0 | 0 | 244.8 | 21.9 |
| sparse | flood | 123 | 0.78 | 0.78 | 0.77 | 0 | 0 | 0 | 455.3 | 21.8 |
| sparse | controlled_v1 | 123 | 0.78 | 0.78 | 0.77 | 0 | 0 | 0 | 455.3 | 21.8 |
| sparse | controlled_v2 | 197 | 0.71 | 0.71 | 0.00 | 39 | 0 | 0 | 258.9 | 26.5 |
| sparse | controlled | 104 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 692.3 | 59.4 |
| sparse | rules_blind | 104 | 1.00 | 1.00 | 0.00 | 0 | 0 | 0 | 692.3 | 59.4 |
| sparse | model_only | 123 | 0.78 | 0.78 | 0.77 | 0 | 0 | 0 | 455.3 | 21.8 |
| missing_docs | flood | 315 | 0.78 | 0.39 | 0.60 | 0 | 0 | 0 | 88.9 | 10.5 |
| missing_docs | controlled_v1 | 258 | 0.83 | 0.42 | 0.22 | 0 | 0 | 0 | 116.3 | 17.8 |
| missing_docs | controlled_v2 | 255 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 117.6 | 14.4 |
| missing_docs | controlled | 196 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 367.3 | 67.9 |
| missing_docs | rules_blind | 196 | 1.00 | 1.00 | 0.00 | 0 | 36 | 0 | 367.3 | 67.7 |
| missing_docs | model_only | 315 | 0.78 | 0.39 | 0.60 | 0 | 0 | 0 | 88.9 | 10.5 |
| missing_fields | flood | 184 | 0.26 | 0.26 | 1.00 | 0 | 0 | 0 | 103.3 | 6.1 |
| missing_fields | controlled_v1 | 193 | 0.25 | 0.25 | 1.00 | 0 | 0 | 0 | 93.3 | 5.5 |
| missing_fields | controlled_v2 | 207 | 0.25 | 0.25 | 0.00 | 23 | 0 | 0 | 87.0 | 8.1 |
| missing_fields | controlled | 149 | 1.00 | 1.00 | 0.00 | 0 | 0 | 48 | 483.2 | 71.0 |
| missing_fields | rules_blind | 149 | 1.00 | 1.00 | 0.00 | 0 | 0 | 47 | 483.2 | 70.0 |
| missing_fields | model_only | 184 | 0.26 | 0.26 | 1.00 | 0 | 0 | 0 | 103.3 | 6.1 |
| sparse_missing | flood | 81 | 0.63 | 0.46 | 0.50 | 0 | 0 | 0 | 407.4 | 15.9 |
| sparse_missing | controlled_v1 | 93 | 0.60 | 0.43 | 0.67 | 0 | 0 | 0 | 333.3 | 14.0 |
| sparse_missing | controlled_v2 | 115 | 0.58 | 0.42 | 0.00 | 5 | 0 | 0 | 260.9 | 15.1 |
| sparse_missing | controlled | 101 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 712.9 | 67.8 |
| sparse_missing | rules_blind | 101 | 1.00 | 1.00 | 0.00 | 0 | 20 | 14 | 712.9 | 67.8 |
| sparse_missing | model_only | 81 | 0.63 | 0.46 | 0.50 | 0 | 0 | 0 | 407.4 | 15.9 |
| hostile | flood | 53 | 0.43 | 0.18 | 0.00 | 0 | 0 | 0 | 245.3 | 7.0 |
| hostile | controlled_v1 | 53 | 0.43 | 0.18 | 0.00 | 0 | 0 | 0 | 245.3 | 7.0 |
| hostile | controlled_v2 | 53 | 0.43 | 0.18 | 0.00 | 1 | 0 | 0 | 245.3 | 6.0 |
| hostile | controlled | 82 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 878.0 | 73.5 |
| hostile | rules_blind | 82 | 1.00 | 1.00 | 0.00 | 0 | 42 | 15 | 878.0 | 73.5 |
| hostile | model_only | 53 | 0.43 | 0.18 | 0.00 | 0 | 0 | 0 | 245.3 | 7.0 |

## Verdict

**Strong partial substitute for triage + evidence chase under capacity constraints.**

### Ladder (controlled)

| Regime | recall | flood | coverage | conf/1k | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | **1.00** | 0.82 | 305 | 236.1 | 57.5 |
| sparse | **1.00** | 0.78 | 104 | 692.3 | 59.4 |
| missing_docs | **1.00** | 0.39 | 196 | 367.3 | 67.9 |
| missing_fields | **1.00** | 0.26 | 149 | 483.2 | 71.0 |
| sparse_missing | **1.00** | 0.46 | 101 | 712.9 | 67.8 |
| hostile | **1.00** | 0.18 | 82 | 878.0 | 73.5 |

### Operating rule

Score → allocate → integrity-gate → escalate → dual-look → lean deadline_pull → last-mile → clearance IC → prefer critical release → **stop filler once readable critical found**.

