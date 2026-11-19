# Substitution under missing / sparse data

# v2 board (integrity gate)

False-IC driven to 0.00 on controlled across regimes by refusing incomplete IC claims. Compare `controlled` vs `controlled_v1`.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | IC-blocked | Confirmed/1k | Shortage | Match-fail | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 51.9 | 86 | 0 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 162.7 | 0 | 0 | 16.7 |
| dense | controlled | 326 | 0.79 | 0.79 | 0.00 | 52 | 174.8 | 0 | 0 | 39.8 |
| dense | rules_blind | 289 | 0.17 | 0.17 | 0.00 | 6 | 41.5 | 0 | 0 | 3.4 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 51.9 | 86 | 0 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 137.8 | 0 | 0 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 128.5 | 0 | 0 | 10.7 |
| sparse | controlled | 247 | 0.49 | 0.49 | 0.00 | 16 | 141.7 | 0 | 0 | 19.8 |
| sparse | rules_blind | 242 | 0.53 | 0.53 | 0.00 | 15 | 157.0 | 0 | 0 | 20.3 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 137.8 | 0 | 0 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 47.4 | 380 | 0 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 100.9 | 378 | 0 | 19.0 |
| missing_docs | controlled | 318 | 0.83 | 0.42 | 0.00 | 18 | 94.3 | 378 | 0 | 13.7 |
| missing_docs | rules_blind | 289 | 0.44 | 0.22 | 0.00 | 5 | 55.4 | 378 | 0 | 5.4 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 47.4 | 380 | 0 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 37.0 | 168 | 192 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 70.2 | 0 | 181 | 4.7 |
| missing_fields | controlled | 294 | 0.28 | 0.28 | 0.00 | 6 | 68.0 | 0 | 177 | 8.4 |
| missing_fields | rules_blind | 264 | 0.12 | 0.12 | 0.00 | 1 | 34.1 | 0 | 139 | 1.9 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 37.0 | 168 | 192 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 70.3 | 270 | 36 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 68.7 | 270 | 36 | 5.7 |
| sparse_missing | controlled | 281 | 0.33 | 0.24 | 0.00 | 4 | 60.5 | 270 | 36 | 5.9 |
| sparse_missing | rules_blind | 282 | 0.29 | 0.21 | 0.00 | 2 | 53.2 | 270 | 32 | 4.8 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 70.3 | 270 | 36 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled(v2) recall@available=0.79 vs flood=0.78, at coverage 326 vs 1080. Efficiency (confirmed/1k) controlled=174.8 vs flood=51.9. → OK to replace *undifferentiated reading*, not partner sign-off.

2. **v2 integrity gate (dense).** False-IC 0.75→0.00; IC blocked by integrity=52; net 16.7→39.8.

3. **Sparse filings.** Controlled recall@available=0.49 (flood=0.49). With thin streams, rules + model usually match or beat flood on recall *per unit attention*, because noise does not exhaust analyst caps before critical docs arrive.

4. **Missing documents.** recall@all controlled=0.42, flood=0.40. Both collapse toward the availability ceiling (available share ≈ 0.50). v2 blocks incomplete IC (18 refusals; false-IC 0.00 vs flood 0.38). **No method recovers risks that never arrive** — escalate data, don't claim completeness.

5. **Missing / redacted fields.** Model-only recall@available=0.33; controlled=0.28. When risk language is stripped, FinanceGPT-class scores go blind. Rules still save capacity, but **cannot invent missing fields** — humans must pull primary sources.

6. **Worst case (sparse_missing).** Controlled recall@all=0.24, false-IC rate=0.00. Expect low completeness; staged diagnosis fires supply-shortage=270, matching-failure=36 as the honest signal.

### What *can* be substituted

- First-pass screening and prioritization under dense or sparse *observable* rooms
- Fatigue / backlog control so analysts do not over-read filler
- Explicit diagnosis of supply shortage vs matching failure

### What *cannot* be substituted

- Risks in documents that never arrive (missing_docs)
- Risks hidden by redaction / absent fields without alternative evidence
- Legal/partner accountability and primary confirmation interviews
- IC decisions when false-IC risk remains material

### Operating rule

Use controlled DD + FinanceGPT as a **capacity-aware co-pilot**. Treat `recall_all` under missingness as the integrity metric: if availability is low, refuse 'complete diligence' claims and escalate data collection.
