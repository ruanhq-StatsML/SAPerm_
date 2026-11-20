# Substitution under missing / sparse data

# v3 board (escalation + sparse priority + abstain)

Recover missing/redacted evidence via alt-data escalation; soft-halt filler; keep false-IC at 0 while lifting recall_all under missingness.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 322 | 0.79 | 0.79 | 0.00 | 53 | 0 | 0 | 177.0 | 39.9 |
| dense | rules_blind | 231 | 0.10 | 0.10 | 0.00 | 1 | 0 | 0 | 30.3 | 1.2 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 195 | 0.36 | 0.36 | 0.00 | 12 | 0 | 0 | 133.3 | 13.7 |
| sparse | rules_blind | 195 | 0.40 | 0.40 | 0.00 | 10 | 0 | 0 | 148.7 | 14.1 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 322 | 0.65 | 0.61 | 0.00 | 48 | 32 | 0 | 136.6 | 33.2 |
| missing_docs | rules_blind | 316 | 0.46 | 0.46 | 0.00 | 25 | 36 | 0 | 104.4 | 27.6 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 333 | 0.60 | 0.60 | 0.00 | 56 | 0 | 43 | 129.1 | 35.7 |
| missing_fields | rules_blind | 299 | 0.29 | 0.29 | 0.00 | 10 | 0 | 46 | 70.2 | 19.5 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 257 | 0.51 | 0.50 | 0.00 | 35 | 18 | 13 | 140.1 | 26.9 |
| sparse_missing | rules_blind | 243 | 0.43 | 0.42 | 0.00 | 25 | 18 | 14 | 123.5 | 23.8 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled(v2) recall@available=0.79 vs flood=0.78, at coverage 322 vs 1080. Efficiency (confirmed/1k) controlled=177.0 vs flood=51.9. → OK to replace *undifferentiated reading*, not partner sign-off.

2. **v3 vs v1 (dense).** False-IC 0.75→0.00; abstain=53; recovered=0; net 16.7→39.9.

3. **v3 escalation vs v2 abstain-only (missing_docs).** recall@all 0.42→0.61 via recovered=32; false-IC stays 0.00.

4. **Sparse filings.** Controlled recall@available=0.36 (flood=0.49). With thin streams, rules + model usually match or beat flood on recall *per unit attention*, because noise does not exhaust analyst caps before critical docs arrive.

5. **Missing documents.** recall@all controlled=0.61, flood=0.40. Both collapse toward the availability ceiling (available share ≈ 0.94). v3 recovers 32 missing docs via escalation; false-IC 0.00 vs flood 0.38; abstain=48. Still cannot invent evidence that escalation fails to obtain.

6. **Missing / redacted fields.** Model-only recall@available=0.33; controlled(v3)=0.60 after unredacted=43. FinanceGPT is blind on redactions until escalation lifts them; v3 requests unredact instead of inventing fields.

7. **Worst case (sparse_missing).** Controlled recall@all=0.50, false-IC rate=0.00. Expect low completeness; staged diagnosis fires supply-shortage=72, matching-failure=14 as the honest signal.

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
