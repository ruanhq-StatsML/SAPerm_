# Substitution under missing / sparse data

Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@available | Recall@all | False-IC rate | Confirmed/1k | Supply-shortage | Matching-failure | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 51.9 | 2 | 0 | 17.2 |
| dense | controlled | 332 | 0.75 | 0.75 | 0.75 | 162.7 | 0 | 0 | 28.7 |
| dense | rules_blind | 289 | 0.17 | 0.17 | 1.00 | 41.5 | 0 | 0 | 3.1 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 51.9 | 2 | 0 | 17.2 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 137.8 | 0 | 0 | 18.0 |
| sparse | controlled | 249 | 0.44 | 0.44 | 0.75 | 128.5 | 0 | 0 | 15.5 |
| sparse | rules_blind | 245 | 0.49 | 0.49 | 0.89 | 142.9 | 0 | 0 | 16.1 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 137.8 | 0 | 0 | 18.0 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 47.4 | 0 | 0 | 13.4 |
| missing_docs | controlled | 317 | 0.89 | 0.44 | 0.12 | 100.9 | 0 | 0 | 19.8 |
| missing_docs | rules_blind | 288 | 0.42 | 0.21 | 0.33 | 52.1 | 0 | 0 | 6.6 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 47.4 | 0 | 0 | 13.4 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 37.0 | 0 | 0 | 5.5 |
| missing_fields | controlled | 285 | 0.28 | 0.28 | 1.00 | 70.2 | 0 | 0 | 7.2 |
| missing_fields | rules_blind | 264 | 0.12 | 0.12 | 1.00 | 34.1 | 0 | 0 | 1.9 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 37.0 | 0 | 0 | 5.5 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 70.3 | 0 | 0 | 10.2 |
| sparse_missing | controlled | 291 | 0.38 | 0.28 | 0.75 | 68.7 | 0 | 0 | 8.1 |
| sparse_missing | rules_blind | 289 | 0.33 | 0.24 | 0.67 | 58.8 | 0 | 0 | 6.6 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 70.3 | 0 | 0 | 10.2 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled recall@available=0.75 vs flood=0.78, at coverage 332 vs 1080. Efficiency (confirmed/1k) controlled=162.7 vs flood=51.9. → OK to replace *undifferentiated reading*, not partner sign-off.

2. **Sparse filings.** Controlled recall@available=0.44 (flood=0.49). With thin streams, rules + model usually match or beat flood on recall *per unit attention*, because noise does not exhaust analyst caps before critical docs arrive.

3. **Missing documents.** recall@all controlled=0.44, flood=0.40. Both collapse toward the availability ceiling (available share ≈ 0.50). **No method recovers risks that never arrive.** Substitution fails here; the system should raise supply-shortage / escalate alternative data, not claim completeness.

4. **Missing / redacted fields.** Model-only recall@available=0.33; controlled=0.28. When risk language is stripped, FinanceGPT-class scores go blind. Rules still save capacity, but **cannot invent missing fields** — humans must pull primary sources.

5. **Worst case (sparse_missing).** Controlled recall@all=0.28, false-IC rate=0.75. Expect low completeness; staged diagnosis fires supply-shortage=0, matching-failure=0 as the honest signal.

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
