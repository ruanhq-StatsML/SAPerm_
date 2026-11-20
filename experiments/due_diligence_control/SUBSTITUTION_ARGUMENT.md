# Substitution under missing / sparse data

# v6 board (reopen abstain after recovery + dual-look + critical lane)

After escalation recovers evidence, abstained deals reopen for IC retry.
Same-day dual-look, critical lane under soft-halt, thin-stream absorb.
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
| missing_docs | controlled | 382 | 0.94 | 0.94 | 0.00 | 25 | 36 | 0 | 178.0 | 62.2 |
| missing_docs | rules_blind | 420 | 0.96 | 0.94 | 0.00 | 27 | 35 | 0 | 161.9 | 60.0 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 373 | 0.81 | 0.81 | 0.00 | 27 | 0 | 43 | 155.5 | 47.2 |
| missing_fields | rules_blind | 386 | 0.75 | 0.75 | 0.00 | 23 | 0 | 43 | 139.9 | 44.1 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 352 | 0.74 | 0.74 | 0.00 | 18 | 20 | 14 | 150.6 | 43.0 |
| sparse_missing | rules_blind | 371 | 0.74 | 0.74 | 0.00 | 21 | 20 | 14 | 142.9 | 44.6 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled(v6) recall@available=0.97 vs flood=0.78, at coverage 345 vs 1080. Efficiency (confirmed/1k) controlled=202.9 vs flood=51.9. → OK to replace *undifferentiated reading*, not partner sign-off.

2. **v6 vs v1 (dense).** False-IC 0.75→0.00; abstain=20; reopened deals resume IC after integrity clears; net 16.7→55.3.

3. **v6 vs v3 (sparse freeze).** Soft-halt had zeroed desk budget; critical lane + dual-look: sparse recall 0.36→0.81 (beats flood 0.49); false-IC stays 0.

4. **v6 escalation+reopen vs v2 abstain-only (missing_docs).** recall@all 0.42→0.94 via recovered=36; false-IC stays 0.00; abstain drops as recoveries reopen.

5. **Sparse filings.** Controlled recall@available=0.81 (flood=0.49). Dual-look + thin-stream absorb keep arriving critical docs readable.

6. **Missing documents.** recall@all controlled=0.94, flood=0.40. v6 recovers 36 missing docs via escalation; false-IC 0.00 vs flood 0.38. Still cannot invent evidence that escalation fails to obtain.

7. **Missing / redacted fields.** Model-only recall@available=0.33; controlled(v6)=0.81 after unredacted=43.

8. **Worst case (sparse_missing).** Controlled recall@all=0.74, false-IC rate=0.00. Staged diagnosis fires supply-shortage / matching-failure as the honest signal.

### What *can* be substituted

- First-pass screening and prioritization under dense or sparse *observable* rooms
- Fatigue / backlog control so analysts do not over-read filler
- Explicit diagnosis of supply shortage vs matching failure
- Soft-halt that still clears a critical lane (v4)
- Same-day partner second look on missed critical confirms (v5)
- Reopening abstained mandates after evidence recovery (v6)

### What *cannot* be substituted

- Risks in documents that never arrive and escalation fails to recover
- Risks hidden by redaction / absent fields without alternative evidence
- Legal/partner accountability and primary confirmation interviews
- IC decisions when false-IC risk remains material

### Operating rule

Use controlled DD + FinanceGPT as a **capacity-aware co-pilot**. Treat `recall_all` under missingness as the integrity metric: if availability is low, refuse 'complete diligence' claims and escalate data collection. Soft-halt blocks filler — never the critical lane. Missed critical confirms get one same-day dual-look. Abstain is temporary until recovery clears integrity.
