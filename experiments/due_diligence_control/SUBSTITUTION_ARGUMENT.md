# Substitution under missing / sparse data

# v4 board (critical lane + thin-stream absorb + escalation)

Soft-halt no longer zeros desk budget (v3 sparse freeze). Critical lane keeps
planted/high-risk readable; thin-stream absorb covers arriving docs; escalation
still recovers missing/redacted evidence with false-IC at 0.


Planted critical risks per deal; methods must find them under each regime.

Seeds=[11, 12, 13, 14], days=18, deals=6, analysts=12, planted/deal=3.

| Regime | Method | Coverage | Recall@avail | Recall@all | False-IC | Abstain | Recovered | Unredacted | Confirmed/1k | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | flood | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| dense | controlled_v1 | 332 | 0.75 | 0.75 | 0.75 | 0 | 0 | 0 | 162.7 | 16.7 |
| dense | controlled_v2 | 326 | 0.79 | 0.79 | 0.00 | 52 | 0 | 0 | 174.8 | 39.8 |
| dense | controlled | 335 | 0.81 | 0.81 | 0.00 | 44 | 0 | 0 | 173.1 | 39.9 |
| dense | rules_blind | 301 | 0.82 | 0.82 | 0.00 | 11 | 0 | 0 | 196.0 | 34.0 |
| dense | model_only | 1080 | 0.78 | 0.78 | 1.00 | 0 | 0 | 0 | 51.9 | 0.4 |
| sparse | flood | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| sparse | controlled_v1 | 249 | 0.44 | 0.44 | 0.75 | 0 | 0 | 0 | 128.5 | 10.7 |
| sparse | controlled_v2 | 247 | 0.49 | 0.49 | 0.00 | 16 | 0 | 0 | 141.7 | 19.8 |
| sparse | controlled | 249 | 0.47 | 0.47 | 0.00 | 18 | 0 | 0 | 136.5 | 19.4 |
| sparse | rules_blind | 244 | 0.50 | 0.50 | 0.00 | 11 | 0 | 0 | 147.5 | 19.1 |
| sparse | model_only | 254 | 0.49 | 0.49 | 0.67 | 0 | 0 | 0 | 137.8 | 13.2 |
| missing_docs | flood | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_docs | controlled_v1 | 317 | 0.89 | 0.44 | 0.12 | 0 | 0 | 0 | 100.9 | 19.0 |
| missing_docs | controlled_v2 | 318 | 0.83 | 0.42 | 0.00 | 18 | 0 | 0 | 94.3 | 13.7 |
| missing_docs | controlled | 356 | 0.74 | 0.69 | 0.00 | 56 | 32 | 0 | 140.4 | 37.2 |
| missing_docs | rules_blind | 376 | 0.69 | 0.68 | 0.00 | 37 | 35 | 0 | 130.3 | 34.3 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 341 | 0.56 | 0.56 | 0.00 | 40 | 0 | 39 | 117.3 | 31.3 |
| missing_fields | rules_blind | 353 | 0.58 | 0.58 | 0.00 | 34 | 0 | 40 | 119.0 | 29.2 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 311 | 0.54 | 0.53 | 0.00 | 27 | 19 | 12 | 122.2 | 28.0 |
| sparse_missing | rules_blind | 312 | 0.51 | 0.51 | 0.00 | 23 | 20 | 13 | 118.6 | 29.8 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled(v4) recall@available=0.81 vs flood=0.78, at coverage 335 vs 1080. Efficiency (confirmed/1k) controlled=173.1 vs flood=51.9. → OK to replace *undifferentiated reading*, not partner sign-off.

2. **v4 vs v1 (dense).** False-IC 0.75→0.00; abstain=44; recovered=0; net 16.7→39.9.

3. **v4 vs v3 (sparse freeze).** Soft-halt had zeroed desk budget; critical lane restores sparse recall 0.36→0.47 (near flood 0.49) while false-IC stays 0.

4. **v4 escalation vs v2 abstain-only (missing_docs).** recall@all 0.42→0.69 via recovered=32; false-IC stays 0.00.

5. **Sparse filings.** Controlled recall@available=0.47 (flood=0.49). Thin-stream absorb + critical lane keep arriving docs readable under soft-halt.

6. **Missing documents.** recall@all controlled=0.69, flood=0.40. v4 recovers 32 missing docs via escalation; false-IC 0.00 vs flood 0.38; abstain=56. Still cannot invent evidence that escalation fails to obtain.

7. **Missing / redacted fields.** Model-only recall@available=0.33; controlled(v4)=0.56 after unredacted=39. FinanceGPT is blind on redactions until escalation lifts them.

8. **Worst case (sparse_missing).** Controlled recall@all=0.53, false-IC rate=0.00. Staged diagnosis fires supply-shortage / matching-failure as the honest signal.

### What *can* be substituted

- First-pass screening and prioritization under dense or sparse *observable* rooms
- Fatigue / backlog control so analysts do not over-read filler
- Explicit diagnosis of supply shortage vs matching failure
- Soft-halt that still clears a critical lane (v4)

### What *cannot* be substituted

- Risks in documents that never arrive (missing_docs)
- Risks hidden by redaction / absent fields without alternative evidence
- Legal/partner accountability and primary confirmation interviews
- IC decisions when false-IC risk remains material

### Operating rule

Use controlled DD + FinanceGPT as a **capacity-aware co-pilot**. Treat `recall_all` under missingness as the integrity metric: if availability is low, refuse 'complete diligence' claims and escalate data collection. Soft-halt blocks filler — never the critical lane.
