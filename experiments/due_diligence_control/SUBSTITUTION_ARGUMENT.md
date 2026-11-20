# Substitution under missing / sparse data

# v7 board (adaptive escalation + reopen + dual-look + critical lane)

Repeated shortage pulses raise alt-data / unredact odds (capped). Abstain
reopens after recovery. Dual-look + critical lane retained. False-IC stays 0.


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
| missing_docs | controlled | 387 | 0.99 | 0.99 | 0.00 | 26 | 36 | 0 | 183.5 | 65.1 |
| missing_docs | rules_blind | 413 | 0.99 | 0.99 | 0.00 | 27 | 36 | 0 | 171.9 | 61.9 |
| missing_docs | model_only | 612 | 0.81 | 0.40 | 0.38 | 0 | 0 | 0 | 47.4 | 11.0 |
| missing_fields | flood | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| missing_fields | controlled_v1 | 285 | 0.28 | 0.28 | 1.00 | 0 | 0 | 0 | 70.2 | 4.7 |
| missing_fields | controlled_v2 | 294 | 0.28 | 0.28 | 0.00 | 6 | 0 | 0 | 68.0 | 8.4 |
| missing_fields | controlled | 380 | 0.82 | 0.82 | 0.00 | 35 | 0 | 46 | 155.3 | 48.3 |
| missing_fields | rules_blind | 400 | 0.79 | 0.79 | 0.00 | 20 | 0 | 47 | 142.5 | 48.6 |
| missing_fields | model_only | 648 | 0.33 | 0.33 | 1.00 | 0 | 0 | 0 | 37.0 | 2.3 |
| sparse_missing | flood | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |
| sparse_missing | controlled_v1 | 291 | 0.38 | 0.28 | 0.75 | 0 | 0 | 0 | 68.7 | 5.7 |
| sparse_missing | controlled_v2 | 281 | 0.33 | 0.24 | 0.00 | 4 | 0 | 0 | 60.5 | 5.9 |
| sparse_missing | controlled | 347 | 0.69 | 0.69 | 0.00 | 20 | 20 | 14 | 144.1 | 40.7 |
| sparse_missing | rules_blind | 358 | 0.72 | 0.72 | 0.00 | 25 | 20 | 14 | 145.3 | 42.4 |
| sparse_missing | model_only | 327 | 0.44 | 0.32 | 0.67 | 0 | 0 | 0 | 70.3 | 7.0 |

## Verdict: can this substitute for traditional due diligence?

### Short answer

**Partial substitute for triage and attention allocation — not a full substitute for primary-source diligence when evidence is missing.**

### Evidence from this board

1. **Dense data room.** Controlled(v7) recall@available=0.97 vs flood=0.78, at coverage 345 vs 1080. Efficiency (confirmed/1k) controlled=202.9 vs flood=51.9.

2. **v7 vs v1 (dense).** False-IC 0.75→0.00; abstain=20 with reopen path; net 16.7→55.3.

3. **v7 vs v3 (sparse freeze).** Critical lane + dual-look: sparse recall 0.36→0.81 (beats flood 0.49); false-IC stays 0.

4. **v7 adaptive escalate+reopen vs v2 (missing_docs).** recall@all 0.42→0.99 via recovered=36; false-IC stays 0.00.

5. **Missing / redacted fields.** controlled(v7)=0.82 after unredacted=46 vs model-only 0.33.

6. **Worst case (sparse_missing).** Controlled recall@all=0.69, false-IC=0.00. Adaptive persistence helps but cannot invent evidence that never recovers.

### What *can* be substituted

- First-pass screening / prioritization under observable rooms
- Fatigue / backlog control; soft-halt with critical lane
- Dual-look on missed critical confirms; reopen after recovery
- Adaptive escalation persistence under repeated shortage

### What *cannot* be substituted

- Risks that escalation never recovers
- Legal/partner accountability and primary confirmation interviews
- IC claims when false-IC risk remains material

### Operating rule

Capacity-aware co-pilot: score → allocate → integrity-gate → escalate (persist) → dual-look → reopen. Soft-halt never zeros the critical lane. `recall_all` under missingness is the integrity metric.
