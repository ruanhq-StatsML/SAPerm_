# Data-digestion pipeline (ready)

Two claims only: **(A) power increase** from meta, **(B) interaction evidence + driving subset** after localization.

## Entrypoint

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# full digest → test → localize → subset (sentiment drivers)
python digest_pipeline.py --scenario sentiment_meta --subset

# smoke
python digest_pipeline.py --scenario sentiment_meta --smoke
```

## Pipeline stages

1. **ingest** — GEN text ↔ BiasFinder BERT align  
2. **extract** — `extract_features.py` (spaCy+VADER) + PCA(Z)  
3. **schema** — `feature_schema.py` list-indices (embed vs meta blocks)  
4. **window** — sentiment / syntax / heldout / null routing  
5. **test** — RFPerm / DRPerm / CFPerm  
6. **subset** — block LOCO → coordinate LOCO → driving meta-feature names  
7. **report** — ρ_PO + top block + subset (`results/digest_*.json`)

## Modules

| File | Role |
| --- | --- |
| `digest_pipeline.py` | **ready software** entrypoint |
| `subset_localize.py` | localization → meta-feature subset |
| `extract_features.py` | text meta only |
| `feature_schema.py` | modality / block indices |
| `perm_tests.py` | RFPerm / DRPerm / CFPerm |
| `dgp_text.py` | text DGP catalog |
| `extract_mm_features.py` | text+image+audio meta (ModaSense) |
| `datasets/modasense_emotion.py` | ModaSense digest → `Digested` + schema |

ModaSense digestion (read → meta → schema): [`docs/modasense_digestion.md`](docs/modasense_digestion.md).

## Business read (subset)

After localization names e.g. `vader_sent`, subset LOCO returns the **meta-features that drive the sentiment shift** (names + indices)—actionable for routing, review, and feature ops.
