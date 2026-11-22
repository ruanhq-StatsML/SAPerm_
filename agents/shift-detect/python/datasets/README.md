# Per-dataset digestion

One script = one dataset. Same five stages every time:

```
read → preprocess → embed → meta → schema
```

| Script | Dataset | Embed | Meta |
| --- | --- | --- | --- |
| `biasfinder_gen.py` | GEN-sarc + BiasFinder BERT | PCA(Z) | spaCy + **VADER** |
| `modasense_emotion.py` | ModaSense/emotion | — | text meta + OpenCV image |

```bash
cd agents/shift-detect/python
python -m datasets.biasfinder_gen --n-max 2000 --write results/digest_biasfinder.npz
python -m datasets.modasense_emotion --n-max 500
```

Downstream tests (`digest_pipeline.py`, `power_*.py`) consume the digested
`Digested` / cached `.npz` — they do not re-invent ingest.

VADER feature-engineering argument: [`../docs/vader_feature_engineering.md`](../docs/vader_feature_engineering.md).
