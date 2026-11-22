# Per-dataset digestion

```
read → preprocess → embed → meta → schema
```

| module | data |
| --- | --- |
| `biasfinder_gen` | GEN + BiasFinder BERT → PCA(Z) + spaCy/VADER |
| `modasense_emotion` | ModaSense text+image meta |

```bash
python -m datasets.biasfinder_gen --n-max 2000
python -m datasets.modasense_emotion --n-max 500
python digest_pipeline.py --scenario sentiment_meta --subset
```

Shared helpers: `_common.py` (`Digested`, `cached_text_meta`, `pca_embed`).  
VADER: [`../docs/vader_feature_engineering.md`](../docs/vader_feature_engineering.md).
