# Per-dataset digestion

```
read → preprocess → [embed] → meta → schema → Digested
```

| module | data | layout |
| --- | --- | --- |
| `biasfinder_gen` | GEN + BiasFinder BERT | `X=[Z_PCA, M_text]` |
| `modasense_emotion` | ModaSense/emotion | `X=[M_text, M_image, M_audio]` |

```bash
python -m datasets.biasfinder_gen --n-max 2000
python -m datasets.modasense_emotion --n-max 500 --audio --write
python digest_pipeline.py --scenario sentiment_meta --subset   # BiasFinder path
```

Shared helpers: `_common.py` (`Digested`, `cached_text_meta`, `pca_embed`).  
- BiasFinder VADER notes: [`../docs/vader_feature_engineering.md`](../docs/vader_feature_engineering.md)  
- **ModaSense digestion (primary multimodal path):** [`../docs/modasense_digestion.md`](../docs/modasense_digestion.md)
