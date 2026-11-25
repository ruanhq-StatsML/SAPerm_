# Dataset-specific digestion only

```
read → meta[/embed] → schema → Digested + make_windows()
```

| module | layout |
| --- | --- |
| `biasfinder_gen` | `X=[Z_PCA, M_text]` |
| `modasense_emotion` | `X=[M_text, M_image, M_audio]` |

FE helpers used only here: `text_meta.py`, `mm_meta.py`, `feature_schema.py`.
