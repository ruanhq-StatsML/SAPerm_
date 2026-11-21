# Text meta-features (spaCy + VADER)

Dedicated feature script + power argument for the attribution agent.

## Scripts

| Script | Role |
| --- | --- |
| `extract_features.py` | **Only features** — spaCy style/syntax/NER + VADER sentiment |
| `power_meta_features.py` | Type-I / power: `baseline` vs `meta` vs `both` |
| `meta_features.py` | Fixture attribution helper used by the Agent tool |

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Extract features

```bash
python extract_features.py --demo --pretty
python extract_features.py --set meta --json texts.json
python extract_features.py --set baseline --json texts.json   # ablation surface stats
```

## Power argument

```bash
python power_meta_features.py --n-rep 20 --n-perm 40 --n-doc 20
# writes results/power_meta.json and results/power_meta.md
```

Latest offline result (α=0.05):

| scenario | baseline | meta | Δ |
| --- | ---: | ---: | ---: |
| null Type-I | 0.05 | 0.05 | controlled |
| sentiment_shift power | 0.55 | **1.00** | **+0.45** |
| syntax_shift power | 0.05 | **1.00** | **+0.95** |
| entity / style | 1.00 | 1.00 | saturated (surface also sees it) |

**Claim:** when the planted signal lives in linguistic/sentiment structure (VADER polarity, spaCy POS), meta-features raise power versus a surface baseline without blowing Type-I.
