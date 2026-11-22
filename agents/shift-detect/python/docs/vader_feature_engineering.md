# VADER feature-engineering

Block `vader_sent` in meta \(M\)（on top of embed \(Z\)）:

| feature | meaning |
| --- | --- |
| `vader_compound` | polarity \([-1,1]\) — window routing |
| `vader_pos` / `neu` / `neg` | component masses |
| `vader_polar_gap` | \(\|pos-neg\|\) (conflict; compound cancels) |

```python
s = SentimentIntensityAnalyzer().polarity_scores(text)
# compound, pos, neu, neg; polar_gap = |pos - neg|
```

Portable lexicon (no GPU / no labels). Separate from spaCy blocks so CFPerm can name emotion vs syntax.

## Results (not propositions)

| evidence | number | reading |
| --- | --- | --- |
| feature-only `sentiment_shift` | baseline 0.40 → meta **1.00** | VADER detects polarity-aligned shift |
| BiasFinder `sentiment_meta` | \(\rho_{PO}\approx\mathbf{0.25}\) | \(Z{+}M\) absorbs emotion covariate |
| same, driving subset | `compound`, `neg`, `polar_gap`, `neu` | named emotion drivers |
| BiasFinder `train_vs_heldout` | RF/DR power\((Z)=1\), \(\rho_{PO}\approx1\) | concept residual stays in embed |

Subset summarization compresses the **emotion axis**; it does not replace \(Z\).  
See [`embed_vs_vader_subset.md`](embed_vs_vader_subset.md).

Digestion: `python -m datasets.biasfinder_gen` → `digest_pipeline.py --scenario sentiment_meta --subset`.
