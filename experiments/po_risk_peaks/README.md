# PO-risk mean and peaks on HC3

The deployed detector is `distilbert-base-uncased`, fine-tuned on HC3 human versus ChatGPT answers (finance, open_qa, medicine, wiki_csai). A 12-layer RoBERTa does not fit the CPU box this run uses. The proxy and the two blocks are unchanged: the proxy is `1 - p(y|x)`, the lexical block is the token embedding, and the semantic block is the last-layer CLS.

Live batches, all paired with the same test texts:

- REF, the clean test texts
- back-translation through `Helsinki-NLP/opus-mt-en-zh` and `opus-mt-zh-en`
- WordNet synonym replacement at 15% of eligible tokens
- TextFooler: delete-to-rank, then substitute a synonym that lowers the true-class probability, stop on a flip
- FGSM on the input embedding, with the step size chosen so the probe flip rate is near one half

The mean of the proxy moves whenever the batch gets harder. The peak gap of a half-corpus bootstrap does not, because those means share one expectation. The figure uses sub-batches of 32, and the tail excess `CVaR95 - mean` of the example residuals.

Malignant response: freeze the embedding and keep training the upper layers on clean/TextFooler pairs from the training split, with a CLS cosine penalty. The live strings stay the ones attacked against the original detector.

Benign response: leave the weights alone and recompute a split-conformal threshold on `p(AI)` so the false positive rate on human text sits at 5%.

```bash
python3 -m pytest experiments/po_risk_peaks/test_core.py
python3 -m experiments.po_risk_peaks.run
```

`results/BOARD.md` is the reading. `results/diagnosis_scatter.png` is the figure.
