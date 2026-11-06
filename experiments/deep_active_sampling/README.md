# One-pass deep active sampling

A continuous loader yields `(x, y, index)` in row order. `one_pass` reads it once. The first `ref_batch_size` rows are the reference window: the first half is domain `T = 0`, the second half is `T = 1`. The remainder is the pool. The test set stays outside the loader. It is scored, not trained on.

The nuisance is fit on the reference only. `mu0` and `mu1` are ridge regressions of the integer class code. The propensity is logistic, clipped to `[0.01, 0.99]`. `tau` regresses the pseudo-outcome. Every pool row is scored as `T = 1`, and the sampling weight is the squared residual. A flat weight leaves the query ranking unchanged, because the residual is turned into a rank percentile before it multiplies the acquisition score.

The network is warm-started on the source half, copied, and then updated with SGD that is not reset between rounds. Each round queries one batch and takes a few epochs over the rows acquired so far.

```python
from experiments.deep_active_sampling import continuous_loader, one_pass
from experiments.deep_active_sampling.data import make_digits_stream

stream = make_digits_stream(seed=0)
loader = continuous_loader(stream["X"], stream["Y"])
one_pass(
    loader,
    batch_size=32,
    query_strategy={"entropy": "entropy", "entropy_po": {"strategy": "entropy", "weight": "po_risk"}},
    ref_batch_size=stream["ref_batch_size"],
    X_test=stream["X_test"],
    Y_test=stream["Y_test"],
    max_rounds=5,
    seed=0,
)
```

Digits, pen digits, and satimage enter through this loader. Digits roll the target image two columns. Pen digits shift the eight horizontal stroke coordinates by `0.25`. Satimage shifts the first nine standardized bands by `1.25`. Labels are unchanged. The pool is half target rows. The test set is target only.

```bash
python3 -m experiments.deep_active_sampling
python3 -m experiments.deep_active_sampling pendigits satimage
```

The runs write `results/BOARD.md`, `results/pendigits/`, `results/satimage/`, and `latex/accuracy_paths.tex`.

Query rules that score one classifier are registered: random, entropy, margin, least confidence, their dropout averages, BALD, a greedy BatchBALD stand-in, core-set, k-center, k-means, cluster-margin, BADGE, the baseline gradient swap, BIM, DeepFool, and active learning by learning. k-center is the same furthest-first rule as core-set. VAAL, WAAL, learning-loss, the GCN pair, and the semi-supervised trainers are not registered. They replace the network.
