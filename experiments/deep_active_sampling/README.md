# Weighted deep active sampling

Query rules follow the supervised strategies in
[cure-lab/deep-active-learning](https://github.com/cure-lab/deep-active-learning/tree/main/query_strategies)
that score a batch from one classifier: random, entropy, margin, least
confidence, the three dropout averages, BALD, BatchBALD, core-set, k-center,
k-means, cluster-margin, BADGE, the baseline gradient swap, BIM, DeepFool,
and active learning by learning.

VAAL, WAAL, learning-loss, the GCN pair, and the semi-supervised trainers are
not registered. They replace the network, not the query.

Each `query` takes a sample-weight vector. A flat weight leaves the original
ranking in place. The comparison weight is the per-row pseudo-outcome residual
from a nuisance fit on the reference window: the first half is domain `T = 0`,
the second half is domain `T = 1`.

The stream is sklearn digits, with pixels left in `[0, 1]`. Target images are
rolled two pixels to the right. The test set is rolled. The pool mixes
original and rolled images. k-center uses the same furthest-first rule as
core-set, so the reported mean counts that curve once. Active learning by
learning scores a batch with the network that queried it, before the SGD step.

```bash
python3 -m experiments.deep_active_sampling
```

The run writes `results/BOARD.md` and `results/accuracy_paths.png`.

```python
from OOD_vimp_wrapper import permuCATE_fit_nuisance
from experiments.deep_active_sampling import ACTIVE_SAMPLING_SCHEMES, deep_active_sampling
```
