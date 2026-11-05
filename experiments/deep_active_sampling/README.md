# Weighted deep active sampling

Query rules follow the supervised strategies in
[cure-lab/deep-active-learning](https://github.com/cure-lab/deep-active-learning/tree/main/query_strategies):
random, entropy, margin, least confidence, BALD, cluster-margin, core-set,
k-means, and BADGE.
Each `query` takes a sample-weight vector. A flat weight leaves the original
ranking in place. Any other weight is a rank percentile and rescales the
acquisition score.

The weights come from the out-of-distribution scores in `ood_vimp.py`:

- PermuCATE pseudo-outcome risk, with a conditional permutation of each coordinate
- a random-forest domain classifier
- RBF MMD, leave-one-covariate-out

Those three are averaged into one weight per coordinate. A pool row is then
scored by its weighted squared distance from the source mean. The per-row
pseudo-outcome residual is a second weight. It uses the observed outcome,
because this loop chooses which rows already in the stream are fed to SGD.

The reference window is split in half. The first half is domain `T = 0`. The
second half is domain `T = 1`. Nuisance regressions are ridge and logistic,
with the same `fit` / `predict` contract as the FSDS model registry.

```bash
python3 -m experiments.deep_active_sampling
```

The run writes `results/BOARD.md`. The stream is a mixture with a known
shifted block, not CIFAR-100. `load_cifar100_arrays` returns flattened
CIFAR-100 arrays for the same `IndexedArrayDataset` when a local copy is
available.

```python
from OOD_vimp_wrapper import permuCATE_fit_nuisance, permuCATE_vimp_withPOModel
from experiments.deep_active_sampling import ACTIVE_SAMPLING_SCHEMES, deep_active_sampling
```
