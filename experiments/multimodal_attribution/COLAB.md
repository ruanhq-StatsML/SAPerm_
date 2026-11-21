# Run a concept-drift power slice on Google Colab

Open the notebook:

[`Colab_ConceptPower_CF_RF_DR.ipynb`](./Colab_ConceptPower_CF_RF_DR.ipynb)

Or from a Colab runtime:

1. **File → Upload notebook** (or open via GitHub URL once pushed).
2. Run cells top to bottom (CPU is enough).
3. Default knobs: `n=200`, `B=30`, `n_perm=40`, methods `cfperm/rfperm/drperm`, combo `text_audio` only.

CLI equivalent:

```bash
git clone -b cursor/cfdr-accel-perm-parallel-e93c https://github.com/ruanhq-StatsML/SAPerm_.git
cd SAPerm_
pip install numpy pandas scikit-learn joblib scipy
python -m experiments.multimodal_attribution.colab_concept_power \
  --n 200 --n-rep 30 --n-perm 40 --n-jobs 2
```

Results land in `experiments/multimodal_attribution/results/colab_concept/` (`power_raw.csv`, `POWER_BOARD.md`). Re-running with a larger `--n-rep` stacks new Monte Carlo reps on the checkpoint.
