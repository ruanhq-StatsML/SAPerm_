# TabReD FSDS classification — run notes

Settings: `max_rows=8000`, `ref=800`, `batch=200`, `n_perm=8`, serve=`keep_selected`,
seed=`2026`. Continuous targets (Weather, Sberbank) median-binarized on the first
half of the time-ordered stream.

## Takeaways

1. **Sberbank Housing** (392 feats, VIMP clusters): clearest FSDS lift.
   `boot_fsds_vimp` / `fsds_vimp` beat flat Thompson by ~0.008 Brier
   (`-0.143` vs `-0.151`).

2. **Weather** (103→97 feats): `sum_VIMP` puts ~66–87% of the late/early shift
   mass on block 3 (last contiguous quarter). `fsds_logo` is the best reward
   policy here (`-0.134` vs flat `-0.136`); pure `sum_VIMP` over-penalizes and
   under-uses that block.

3. **Ecom Offers / Homesite**: native binary labels; gaps are smaller. LOGO
   edges flat TS on ecom; homesite is nearly tied across policies.

4. Arm paths + `early_w` / `late_w` are the attribution readout: FSDS is doing
   block-level shift monitoring even when reward gaps are modest.
