# TabReD FSDS classification — run notes

Settings: `max_rows=8000`, `ref=800`, `n_perm=8`, serve=`keep_selected`,
seed=`2026`. Continuous targets (Weather, Sberbank) median-binarized on the
first half of the time-ordered stream.

Batch-size sweep: `100 / 200 / 400` (72 / 36 / 18 online steps). See
`BOARD_BY_BATCH.md`.

## Takeaways

1. **Sberbank Housing** remains the clearest FSDS win, and it prefers **smaller
   batches**. At `b=100` / `b=200`, `boot_fsds_vimp` beats flat by ~+0.008 Brier;
   at `b=400` the gap collapses (only 18 updates, less room to adapt).

2. **Weather**: `fsds_logo` stays slightly above flat at every batch size
   (`+0.001–0.003`). Absolute rewards are best at `b=100` (more steps).
   `sum_VIMP` still over-penalizes block 3 and underperforms on reward, while
   still flagging that block as the late-shift mass (~66–80%).

3. **Ecom / Homesite**: gaps stay small. `b=200` is the sweet spot for LOGO on
   ecom (`+0.0016` vs flat); homesite is nearly tied across batches.

4. Practical default: **`batch_size=100` or `200`** with `ref=800`. Going to
   `400` speeds the run but flattens the FSDS-vs-flat separation on Sberbank.
