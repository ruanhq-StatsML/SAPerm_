# TabReD FSDS classification — run notes

Settings: `max_rows=8000`, `ref=800`, `n_perm=8`, serve=`keep_selected`,
seed=`2026`. Continuous targets (Weather, Sberbank) median-binarized on the
first half of the time-ordered stream.

Batch-size sweep: `50 / 100 / 200 / 400`. See `BOARD_BY_BATCH.md`.

## Online PO-risk adjustment (small batches)

For the FSDS penalty only, the reference domain is matched to the batch:

```
po_ref_size = min(ref_batch_size, max(4 * batch_size, 200))
```

So `b=50` uses `po_ref=200` (not 800-vs-50). The reward RF still fits on the
full 800-row reference. Larger batches (`b≥200`) keep `po_ref=800`.

## Takeaways

1. **Sberbank** keeps a clear FSDS lift at small batches: at **`b=50`**,
   `fsds_logo` / `boot_fsds_vimp` beat flat by **~+0.008** Brier — same order
   as `b=100/200`. The gap still collapses at `b=400`.

2. **Weather** at `b=50`: flat is slightly best; `fsds_logo` is essentially
   tied (`−0.0005`). `sum_VIMP` still over-penalizes block 3 (~79% late mass)
   and underperforms. Absolute rewards remain strong in the small-batch regime.

3. **Ecom** LOGO prefers small/medium batches (`b=50` best absolute; `b=200`
   best gap vs flat). **Homesite** LOGO is best at **`b=50`** (`+0.0006`).

4. Practical default for online PO-risk: **`batch_size=50–100`** with matched
   `po_ref≈4·batch`. Going to `400` speeds the run but flattens FSDS separation
   on Sberbank.
