# Where to paste in the ACL paper

In `Multi-Modal Datasets`, after the paragraph that starts

> Higher-order interaction and leave-one-modality-out power.

replace that short sketch (or append) with:

```latex
\input{i3_lomo_acl_fragment.tex}
```

Suggested order in the section:

1. Multi-modal Omni-3B setup + Table `\ref{tab:full-benchmark-power-typeI}` (power / type-I)
2. `\input{i3_lomo_acl_fragment.tex}` — defs + LOMO table + RF $I^{(2)}/I^{(3)}$ table + cross-validation conclusion
3. Continue to R-risk Tests

The fragment references `\ref{tab:full-benchmark-power-typeI}`, so keep that table label unchanged.
