# Real-data ρ_PO × power (BiasFinder)

ρ_PO = PO(Z+M) / PO(Z). B=150.

| scenario | DR e→m | ρ_PO | ΔCF | signature |
| --- | ---: | ---: | ---: | --- |
| null_shift | 0.07→0.07 | 1.07 | +0.03 | calibration |
| sentiment_meta | 0.00→0.00 | **0.25** | +0.01 | absorb |
| syntax_meta | 0.00→0.00 | **0.19** | **+0.09** | localize |
| train_vs_heldout | 1.00→1.00 | **1.03** | **+0.23** | residual |

ρ_PO is the channel read; ΔCF names the block; power says reject rate.
Canonical: `unified_meta_power.tex`.
