# Real-data PO-risk × power cross-check (BiasFinder GEN)

Not synthetic. $B{=}150$. $\rho_{PO}=PO(Z{+}M)/PO(Z)$.

| scenario | DR power e→m | ρ_PO | ΔCF | signature |
| --- | ---: | ---: | ---: | --- |
| null_shift | 0.07→0.07 | 1.07 | +0.03 | null calibration |
| sentiment_meta | 0.00→0.00 | **0.25** | +0.01 | **absorb** (meta×embed interaction in nuisance) |
| syntax_meta | 0.00→0.00 | **0.19** | **+0.09** | **localize** (CF names meta; PO absorbed) |
| train_vs_heldout | 1.00→1.00 | **1.03** | **+0.23** | **concept residual** + localize |

Reading: power and ρ_PO jointly match the real-data regime — that joint match **is** the interaction evidence.

ModaSense (new multimodal real data) full run in progress → `power_mm_modasense_full.md`.
