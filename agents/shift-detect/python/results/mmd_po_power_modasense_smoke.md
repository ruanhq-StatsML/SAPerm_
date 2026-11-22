# ModaSense: MMD × ρ_PO × power

B=6, n_doc=40, RBF-MMD² on z-scored modality blocks (batch0 vs batch1).

## Per-modality shift MMD vs power

| scenario | MMD text | MMD image | MMD audio | MMD all | RF t/i/a | ρ TI/T | ρ all/T | align |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| audio_energy_shift | 0.0219 | 0.0202 | 0.148 | 0.0554 | 0.00/0.00/0.00 | 1.85 | 1.06 | MMD_matches_routing |
| image_brightness_shift | 0.0239 | 0.157 | 0.0214 | 0.0572 | 0.00/0.00/0.00 | 1.58 | 1.94 | MMD_matches_routing |
| null_shift | 0.032 | 0.0317 | 0.0254 | 0.0297 | 0.17/0.17/0.17 | 1.71 | 2.57 | null |
| text_sentiment_shift | 0.101 | 0.0264 | 0.0344 | 0.0676 | 0.83/0.83/0.83 | 2.03 | 1.94 | MMD_matches_routing |

## How to read (argument)

1. **MMD** = pure distribution gap on that modality's features (no Y). If the window is routed by text polarity, MMD_text should dominate.
2. **RF power** = whether that modality alone detects the shift for Y. High MMD + high RF ⇒ that modality both moves and is outcome-relevant.
3. **ρ_PO** = PO(rich)/PO(text). ρ≪1 ⇒ added modality absorbed covariate into nuisance; ρ≈1 or ≫1 ⇒ did not absorb / residual grew.
4. **Insight**: MMD tells *which channel moved*; power tells *who can reject*; ρ_PO tells *whether concatenating that channel fixed the PO residual*. They need not agree — that disagreement is the point.
