# Results index (ρ_PO–first)

Canonical writeup: [`unified_meta_power.tex`](unified_meta_power.tex)

| artifact | what |
| --- | --- |
| `interaction_signatures.json` | BiasFinder ρ_PO + Δpower |
| `power_cross_modality_full.md` | CFPerm block powers (B=150) |
| `power_mm_modasense_full.md` | ModaSense text/image ρ_PO |
| `case_modasense_business.md` | CFPerm subset → business map |
| `poc_mask_image_patches_full.md` | patch-mask ρ_PO≈1.95 (side POC) |
| `mmd_two_datasets.md` | BiasFinder + ModaSense MMD × ρ_PO × power |
| `mmd_modalities.py` | ready: arrays in → per-mod MMD out |
| `realdata_po_power_crosscheck.md` | absorb / localize / residual table |

**ρ_PO cheat sheet**

| setting | ρ_PO | read |
| --- | ---: | --- |
| BiasFinder sentiment | 0.25 | absorb |
| BiasFinder syntax | 0.19 | absorb + localize (ΔCF +0.09) |
| BiasFinder heldout | 1.03 | residual |
| ModaSense image brightness | 0.62 | absorb |
| Patch-mask image (POC) | 1.95 | nuisance degraded |
