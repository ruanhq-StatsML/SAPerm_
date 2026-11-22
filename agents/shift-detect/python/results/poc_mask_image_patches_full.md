# POC: mask image patches → ρ_PO / power

grid=4×4, mask_frac=0.5, B=40, windows=`image_brightness_shift`.

| set | RF | DR | PO |
| --- | ---: | ---: | ---: |
| text_only | 0.00 | 0.07 | 0.00048 |
| image_full | 0.03 | 0.00 | 0.00085 |
| image_masked | 0.03 | 0.00 | 0.00165 |
| mm_full | 0.00 | 0.00 | 0.00029 |
| mm_masked | 0.00 | 0.00 | 0.00036 |

**ρ_PO (mask / full)**

| contrast | ρ_PO |
| --- | ---: |
| image_masked / image_full | **1.95** |
| mm_masked / mm_full | 1.21 |

ΔRF image/mm = 0.00. mean |Δimage_meta| = 0.088.

Reading: RF power was already near zero on this axis; the informative
change is ρ_PO≈1.95 — patch mask degrades the image nuisance, PO-risk up.
Power unchanged. Side POC only.
