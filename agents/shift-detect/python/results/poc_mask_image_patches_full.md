# POC: mask image patches → PO / power

grid=4×4, mask_frac=0.5, B=40, scenario=image_brightness_shift

| set | RF power | DR power | PO |
| --- | ---: | ---: | ---: |
| `text_only` | 0.00 | 0.07 | 0.0004772 |
| `image_full` | 0.03 | 0.00 | 0.0008473 |
| `image_masked` | 0.03 | 0.00 | 0.001654 |
| `mm_full` | 0.00 | 0.00 | 0.0002941 |
| `mm_masked` | 0.00 | 0.00 | 0.000356 |

ρ_PO image (mask/full) = 1.9516470054006667
ρ_PO mm (mask/full) = 1.2108042398730792
ΔRF image (mask−full) = +0.00
ΔRF mm (mask−full) = +0.00
mean |Δimage_meta| = 0.0880

Primary contrast is image_full vs image_masked under fixed brightness windows. Patch mask should drop image RF power if visual meta carried the shift; ρ_PO(mask/full) tracks PO-risk change.
