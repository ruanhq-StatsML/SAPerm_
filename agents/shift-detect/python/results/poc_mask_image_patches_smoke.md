# POC: mask image patches → PO / power

grid=4×4, mask_frac=0.5, B=8, scenario=image_brightness_shift

| set | RF power | DR power | PO |
| --- | ---: | ---: | ---: |
| `text_only` | 0.00 | 0.00 | 0.0008272 |
| `mm_full` | 0.00 | 0.00 | 0.00148 |
| `mm_masked` | 0.00 | 0.00 | 0.001499 |

ρ_PO(mask/full) = 1.013
ΔRF (mask−full) = +0.00
ΔDR (mask−full) = +0.00

Masking patches ablates visual signal before meta extract. If image channel mattered, mm_masked should lose RF power and/or move PO vs mm_full under the same brightness windows.
