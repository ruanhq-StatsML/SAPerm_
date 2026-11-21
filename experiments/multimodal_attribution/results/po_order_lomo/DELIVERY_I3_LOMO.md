# Delivery — $I^{(3)}$ + LOMO cross-validation (RF)

LaTeX (include from paper):
- `latex/main_i3_lomo.tex` → interpretation + tables
- `latex/higher_order_lomo_interpretation.tex`
- `latex/i3_lomo_delivery.tex`
- Wired into `latex/main_delivery_po.tex`

## Locked conclusion

1. **LOMO (concept drift):** `Power(TAV) < max Power(pair)` for CF/DR/RF on top30 and RF on pc31_100. Dropping video from TAV *raises* CF/DR/RF power on top30.
2. **RF $I_{\mathrm{cal}}^{(2)}$ / $I_{\mathrm{cal}}^{(3)}$:** all mean calibrated interactions **negative** on concept (frac $I_{\mathrm{cal}}>0=0$); null stays near-quiet on calibrated sign.
3. **Cross-validation:** negative $I_{\mathrm{cal}}^{(3)}$ ↔ negative LOMO `vs_best` → over-complete / conflicting triple early fusion. Ship dominant pair; third modality optional / late-fuse.
4. Not unique attribution — sign agreement only.

## Numbers (RF PO, $n=400$, $B=12$)

| dataset | mean $I^{(3)}$ | mean $I^{(3)}_{\mathrm{cal}}$ | frac $I_{\mathrm{cal}}>0$ |
|---|---:|---:|---:|
| null top30 | 15.39 | −5.77 | 0.08 |
| concept top30 | 10.89 | −2.22 | 0.00 |
| concept pc31_100 | 111.0 | −779.0 | 0.00 |
