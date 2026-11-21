# Cross-modal readout: modality-specific vs signed interactions

## 1. Two different questions

| Question | Object | Sign? |
|---|---|---|
| **Modality-specific proportion** | How much detectable drift lives in one modality *block* (text / audio / video)? | Direction of that block’s additive shift (or its share of risk drop when the block is kept). |
| **Modality interaction (+/−)** | Does the *joint* of two modalities move the domain gap beyond the sum of singles? | **+** synergistic / amplifying; **−** antagonistic / cancelling. |

Do **not** read the concept-drift combo table (T×A vs T×V) as signed attribution. That table answers a third, simpler question: *with the same $Y\mid X$ drift, which modality subset makes CF/RF/DRPerm more powerful?*

## 2. What our finished T×A / T×V numbers say

Under **concept drift** ($X$ identical across domains; only $Y\mid X$ changes):

| | CFPerm | DRPerm | RFPerm |
|---|---:|---:|---:|
| T×A | 0.60 | 0.40 | 1.00 |
| T×V | 0.34 | 0.22 | 0.99 |

- **Modality-subset “proportion” (detectability):** text+audio carries a larger share of the *usable* signal for CF/DR than text+video on this DGP (PCA top-30, `concept_shift=0.35`). RFPerm saturates both.
- **Classical TST ≈ 0** on both combos: expected — they look at $X$, and $X$ did not move.
- This is **not** yet a signed text×audio interaction estimate; it is combo-wise global power.

## 3. Where signed interactions live (board design)

Standardized DGPs in `DatasetSpec`:

1. **`mean_shift`** — modality-specific: add a constant to one block (e.g. video).  
   - “Proportion” ≈ power when that modality is in the combo vs left out.  
   - Sign of the shift = direction of the modality-specific effect on $X$.

2. **`interaction_shift`** — inject a product term on a pair $(a,b)$ into $X$ and into $Y$.  
   - Detectors: `crosscorr_*` (dependence change) and `interact_PO_*`  
     with PO interaction  
     $\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)$.  
   - **Positive interaction:** leaving out both modalities hurts *less* than the sum of single leave-outs would predict in the inclusion–exclusion sense used here — the joint structure is doing extra work (synergy toward domain/outcome risk).  
   - **Negative interaction:** the opposite — the pair cancels; joint leave-out pattern indicates antagonism.

3. **`concept_drift`** — no $X$ interaction by construction; mid-PC nonlinear $Y\mid X$ (includes a product of PCA coords). Combo power ranks which modality *supports* that regression signal after concat.

## 4. Practical reading rule

- Report **T×A / T×V power** as cross-modal *coverage / detectability*.  
- Report **mean_shift** boards as modality-specific proportions.  
- Report **interact_PO / crosscorr** (with sign of the PO interaction statistic, not only the p-value) as positive vs negative modality interactions.  
- Always pair with **type-I** on null: CF/DR ≈ α; SKK13/ZWL inflated.
