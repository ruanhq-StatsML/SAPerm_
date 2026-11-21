# Polished results argument — modality / subset / marginal

## What we claim (no more)

The board answers three natural inputs under a shift:

| Question | Quantity | What we see (concept drift) |
|---|---|---|
| Which **modality**? | single / leave-one from pair | video drop from TAV *raises* CF/DR/RF power (top30) |
| Which **subset**? | pair / TAV power | T×A > TAV (top30); T×V best on pc31_100 |
| Which **marginal**? | LOMO $\Delta^{\mathrm{pow}}_m$, `vs_best` | all primary `vs_best` < 0 |

$I^{(2)}$ / $I^{(3)}$ / $I_{\mathrm{cal}}$ read non-additivity off those increments.

## Locked numbers

**LOMO (n=500, B≥120, n_perm=150)**  
- top30 CF: TAV 0.400 vs T×A 0.600 → vs_best −0.200  
- top30 DR: −0.100; RF: −0.067  
- pc31_100 RF: TAV 0.573 vs T×V 0.693 → −0.120  
- null RF: 0  

**RF PO (n=400, B=12)** — use $I_{\mathrm{cal}}$  
- concept top30 / pc31_100 / MOSEI: $I_{\mathrm{cal}}^{(3)}<0$, frac $I_{\mathrm{cal}}>0=0$  
- null: frac $I_{\mathrm{cal}}>0\approx 0.08$  
- planted TA: $I_{\mathrm{cal}}^{(2)}$ more negative under $\lambda<0$ (−31.3) than $\lambda>0$ (−8.0)

## Effect in one line

Cross-modal interaction is measurable; under this concept drift it is **not** super-additive for early-concat TAV—best pair wins, matching $I_{\mathrm{cal}}^{(3)}<0$.

## What else (optional, not required for the claim)

- Singles power grid (text / audio / video alone) for cleaner “which modality” column  
- MOSEI LOMO once emotion power hits B≥100 on all combos (running)  
- Audio×video combo power on concept (currently sparse in CF/DR boards)
