# Sensitivity argument: ±modality shifts × signed PO × single/pair/TAV power

**Methods only:** CFPerm / DRPerm / RFPerm (+ `interact_PO` / `crosscorr` as mechanism readouts).

## What we vary

| Axis | Levels | Role |
|---|---|---|
| Scenario | `interaction_shift`, `mean_shift`, `no_shift` | plant synergy / plant main effect / null |
| Direction | $\lambda=\pm 1.8$ (interact); mean $\pm 1.0$ | sign of effect |
| Pair / modality | T×A, T×V, A×V; text/audio/video | where the plant sits |
| Combo hierarchy | **single → pair → TAV** | detectability ladder |

## Two readouts of the *same* story

### A. Mechanism (PO interaction)

\[
I(a,b)=\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)
\]

- **Matched signed power** =
  $\mathbb{P}\bigl(p(|I|)<\alpha \;\wedge\; \mathrm{sign}(\hat I)=\mathrm{sign}(\lambda)\bigr)$.
- $\lambda>0$ plant → expect $\hat I>0$ (synergy).
- $\lambda<0$ plant → expect $\hat I<0$ (antagonism).

### B. Detectability (CF / DR / RF on combos)

| Combo | Question |
|---|---|
| `power_a`, `power_b` | Can a **single** modality see the shift? |
| `power_pair` | Does **concatenating the planted pair** help? |
| `power_tav` | Does adding the third modality help further? |
| **Fusion lift** `power_pair - \max(power_a,power_b)` | Operational twin of positive $I$ |

## Claims to validate (unique insights)

1. **Synergy bridge.** Under $\lambda>0$: matched signed power ↑ **and** fusion lift $>0$
   (pair beats better single). Corr(matched signed-power, fusion lift) $>0$ across cells.
2. **Antagonism.** Under $\lambda<0$: matched (neg) signed power ↑; fusion lift shrinks/reverses —
   concatenating does **not** buy a free lunch.
3. **Main-effect control.** ±mean_shift on modality $m$: `power_single(m)` high;
   signed $I$ on pairs ≈ null; pair/TAV ≈ single (no interaction credit).
4. **±symmetry.** For two-sided CF/DR/RF, power(+mean) ≈ power(−mean).
5. **TAV discipline.** Pair-planted interaction: `power_tav` ≉ much above `power_pair`
   (third channel is mostly noise). Mean-shift: TAV tracks pairs that contain the shifted mod.

## Why this delays delivery (worth it)

Polished T×A/T×V concept tables answer *coverage*.
This board answers *mechanism ↔ detectability*:
**when PO says synergy, do CF/DR/RF actually gain from fusion?**
That link is the unique insight for the multimodal attribution claim.

## Artifacts

| File | Content |
|---|---|
| `power_raw.csv` | per-rep CF/DR/RF (+ optional interact_PO p) |
| `signed_I_raw.csv` | $\hat I$, sign, p, PO pieces |
| `single_vs_pair_contrast.csv` | fusion lift × matched signed power |
| `mean_shift_direction_board.csv` | ±mean single/pair/TAV |
| `SENSITIVITY_BOARD.md` | auto insights + corr |
