# Sensitivity: ±modality shift × single vs pair × signed PO interaction

_α=0.05; auto-generated from `power_raw.csv` / `signed_I_raw.csv`_

## Claim map

| Quantity | Meaning |
|---|---|
| `power_a` / `power_b` | CF/DR/RF power on each single modality |
| `power_pair` | power on the planted pair concat |
| `power_tav` | power on text+audio+video |
| `pair_minus_max_single` | **fusion lift** vs better single |
| `mean_I` / `frac_I_pos` | PO interaction direction |
| `signed_power_matched` | $P(\mathrm{reject}\wedge \mathrm{sign}(\hat I)=\mathrm{sign}(\lambda))$ |

### Insight hypothesis

1. **$\lambda>0$** → `mean_I>0`, high matched signed power, `power_pair > max(power_a,power_b)` (synergy: fusion beats singles).
2. **$\lambda<0$** → `mean_I<0`, matched (neg) signed power up, fusion lift shrinks.
3. **TAV vs pair**: under a *pair* plant, TAV should not beat the planted pair by much (third modality is noise); under mean_shift of one mod, TAV ≈ pairs containing that mod.
4. **±mean_shift** → single of shifted modality carries power; signed PO near α (main effect ≠ interaction); +/− directions ~symmetric for two-sided tests.

## Argument (signed power ↔ fusion lift ↔ modality hierarchy)

1. **Plant**: `interaction_shift` with signed $\lambda$ injects a product on $(a,b)$ in $X$ and $Y$. $\lambda>0$ = synergistic plant; $\lambda<0$ = antagonistic plant.
2. **Mechanism readout**: $I=\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)$; matched signed power = rate of rejecting *and* getting $\mathrm{sign}(\hat I)=\mathrm{sign}(\lambda)$.
3. **Detectability readout (CF/DR/RF)**: compare `power_a`, `power_b`, `power_pair`, `power_tav`. Fusion lift `power_pair - max(power_a,power_b)` is the operational twin of positive $I$: if the pair is truly synergistic, the concatenated test should beat either single.
4. **Association**: across $\pm\lambda$ and pairs, corr(matched signed-power, fusion lift) validates that PO interaction and perm-test fusion tell the same story.
5. **Control**: ±mean_shift moves one modality — singles light up, signed $I$ stays null, pair/TAV ≈ single (no extra interaction credit).

