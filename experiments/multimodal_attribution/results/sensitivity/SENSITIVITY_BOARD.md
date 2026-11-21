# Sensitivity: ±modality shift × single vs pair × signed PO interaction

_α=0.05; auto-generated from `power_raw.csv` / `signed_I_raw.csv`_

## Claim map

| Quantity | Meaning |
|---|---|
| `power_a` / `power_b` | CF/DR/RF power on each single modality |
| `power_pair` | power on the planted pair concat |
| `power_tav` | power on text+audio+video |
| `pair_minus_max_single` | **fusion lift** vs better single |
| `I_cal=I-PO(full)` | **calibrated** interaction (raw $I$ is often $+$-biased) |
| `signed_power_matched` | $P(\mathrm{reject}\wedge \mathrm{sign}(I_{cal})=\mathrm{sign}(\lambda))$ |

### Insight hypothesis

1. **$\lambda>0$** → `I_cal>0`, high matched signed power, `power_pair > max(power_a,power_b)` (synergy: fusion beats singles).
2. **$\lambda<0$** → `I_cal<0`, matched (neg) signed power up, fusion lift shrinks.
3. **TAV vs pair**: under a *pair* plant, TAV should not beat the planted pair by much (third modality is noise); under mean_shift of one mod, TAV ≈ pairs containing that mod.
4. **±mean_shift** → single of shifted modality carries power; signed PO near α (main effect ≠ interaction); +/− directions ~symmetric for two-sided tests.

## Interaction_shift: single vs pair vs TAV vs signed power

### cfperm

| pair | λ | a | b | pair | TAV | pair−max₁ | TAV−pair | I_cal | frac+(cal) | matched signed-pow | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.80 | 0.90 | 0.00 | 0.54 | 0.62 | -0.36 | +0.08 | -14.937 | 0.08 | 0.00 | 24 |

### drperm

| pair | λ | a | b | pair | TAV | pair−max₁ | TAV−pair | I_cal | frac+(cal) | matched signed-pow | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.80 | 0.30 | 0.00 | 0.00 | 0.00 | -0.30 | +0.00 | -14.937 | 0.08 | 0.00 | 24 |

### rfperm

| pair | λ | a | b | pair | TAV | pair−max₁ | TAV−pair | I_cal | frac+(cal) | matched signed-pow | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.80 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | +0.00 | -14.937 | 0.08 | 0.00 | 24 |

## Argument (signed power ↔ fusion lift ↔ modality hierarchy)

1. **Plant**: `interaction_shift` with signed $\lambda$ injects a product on $(a,b)$ in $X$ and $Y$. $\lambda>0$ = synergistic plant; $\lambda<0$ = antagonistic plant.
2. **Mechanism readout**: $I=\mathrm{PO}(-a)+\mathrm{PO}(-b)-\mathrm{PO}(-a,-b)$, calibrated $I_{cal}=I-\mathrm{PO}(\mathrm{full})$. Matched signed power rejects *and* matches $\mathrm{sign}(I_{cal})=\mathrm{sign}(\lambda)$. Empirically raw $I$ stays positive under null — **do not** sign off raw $I$.
3. **Detectability readout (CF/DR/RF)**: compare `power_a`, `power_b`, `power_pair`, `power_tav`. Fusion lift is the operational twin of $I_{cal}>0$: synergistic label decisions should make the pair beat either single.
4. **Sentiment / decision**: $I_{cal}>0$ ⇒ modalities *agree* in the label head (fuse); $I_{cal}<0$ ⇒ *conflict* (gate / late-fuse / trust dominant channel).
5. **Association**: corr(matched signed-power, fusion lift) across $\pm\lambda$ links mechanism to detection. ±mean_shift is the additive control.

