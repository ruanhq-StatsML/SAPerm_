# Concept-drift synthetic: power × PO-risk cross-check

With vs without meta. $\rho_{PO}=PO(Z{+}M)/PO(Z)$.
B=8, N_perm=40, n=160, p_z=8, p_m=4.

| DGP | RF e→m | DR e→m | ρ_PO | CF meta−placebo | signature | match |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| concept_embed | 0.00→0.12 (+0.12) | 0.50→0.25 (-0.25) | 1.02 | +0.25 | localize_ok | · |
| concept_meta | 0.12→0.00 (-0.12) | 0.12→0.25 (+0.12) | 1.05 | -0.12 | mixed | · |
| covariate_meta | 0.00→0.00 (+0.00) | 0.00→0.00 (+0.00) | 0.69 | +0.12 | absorb_ok | ✓ |
| null | 0.00→0.12 (+0.12) | 0.12→0.12 (+0.00) | 1.09 | +0.00 | null_calibration | ✓ |

## How to read (cross-validation = interaction evidence)

- **covariate_meta**: $\rho_{PO}\ll 1$, DR power does not need to rise → meta *absorbs* (interaction in nuisance).
- **concept_embed**: $\rho_{PO}\approx 1$, DR high both sides → concept residual; meta cannot erase it.
- **concept_meta**: CFPerm meta $\gg$ placebo → localization names the interacting block; PO stays informative.
- Joint $(\Delta$Power, $\rho_{PO}$, $\Delta$CF) matching the DGP design *is* the cross-check.
