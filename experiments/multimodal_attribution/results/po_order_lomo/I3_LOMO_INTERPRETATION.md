# $I^{(3)}$ + LOMO power — polished interpretation

Source LaTeX: `experiments/multimodal_attribution/latex/higher_order_lomo_interpretation.tex`.

## One-paragraph abstract (English)

One step further, we quantify how efficiently modalities interact for domain / label characterization. Writing $\Delta_{\mathrm{PO}}(\mathcal{S})$ for the PO-risk increment attributable to modality subset $\mathcal{S}$, the three-way interaction among $M_1,M_2,M_3$ (text/audio/video) is the inclusion–exclusion residual
$$
I^{(3)}=\Delta_{\mathrm{PO}}(M_1,M_2,M_3)-\sum_{i<j}\Delta_{\mathrm{PO}}(M_i,M_j)+\sum_i\Delta_{\mathrm{PO}}(M_i).
$$
Complementing this residual, the leave-one-modality-out power gap $\Delta^{\mathrm{pow}}_m=\mathrm{Power}(\mathrm{full})-\mathrm{Power}(\mathrm{full}\setminus\{m\})$ shows which channel moves detection: positive loss means the modality helps the joint test; negative loss means early fusion is stronger without it. LOMO is a **cross-validation** of $I^{(3)}$ (sign agreement), not a second attribution theorem: $I^{(3)}>0$ ↔ jointly positive LOMO gaps; $I^{(3)}\approx 0$ ↔ pairwise mechanisms; $I^{(3)}<0$ or $\mathrm{Power}(\mathrm{TAV})<\max\mathrm{Power}(\mathrm{pair})$ ↔ over-complete / conflicting triple fusion. Disagreement flags noise or baseline mismatch—not unique modality shares. Practically, $(I^{(3)},\{\Delta^{\mathrm{pow}}_m\})$ still decides keep / drop / late-fuse.

## Empirical LOMO cross-check (concept drift, already in board)

| setting | method | vs best pair (TAV − best) |
|---|---|---:|
| top30 | CFPerm | −0.200 (TAV 0.400 < TA 0.600) |
| top30 | DRPerm | −0.100 |
| top30 | RFPerm | −0.067 |
| pc31_100 | RFPerm | −0.120 (TAV 0.573 < TV 0.693) |

Negative vs-best-pair gaps ⇒ triple early fusion inefficient — the deployment face of $I^{(3)}\le 0$ / over-complete.
