# Dual ranking insights: strength \(A_j\) vs PO-risk \(\Delta^{\mathrm{PO}}_j\)

DSS is **not** only a variance / \(\eta^2\) story. The same fits produce two rankings; ranking recovery tells you which scientific question each ranking answers.

## Two rankings from one fit

| Ranking | Score | What it sorts |
| --- | --- | --- |
| **Strength** | \(A_j=\eta^2_{1,j}-\eta^2_{0,j}\) | Did dependence **intensity** rise or fall? (signed redundancy) |
| **PO-risk** | \(\Delta^{\mathrm{PO}}_j=R(0)-R(\tau_j)\) | Did the **conditional relationship** move? (including flips) |

\(\eta^2\) is explained-variance / correlation-ratio. PO-risk is a contrast risk: how much better the fitted map difference \(\tau=m_1-m_0\) predicts the pseudo-outcome than the zero map. Large PO-risk ≠ stronger dependence.

## What ranking recovery buys you

Against a graded oracle ladder (weaken → … → strengthen, flip with \(A=0\)):

1. **High Spearman(\(A\)) for strength, low for PO-risk**  
   Strength recovers the signed intensity order; PO-risk does not, because it scores *change of map*, not *change of \(\eta^2\)*.  
   Insight: if you only ranked by PO-risk (or CMS / \(|\Delta\rho|\)), you would mis-order “who got more redundant.”

2. **Flip: \(A\approx 0\), PO-risk huge**  
   Residual / factor channel reversed but reconstructibility unchanged.  
   Insight: correlation-sign or hedge-ratio flip without a diversification regime change. Strength alone would miss the alarm; PO-risk alone would look like “biggest shift” and invite the wrong conclusion (as if redundancy exploded).

3. **Sign agreement on strength ≈ 1, on PO-risk ≈ 0.5**  
   Only \(A_j\) tells strengthen vs weaken. PO-risk is effectively unsigned for that question.  
   Insight: for risk-budget / residual-redundancy monitoring, use \(\mathrm{sign}(A_j)\); for relationship / hedging-map monitoring, use PO-risk rank.

4. **On factor residuals**  
   After \(F_j=\beta'G+\varepsilon_j\):  
   - \(A_j(\varepsilon)\): idiosyncratic block more/less collinear → residual diversification up/down.  
   - \(\Delta^{\mathrm{PO}}_j(\varepsilon)\): residual conditional map moved (e.g. long–short residual link flipped).  
   - Incremental \(V_{j\leftarrow k}\): *which* other residual’s unique information changed.  
   Insight: a name can have large PO-risk (map flipped) and near-zero \(A\) (still same residual R²), or large negative \(A\) (less explained by peers) with moderate PO-risk (map changed little in shape but noise rose).

## Operational read-out (quant)

| Pattern | Strength \(A\) | PO-risk | Actionable reading |
| --- | ---: | ---: | --- |
| Redundancy regime | ↓ large | small–mid | Residual diversification improved; peers explain less |
| Crowding / collapse | ↑ large | small–mid | Residuals more reconstructible; unique alpha harder |
| Hedge / sign flip | ≈ 0 | ↑ large | Relationship reversed; intensity unchanged — re-check hedges, not just variance |
| Both large | ↑ or ↓ | ↑ large | Intensity *and* map moved — full residual redesign |

Ranking recovery is the audit that these two rankings are not interchangeable: Spearman against oracle \(A\) is high for strength and low for PO-risk by design. That gap *is* the insight, not a failure of either score.
