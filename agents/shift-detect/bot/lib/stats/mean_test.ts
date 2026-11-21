import { assertMatrix, columnMeans, columnVars } from "./matrices.js";
import type { DetectResult } from "./types.js";

/** Diagonal Hotelling-style mean-difference statistic with normal p approx. */
export function meanDiffTest(X: number[][], Y: number[][]): DetectResult {
  assertMatrix(X, "X");
  assertMatrix(Y, "Y");
  const m1 = columnMeans(X);
  const m2 = columnMeans(Y);
  const v1 = columnVars(X, m1);
  const v2 = columnVars(Y, m2);
  const n1 = X.length;
  const n2 = Y.length;
  let stat = 0;
  for (let j = 0; j < m1.length; j++) {
    const pooled = v1[j] / n1 + v2[j] / n2;
    if (pooled <= 1e-12) continue;
    const d = m1[j] - m2[j];
    stat += (d * d) / pooled;
  }
  // Chi-square approx with p degrees; use Wilson-Hilferty normal approx for p-value.
  const p = m1.length;
  const h = Math.cbrt(stat / p);
  const mu = 1 - 2 / (9 * p);
  const sigma = Math.sqrt(2 / (9 * p));
  const z = (h - mu) / Math.max(sigma, 1e-8);
  const t = 1 / (1 + 0.3275911 * Math.abs(z));
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  let pValue = Math.exp(-z * z) * poly;
  if (z < 0) pValue = 1 - pValue;
  pValue = Math.min(1, Math.max(1e-6, pValue));

  return {
    method: "mean_diff",
    statistic: stat,
    pValue,
    rejectAt005: pValue < 0.05,
    rejectAt001: pValue < 0.01,
    nRef: n1,
    nNew: n2,
    nFeatures: p,
    notes: [
      "Diagonal Hotelling mean test (Bai/Srivastava-style high-dim flavor).",
      "Sensitive to covariate location shift; weak on pure concept drift.",
    ],
  };
}
