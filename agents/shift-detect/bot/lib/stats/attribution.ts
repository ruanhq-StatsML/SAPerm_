import { assertMatrix, columnMeans, mean } from "./matrices.js";
import { conceptScores } from "./concept.js";
import type { AttributeResult, AttributionHit } from "./types.js";

function rankShares(
  scores: Record<string, number>,
): AttributionHit[] {
  const entries = Object.entries(scores).map(([name, score]) => ({
    name,
    score: Math.abs(score),
  }));
  const total = entries.reduce((a, b) => a + b.score, 0) || 1;
  entries.sort((a, b) => b.score - a.score);
  return entries.map((e, i) => ({
    name: e.name,
    score: e.score,
    share: e.score / total,
    rank: i + 1,
  }));
}

/** Block-level mean absolute difference attribution (tabular-drift-attribution style). */
export function blockMeanAttribution(
  XRef: number[][],
  XNew: number[][],
  blocks: Record<string, number[]>,
  groundTruth: string[] = [],
  featureNames?: string[],
): AttributeResult {
  assertMatrix(XRef, "XRef");
  assertMatrix(XNew, "XNew");
  const mRef = columnMeans(XRef);
  const mNew = columnMeans(XNew);
  const blockScores: Record<string, number> = {};
  for (const [name, idxs] of Object.entries(blocks)) {
    let s = 0;
    for (const j of idxs) s += Math.abs(mNew[j] - mRef[j]);
    blockScores[name] = s / Math.max(idxs.length, 1);
  }
  const ranked = rankShares(blockScores);
  const top = ranked[0]?.name ?? null;
  const topHit =
    groundTruth.length === 0 ? null : top !== null && groundTruth.includes(top);

  // Also emit top features for readability.
  const featScores: Record<string, number> = {};
  for (let j = 0; j < mRef.length; j++) {
    const name = featureNames?.[j] ?? `f${j}`;
    featScores[name] = Math.abs(mNew[j] - mRef[j]);
  }
  const featRanked = rankShares(featScores).slice(0, 10);

  return {
    method: "block_mean_diff",
    ranked: ranked.concat(
      featRanked.map((h) => ({ ...h, name: `feature:${h.name}` })),
    ),
    top,
    topHit,
    groundTruth,
  };
}

/** Leave-one-covariate-out style PO-risk contribution for concept drivers. */
export function locoPoAttribution(
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
  featureNames?: string[],
  groundTruth: string[] = [],
): AttributeResult {
  assertMatrix(XRef, "XRef");
  assertMatrix(XNew, "XNew");
  const base = conceptScores(XRef, yRef, XNew, yNew).poRisk;
  const p = XRef[0].length;
  const scores: Record<string, number> = {};
  for (let j = 0; j < p; j++) {
    const drop = (M: number[][]) => M.map((row) => row.filter((_, k) => k !== j));
    const dropped = conceptScores(drop(XRef), yRef, drop(XNew), yNew).poRisk;
    const name = featureNames?.[j] ?? `f${j}`;
    scores[name] = Math.max(0, base - dropped);
  }
  // If LOCO is flat, fall back to |mean(y)| correlation proxy via feature mean shift * outcome gap.
  if (mean(Object.values(scores)) < 1e-9) {
    const mRef = columnMeans(XRef);
    const mNew = columnMeans(XNew);
    for (let j = 0; j < p; j++) {
      const name = featureNames?.[j] ?? `f${j}`;
      scores[name] = Math.abs(mNew[j] - mRef[j]);
    }
  }
  const ranked = rankShares(scores);
  const top = ranked[0]?.name ?? null;
  const topHit =
    groundTruth.length === 0 ? null : top !== null && groundTruth.includes(top);
  return {
    method: "loco_po_risk",
    ranked,
    top,
    topHit,
    groundTruth,
  };
}
