/**
 * Pathway-style block attribution.
 * Mirrors tabular-drift-attribution: for each block, replace today's block
 * with a reference-neighbor draw conditioned on the remaining columns, then
 * measure how much distance that rewrite removes.
 */
import { assertMatrix, mulberry32, pairwiseSqDist } from "./matrices.js";
import type { AttributeResult, AttributionHit } from "./types.js";

function nearestRefRow(
  XRef: number[][],
  condIdx: number[],
  query: number[],
): number[] {
  let best = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < XRef.length; i++) {
    let d = 0;
    for (const j of condIdx) {
      const diff = XRef[i][j] - query[j];
      d += diff * diff;
    }
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return XRef[best];
}

function blockDistance(A: number[][], B: number[][], idxs: number[]): number {
  let s = 0;
  const n = Math.min(A.length, B.length);
  for (let i = 0; i < n; i++) {
    for (const j of idxs) {
      const d = A[i][j] - B[i][j];
      s += Math.abs(d);
    }
  }
  return s / Math.max(n * idxs.length, 1);
}

function rankShares(scores: Record<string, number>): AttributionHit[] {
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

/**
 * Counterfactual pathway attribution.
 * For block b: Z_{t,b} <- nearest reference row under X_{-b}=X_{t,-b}.
 * Contribution c(b) = d(X_b, Z_b) / sum_{b'} d(X_{b'}, Z_{b'}).
 */
export function pathwayBlockAttribution(
  XRef: number[][],
  XNew: number[][],
  blocks: Record<string, number[]>,
  groundTruth: string[] = [],
  seed = 0,
): AttributeResult & {
  method: "pathway_counterfactual";
  rowDiagnostics: Array<{ t: number; top: string; shares: Record<string, number> }>;
} {
  assertMatrix(XRef, "XRef");
  assertMatrix(XNew, "XNew");
  const p = XRef[0].length;
  const blockNames = Object.keys(blocks);
  if (blockNames.length === 0) {
    throw new Error("blocks must be non-empty for pathway attribution");
  }

  const agg: Record<string, number> = {};
  for (const name of blockNames) agg[name] = 0;

  const rng = mulberry32(seed + 91);
  const maxRows = Math.min(XNew.length, 120);
  const rowDiagnostics: Array<{
    t: number;
    top: string;
    shares: Record<string, number>;
  }> = [];

  for (let t = 0; t < maxRows; t++) {
    // mild subsample stability
    if (XNew.length > maxRows && rng() > maxRows / XNew.length) continue;
    const x = XNew[t];
    const rowScores: Record<string, number> = {};
    for (const [name, idxs] of Object.entries(blocks)) {
      const condIdx = Array.from({ length: p }, (_, j) => j).filter(
        (j) => !idxs.includes(j),
      );
      const donor = nearestRefRow(XRef, condIdx, x);
      let d = 0;
      for (const j of idxs) d += Math.abs(x[j] - donor[j]);
      rowScores[name] = d / Math.max(idxs.length, 1);
      agg[name] += rowScores[name];
    }
    const shares = rankShares(rowScores);
    const shareMap: Record<string, number> = {};
    for (const h of shares) shareMap[h.name] = h.share;
    rowDiagnostics.push({
      t,
      top: shares[0]?.name ?? "",
      shares: shareMap,
    });
  }

  const ranked = rankShares(agg);
  const top = ranked[0]?.name ?? null;
  const topHit =
    groundTruth.length === 0 ? null : top !== null && groundTruth.includes(top);

  return {
    method: "pathway_counterfactual",
    ranked,
    top,
    topHit,
    groundTruth,
    rowDiagnostics: rowDiagnostics.slice(0, 8),
  };
}

/** Convenience: total mean embedding distance between windows. */
export function windowDistance(XRef: number[][], XNew: number[][]): number {
  const n = Math.min(XRef.length, XNew.length, 80);
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.sqrt(pairwiseSqDist(XRef[i], XNew[i]));
  return s / Math.max(n, 1);
}
