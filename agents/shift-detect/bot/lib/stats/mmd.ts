import {
  assertMatrix,
  median,
  mulberry32,
  pairwiseSqDist,
  sampleRows,
  shuffleInPlace,
} from "./matrices.js";
import type { DetectResult } from "./types.js";

function rbfGamma(X: number[][], Y: number[][]): number {
  const pool = X.concat(Y);
  const dists: number[] = [];
  const n = Math.min(pool.length, 120);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = pairwiseSqDist(pool[i], pool[j]);
      if (d > 0) dists.push(d);
    }
  }
  const med = median(dists);
  if (!Number.isFinite(med) || med <= 0) return 1;
  return 1 / (2 * med);
}

function kernelMean(A: number[][], B: number[][], gamma: number, diagonalZero: boolean): number {
  let s = 0;
  let count = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B.length; j++) {
      if (diagonalZero && A === B && i === j) continue;
      const k = Math.exp(-gamma * pairwiseSqDist(A[i], B[j]));
      s += k;
      count++;
    }
  }
  return count === 0 ? 0 : s / count;
}

export function mmdStatistic(X: number[][], Y: number[][], seed = 0): number {
  assertMatrix(X, "X");
  assertMatrix(Y, "Y");
  const rng = mulberry32(seed + 17);
  const Xs = sampleRows(X, 200, rng);
  const Ys = sampleRows(Y, 200, rng);
  const gamma = rbfGamma(Xs, Ys);
  const kxx = kernelMean(Xs, Xs, gamma, true);
  const kyy = kernelMean(Ys, Ys, gamma, true);
  const kxy = kernelMean(Xs, Ys, gamma, false);
  return Math.max(0, kxx + kyy - 2 * kxy);
}

export function mmdTest(
  X: number[][],
  Y: number[][],
  nPerm = 40,
  seed = 0,
): DetectResult {
  assertMatrix(X, "X");
  assertMatrix(Y, "Y");
  const observed = mmdStatistic(X, Y, seed);
  const pooled = X.concat(Y);
  const n = X.length;
  const rng = mulberry32(seed + 11);
  let exceed = 0;
  for (let i = 0; i < nPerm; i++) {
    const idx = Array.from({ length: pooled.length }, (_, k) => k);
    shuffleInPlace(idx, rng);
    const Xp = idx.slice(0, n).map((j) => pooled[j]);
    const Yp = idx.slice(n).map((j) => pooled[j]);
    if (mmdStatistic(Xp, Yp, seed + 100 + i) >= observed - 1e-12) exceed++;
  }
  const pValue = (1 + exceed) / (1 + nPerm);
  return {
    method: "mmd",
    statistic: observed,
    pValue,
    rejectAt005: pValue < 0.05,
    rejectAt001: pValue < 0.01,
    nRef: X.length,
    nNew: Y.length,
    nFeatures: X[0].length,
    notes: [
      "RBF-kernel MMD with median heuristic bandwidth.",
      "Permutation p-value; mirrors Gretton et al. KMMD / SAPerm benchmark kmmd.",
    ],
  };
}
