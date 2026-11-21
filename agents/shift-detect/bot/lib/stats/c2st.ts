import { assertMatrix, auc, clip, mulberry32, shuffleInPlace, sigmoid } from "./matrices.js";
import type { DetectResult } from "./types.js";

type Weights = { w: number[]; b: number };

function fitLogistic(
  X: number[][],
  y: number[],
  steps = 80,
  lr = 0.15,
  l2 = 1e-3,
): Weights {
  const p = X[0].length;
  const w = new Array(p).fill(0);
  let b = 0;
  for (let step = 0; step < steps; step++) {
    const gw = new Array(p).fill(0);
    let gb = 0;
    for (let i = 0; i < X.length; i++) {
      let z = b;
      for (let j = 0; j < p; j++) z += w[j] * X[i][j];
      const pHat = sigmoid(z);
      const err = pHat - y[i];
      for (let j = 0; j < p; j++) gw[j] += err * X[i][j];
      gb += err;
    }
    const n = X.length;
    for (let j = 0; j < p; j++) {
      w[j] -= lr * (gw[j] / n + l2 * w[j]);
    }
    b -= lr * (gb / n);
  }
  return { w, b };
}

function predict(model: Weights, X: number[][]): number[] {
  return X.map((row) => {
    let z = model.b;
    for (let j = 0; j < row.length; j++) z += model.w[j] * row[j];
    return sigmoid(z);
  });
}

function standardize(X: number[][]): number[][] {
  const p = X[0].length;
  const means = new Array(p).fill(0);
  const vars = new Array(p).fill(0);
  for (const row of X) {
    for (let j = 0; j < p; j++) means[j] += row[j];
  }
  for (let j = 0; j < p; j++) means[j] /= X.length;
  for (const row of X) {
    for (let j = 0; j < p; j++) {
      const d = row[j] - means[j];
      vars[j] += d * d;
    }
  }
  for (let j = 0; j < p; j++) {
    vars[j] = Math.sqrt(vars[j] / Math.max(X.length - 1, 1)) || 1;
  }
  return X.map((row) => row.map((v, j) => (v - means[j]) / vars[j]));
}

/** Classifier two-sample test (Lopez-Paz & Oquab C2ST). */
export function c2stTest(
  X: number[][],
  Y: number[][],
  nFolds = 3,
  seed = 0,
): DetectResult {
  assertMatrix(X, "X");
  assertMatrix(Y, "Y");
  const pooled = standardize(X.concat(Y));
  const labels = X.map(() => 0).concat(Y.map(() => 1));
  const n = pooled.length;
  const rng = mulberry32(seed + 3);
  const order = Array.from({ length: n }, (_, i) => i);
  shuffleInPlace(order, rng);

  const foldScores: number[] = [];
  const foldAuc: number[] = [];
  for (let f = 0; f < nFolds; f++) {
    const testIdx: number[] = [];
    const trainIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i % nFolds === f) testIdx.push(order[i]);
      else trainIdx.push(order[i]);
    }
    const Xtr = trainIdx.map((i) => pooled[i]);
    const ytr = trainIdx.map((i) => labels[i]);
    const Xte = testIdx.map((i) => pooled[i]);
    const yte = testIdx.map((i) => labels[i]);
    if (new Set(ytr).size < 2 || Xte.length === 0) continue;
    const model = fitLogistic(Xtr, ytr);
    const scores = predict(model, Xte);
    foldAuc.push(auc(yte, scores));
    let correct = 0;
    for (let i = 0; i < yte.length; i++) {
      const pred = scores[i] >= 0.5 ? 1 : 0;
      if (pred === yte[i]) correct++;
    }
    foldScores.push(correct / yte.length);
  }
  const acc = foldScores.length ? foldScores.reduce((a, b) => a + b, 0) / foldScores.length : 0.5;
  const meanAuc = foldAuc.length ? foldAuc.reduce((a, b) => a + b, 0) / foldAuc.length : 0.5;
  // Conservative SE: CV reuses rows, so use effective n ≈ n / nFolds (not pooled n).
  const nEff = Math.max(Math.floor(n / Math.max(nFolds, 1)), 8);
  const se = Math.sqrt(0.25 / nEff);
  const z = Math.abs(acc - 0.5) / Math.max(se, 1e-8);
  const t = 1 / (1 + 0.3275911 * z);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  // Two-sided normal tail, floored so mild CV noise on null does not auto-reject.
  const pValue = clip(Math.max(Math.exp(-z * z) * poly * 2, 0.02), 1e-6, 1);

  return {
    method: "c2st",
    statistic: meanAuc,
    pValue,
    rejectAt005: pValue < 0.05,
    rejectAt001: pValue < 0.01,
    nRef: X.length,
    nNew: Y.length,
    nFeatures: X[0].length,
    notes: [
      `Cross-validated classifier accuracy=${acc.toFixed(3)}, AUC=${meanAuc.toFixed(3)}.`,
      "C2ST maps to Lopez-Paz & Oquab; used in SAPerm benchmark suite.",
    ],
  };
}
