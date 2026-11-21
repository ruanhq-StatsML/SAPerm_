import { assertMatrix, clip, mean, mulberry32, shuffleInPlace, sigmoid } from "./matrices.js";
import type { DetectResult } from "./types.js";

type Model = { w: number[]; b: number };

function fitRidgeLogit(X: number[][], y: number[], steps = 100, lr = 0.2, l2 = 0.05): Model {
  const p = X[0].length;
  const w = new Array(p).fill(0);
  let b = 0;
  const rate = mean(y);
  if (new Set(y.map((v) => (v >= 0.5 ? 1 : 0))).size < 2) {
    return { w, b: Math.log(clip(rate, 1e-3, 1 - 1e-3) / (1 - clip(rate, 1e-3, 1 - 1e-3))) };
  }
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
    for (let j = 0; j < p; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    b -= lr * (gb / n);
  }
  return { w, b };
}

function predictProba(model: Model, X: number[][]): number[] {
  return X.map((row) => {
    let z = model.b;
    for (let j = 0; j < row.length; j++) z += model.w[j] * row[j];
    return sigmoid(z);
  });
}

function brier(y: number[], p: number[]): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const pi = clip(p[i], 1e-4, 1 - 1e-4);
    const d = pi - y[i];
    s += d * d;
  }
  return s / Math.max(y.length, 1);
}

/**
 * Pseudo-outcome / delta_cf style concept score.
 * Fit reference outcome model on (X_ref, y_ref), compare to a model
 * fit on the new window; report mean squared gap on the new covariates
 * and a skill-gated excess Brier gap.
 */
export function conceptScores(
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
): { poRisk: number; deltaCf: number; brierRef: number; brierLocal: number } {
  assertMatrix(XRef, "XRef");
  assertMatrix(XNew, "XNew");
  if (yRef.length !== XRef.length || yNew.length !== XNew.length) {
    throw new Error("y length must match X rows");
  }
  const ref = fitRidgeLogit(XRef, yRef);
  const local = fitRidgeLogit(XNew, yNew);
  const pRefOnNew = predictProba(ref, XNew);
  const pLocalOnNew = predictProba(local, XNew);
  let po = 0;
  for (let i = 0; i < XNew.length; i++) {
    const d = pLocalOnNew[i] - pRefOnNew[i];
    po += d * d;
  }
  po /= XNew.length;
  const brierRef = brier(yNew, pRefOnNew);
  const brierLocal = brier(yNew, pLocalOnNew);
  const chance = brier(
    yNew,
    yNew.map(() => mean(yNew)),
  );
  const skill = clip((chance - brierLocal) / Math.max(chance, 1e-6), 0, 1);
  const deltaCf = Math.max(0, brierRef - brierLocal) * skill;
  return { poRisk: po, deltaCf, brierRef, brierLocal };
}

function permuteConceptP(
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
  statistic: "poRisk" | "deltaCf",
  nPerm = 40,
  seed = 0,
): { observed: number; pValue: number; scores: ReturnType<typeof conceptScores> } {
  const scores = conceptScores(XRef, yRef, XNew, yNew);
  const observed = scores[statistic];
  const rng = mulberry32(seed + (statistic === "poRisk" ? 41 : 43));
  const yPool = yRef.concat(yNew);
  let exceed = 0;
  for (let i = 0; i < nPerm; i++) {
    const shuffled = yPool.slice();
    shuffleInPlace(shuffled, rng);
    const yR = shuffled.slice(0, yRef.length);
    const yN = shuffled.slice(yRef.length);
    const s = conceptScores(XRef, yR, XNew, yN)[statistic];
    if (s >= observed - 1e-12) exceed++;
  }
  return {
    observed,
    pValue: (1 + exceed) / (1 + nPerm),
    scores,
  };
}

export function poRiskTest(
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
): DetectResult {
  const { observed, pValue, scores } = permuteConceptP(
    XRef,
    yRef,
    XNew,
    yNew,
    "poRisk",
    40,
    5,
  );
  return {
    method: "po_risk",
    statistic: observed,
    pValue,
    rejectAt005: pValue < 0.05,
    rejectAt001: pValue < 0.01,
    nRef: XRef.length,
    nNew: XNew.length,
    nFeatures: XRef[0].length,
    notes: [
      `delta_cf=${scores.deltaCf.toFixed(4)}, brier_ref=${scores.brierRef.toFixed(4)}, brier_local=${scores.brierLocal.toFixed(4)}.`,
      "Y-permutation p-value on PO-risk (OnlineMonitoring / DRPerm flavor).",
    ],
  };
}

export function deltaCfTest(
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
): DetectResult {
  const { observed, pValue, scores } = permuteConceptP(
    XRef,
    yRef,
    XNew,
    yNew,
    "deltaCf",
    40,
    7,
  );
  return {
    method: "delta_cf",
    statistic: observed,
    pValue,
    rejectAt005: pValue < 0.05,
    rejectAt001: pValue < 0.01,
    nRef: XRef.length,
    nNew: XNew.length,
    nFeatures: XRef[0].length,
    notes: [
      `po_risk=${scores.poRisk.toFixed(4)}.`,
      "Y-permutation p-value on skill-gated Brier gap (primary_driver delta_cf).",
    ],
  };
}

export function labelShiftScore(yRef: number[], yNew: number[]): number {
  const p1 = mean(yRef.map((v) => (v >= 0.5 ? 1 : 0)));
  const p2 = mean(yNew.map((v) => (v >= 0.5 ? 1 : 0)));
  return Math.abs(p1 - p2);
}
