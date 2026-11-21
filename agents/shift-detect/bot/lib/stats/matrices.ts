/** Lightweight matrix helpers (no external numeric deps). */

export function assertMatrix(X: number[][], name: string): void {
  if (!Array.isArray(X) || X.length === 0) {
    throw new Error(`${name} must be a non-empty row matrix`);
  }
  const p = X[0]?.length ?? 0;
  if (p === 0) throw new Error(`${name} has zero columns`);
  for (let i = 0; i < X.length; i++) {
    if (!X[i] || X[i].length !== p) {
      throw new Error(`${name} row ${i} has inconsistent width`);
    }
    for (const v of X[i]) {
      if (!Number.isFinite(v)) {
        throw new Error(`${name} contains a non-finite value`);
      }
    }
  }
}

export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffleInPlace<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

export function sampleRows(X: number[][], size: number, rng: () => number): number[][] {
  if (X.length <= size) return X.map((r) => r.slice());
  const idx = Array.from({ length: X.length }, (_, i) => i);
  shuffleInPlace(idx, rng);
  return idx.slice(0, size).map((i) => X[i].slice());
}

export function columnMeans(X: number[][]): number[] {
  const p = X[0].length;
  const means = new Array(p).fill(0);
  for (const row of X) {
    for (let j = 0; j < p; j++) means[j] += row[j];
  }
  for (let j = 0; j < p; j++) means[j] /= X.length;
  return means;
}

export function columnVars(X: number[][], means?: number[]): number[] {
  const m = means ?? columnMeans(X);
  const p = X[0].length;
  const vars = new Array(p).fill(0);
  for (const row of X) {
    for (let j = 0; j < p; j++) {
      const d = row[j] - m[j];
      vars[j] += d * d;
    }
  }
  const denom = Math.max(X.length - 1, 1);
  for (let j = 0; j < p; j++) vars[j] /= denom;
  return vars;
}

export function pairwiseSqDist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

export function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return 0.5 * (sorted[mid - 1] + sorted[mid]);
  }
  return sorted[mid];
}

export function clip(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** AUC for binary labels in {0,1} and scores. */
export function auc(y: number[], scores: number[]): number {
  const pairs = y.map((label, i) => ({ label, score: scores[i] }));
  pairs.sort((a, b) => b.score - a.score);
  let pos = 0;
  let neg = 0;
  for (const p of pairs) {
    if (p.label === 1) pos++;
    else neg++;
  }
  if (pos === 0 || neg === 0) return 0.5;
  let tp = 0;
  let fp = 0;
  let area = 0;
  let prevScore = Number.POSITIVE_INFINITY;
  let tpAtPrev = 0;
  let fpAtPrev = 0;
  for (const p of pairs) {
    if (p.score !== prevScore) {
      area += (fp - fpAtPrev) * (tp + tpAtPrev) * 0.5;
      prevScore = p.score;
      tpAtPrev = tp;
      fpAtPrev = fp;
    }
    if (p.label === 1) tp++;
    else fp++;
  }
  area += (fp - fpAtPrev) * (tp + tpAtPrev) * 0.5;
  return area / (pos * neg);
}
