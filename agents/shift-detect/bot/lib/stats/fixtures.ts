import { mulberry32 } from "./matrices.js";
import type { FixtureBundle, FixtureId, SampleMatrix, ShiftLabel } from "./types.js";

function makeBlocks(p: number, blockSize = 4): Record<string, number[]> {
  const blocks: Record<string, number[]> = {};
  let b = 0;
  for (let j = 0; j < p; j += blockSize) {
    blocks[`block_${b}`] = Array.from(
      { length: Math.min(blockSize, p - j) },
      (_, k) => j + k,
    );
    b++;
  }
  return blocks;
}

function drawGaussian(rng: () => number, mean: number, sd: number): number {
  const u = Math.max(rng(), 1e-12);
  const v = Math.max(rng(), 1e-12);
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function baseFeatures(n: number, p: number, rng: () => number): number[][] {
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < p; j++) row.push(drawGaussian(rng, 0, 1));
    X.push(row);
  }
  return X;
}

function outcomeFromX(X: number[][], rng: () => number, coefShift = 0): number[] {
  return X.map((row) => {
    const z = 0.8 * row[0] + 0.5 * row[1] - 0.3 * row[2] + coefShift * row[0];
    const p = 1 / (1 + Math.exp(-z));
    return rng() < p ? 1 : 0;
  });
}

function withNames(X: number[][], y: number[]): SampleMatrix {
  return {
    X,
    y,
    featureNames: X[0].map((_, j) => `f${j}`),
  };
}

function pack(
  id: FixtureId,
  description: string,
  groundTruth: ShiftLabel,
  driftedBlocks: string[],
  XRef: number[][],
  yRef: number[],
  XNew: number[][],
  yNew: number[],
): FixtureBundle {
  return {
    id,
    description,
    groundTruth,
    driftedBlocks,
    ref: withNames(XRef, yRef),
    neu: withNames(XNew, yNew),
    blocks: makeBlocks(XRef[0].length, 4),
  };
}

const FIXTURES: Record<FixtureId, () => FixtureBundle> = {
  null_shift: () => {
    const rng = mulberry32(11);
    const XRef = baseFeatures(180, 12, rng);
    const XNew = baseFeatures(180, 12, rng);
    return pack(
      "null_shift",
      "Same P(X,Y). Null control for Type-I behavior.",
      "none",
      [],
      XRef,
      outcomeFromX(XRef, rng, 0),
      XNew,
      outcomeFromX(XNew, rng, 0),
    );
  },
  covariate_shift: () => {
    const rng = mulberry32(22);
    const XRef = baseFeatures(180, 12, rng);
    const XNew = baseFeatures(180, 12, rng);
    for (const row of XNew) {
      row[4] += 1.6;
      row[5] += 1.4;
      row[6] += 1.5;
      row[7] += 1.3;
    }
    return pack(
      "covariate_shift",
      "Location shift on block_1; concept P(Y|X) held fixed.",
      "covariate",
      ["block_1"],
      XRef,
      outcomeFromX(XRef, rng, 0),
      XNew,
      outcomeFromX(XNew, rng, 0),
    );
  },
  concept_drift: () => {
    const rng = mulberry32(33);
    const XRef = baseFeatures(180, 12, rng);
    const XNew = baseFeatures(180, 12, rng);
    return pack(
      "concept_drift",
      "Same covariates; outcome coefficients flip on f0 (concept drift).",
      "concept",
      [],
      XRef,
      outcomeFromX(XRef, rng, 0),
      XNew,
      outcomeFromX(XNew, rng, -3.5),
    );
  },
  mixed_shift: () => {
    const rng = mulberry32(44);
    const XRef = baseFeatures(180, 12, rng);
    const XNew = baseFeatures(180, 12, rng);
    for (const row of XNew) {
      row[0] += 1.4;
      row[1] += 1.2;
    }
    return pack(
      "mixed_shift",
      "Covariate shift on block_0 plus concept coefficient change.",
      "mixed",
      ["block_0"],
      XRef,
      outcomeFromX(XRef, rng, 0),
      XNew,
      outcomeFromX(XNew, rng, -3.0),
    );
  },
  label_noise: () => {
    const rng = mulberry32(55);
    const XRef = baseFeatures(180, 12, rng);
    const XNew = baseFeatures(180, 12, rng);
    const yRef = outcomeFromX(XRef, rng, 0);
    // Strong marginal label shift: new window is ~90% positive, X unchanged.
    const yNew = XNew.map(() => (rng() < 0.9 ? 1 : 0));
    return pack(
      "label_noise",
      "Covariates unchanged; marginal label rate jumps (~0.5 → ~0.9).",
      "label",
      [],
      XRef,
      yRef,
      XNew,
      yNew,
    );
  },
};

export function listFixtures(): Array<{
  id: FixtureId;
  description: string;
  groundTruth: string;
}> {
  return (Object.keys(FIXTURES) as FixtureId[]).map((id) => {
    const f = FIXTURES[id]();
    return { id: f.id, description: f.description, groundTruth: f.groundTruth };
  });
}

export function loadFixture(id: FixtureId): FixtureBundle {
  const factory = FIXTURES[id];
  if (!factory) throw new Error(`Unknown fixture: ${id}`);
  return factory();
}

export function parseMatrixPayload(input: {
  X: number[][];
  y?: number[];
  featureNames?: string[];
}): SampleMatrix {
  return {
    X: input.X,
    y: input.y,
    featureNames: input.featureNames,
  };
}
