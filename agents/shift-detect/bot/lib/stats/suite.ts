import { blockMeanAttribution, locoPoAttribution } from "./attribution.js";
import { classifyShift } from "./classify.js";
import { deltaCfTest, poRiskTest } from "./concept.js";
import { c2stTest } from "./c2st.js";
import { loadFixture, listFixtures } from "./fixtures.js";
import { evalGate } from "./gate.js";
import { meanDiffTest } from "./mean_test.js";
import { mmdTest } from "./mmd.js";
import { pathwayBlockAttribution } from "./pathway.js";
import type {
  DetectResult,
  FixtureId,
  GateVerdict,
  MethodId,
  ShiftLabel,
} from "./types.js";

export type SuiteCaseResult = {
  fixture: FixtureId;
  groundTruth: ShiftLabel;
  predicted: ShiftLabel;
  detect: DetectResult[];
  attributionTop: string | null;
  attributionTopHit: boolean | null;
};

function runDetectors(id: FixtureId): DetectResult[] {
  const f = loadFixture(id);
  const out: DetectResult[] = [
    mmdTest(f.ref.X, f.neu.X, 24, 1),
    c2stTest(f.ref.X, f.neu.X, 3, 2),
    meanDiffTest(f.ref.X, f.neu.X),
  ];
  if (f.ref.y && f.neu.y) {
    out.push(poRiskTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y));
    out.push(deltaCfTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y));
  }
  return out;
}

/** Primary detectors used for Type-I / covariate power (C2ST is advisory only). */
function primaryCovariateReject(detects: DetectResult[], alpha = 0.05): boolean {
  return detects.some(
    (d) => (d.method === "mmd" || d.method === "mean_diff") && d.pValue < alpha,
  );
}

function conceptReject(detects: DetectResult[], alpha = 0.05): boolean {
  return detects.some(
    (d) =>
      (d.method === "po_risk" || d.method === "delta_cf") && d.pValue < alpha,
  );
}

export function runFixtureSuite(fixtureIds?: FixtureId[]): {
  cases: SuiteCaseResult[];
  metrics: {
    typeI: number;
    powerCovariate: number;
    powerConcept: number;
    attributionTopHit: number;
    typeAccuracy: number;
  };
  gate: GateVerdict;
} {
  const ids =
    fixtureIds ??
    (listFixtures().map((f) => f.id) as FixtureId[]);

  const cases: SuiteCaseResult[] = [];
  for (const id of ids) {
    const f = loadFixture(id);
    const detect = runDetectors(id);
    const predicted = classifyShift(f.ref, f.neu).label;

    let attributionTop: string | null = null;
    let attributionTopHit: boolean | null = null;
    if (f.driftedBlocks.length > 0) {
      const attr = pathwayBlockAttribution(
        f.ref.X,
        f.neu.X,
        f.blocks,
        f.driftedBlocks,
      );
      attributionTop = attr.top;
      attributionTopHit = attr.topHit;
    } else if (f.groundTruth === "concept" && f.ref.y && f.neu.y) {
      const attr = locoPoAttribution(
        f.ref.X,
        f.ref.y,
        f.neu.X,
        f.neu.y,
        f.ref.featureNames,
        ["f0"],
      );
      attributionTop = attr.top;
      attributionTopHit = attr.topHit;
    } else {
      const attr = blockMeanAttribution(
        f.ref.X,
        f.neu.X,
        f.blocks,
        f.driftedBlocks,
        f.ref.featureNames,
      );
      attributionTop = attr.top;
      attributionTopHit = attr.topHit;
    }

    cases.push({
      fixture: id,
      groundTruth: f.groundTruth,
      predicted,
      detect,
      attributionTop,
      attributionTopHit,
    });
  }

  const nullCase = cases.find((c) => c.groundTruth === "none");
  const covCase = cases.find((c) => c.groundTruth === "covariate");
  const conceptCase = cases.find((c) => c.groundTruth === "concept");

  const typeI = nullCase ? (primaryCovariateReject(nullCase.detect) ? 1 : 0) : 0;
  const powerCovariate = covCase
    ? primaryCovariateReject(covCase.detect)
      ? 1
      : 0
    : 0;
  const powerConcept = conceptCase
    ? conceptReject(conceptCase.detect)
      ? 1
      : 0
    : 0;

  const attrCases = cases.filter((c) => c.attributionTopHit !== null);
  const attributionTopHit =
    attrCases.length === 0
      ? 0
      : attrCases.filter((c) => c.attributionTopHit).length / attrCases.length;

  const typeAccuracy =
    cases.filter((c) => c.predicted === c.groundTruth).length /
    Math.max(cases.length, 1);

  const metrics = {
    typeI,
    powerCovariate,
    powerConcept,
    attributionTopHit,
    typeAccuracy,
  };

  return { cases, metrics, gate: evalGate(metrics) };
}

export function compareMethods(
  methods: MethodId[],
  fixture: FixtureId = "covariate_shift",
): Array<{ method: MethodId; result: DetectResult }> {
  const f = loadFixture(fixture);
  const runners: Record<MethodId, () => DetectResult> = {
    mmd: () => mmdTest(f.ref.X, f.neu.X),
    c2st: () => c2stTest(f.ref.X, f.neu.X),
    mean_diff: () => meanDiffTest(f.ref.X, f.neu.X),
    po_risk: () => {
      if (!f.ref.y || !f.neu.y) throw new Error("fixture lacks outcomes");
      return poRiskTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y);
    },
    delta_cf: () => {
      if (!f.ref.y || !f.neu.y) throw new Error("fixture lacks outcomes");
      return deltaCfTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y);
    },
    block_attribution: () => {
      throw new Error("block_attribution is not a detector; use attribute tools");
    },
  };
  return methods
    .filter((m) => m !== "block_attribution")
    .map((method) => ({ method, result: runners[method]() }));
}
