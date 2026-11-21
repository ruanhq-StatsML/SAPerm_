import { alignCheckpoints, compareModelsOnCase } from "./cross_model.js";
import {
  FIXTURE_OUTPUTS,
  listCases,
  listFixtureModels,
  outputsForModel,
} from "./datasets.js";
import { evaluateModel } from "./evaluators.js";
import { trustGate } from "./gate.js";
import { explainBehavior } from "./interpret.js";
import { listRubrics } from "./rubrics.js";

export function runTrustSuite(opts?: {
  baseModelId?: string;
  candidateModelId?: string;
  weakModelId?: string;
}) {
  const baseModelId = opts?.baseModelId ?? "base-text-v1";
  const candidateModelId = opts?.candidateModelId ?? "candidate-text-v2";
  const weakModelId = opts?.weakModelId ?? "weak-vl-v0";

  const baseEvals = evaluateModel(baseModelId, outputsForModel(baseModelId));
  const candEvals = evaluateModel(
    candidateModelId,
    outputsForModel(candidateModelId),
  );
  const weakEvals = evaluateModel(weakModelId, outputsForModel(weakModelId));

  const mean = (xs: number[]) =>
    xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);

  const cross = listCases()
    .slice(0, 4)
    .map((c) =>
      compareModelsOnCase(
        c.id,
        FIXTURE_OUTPUTS.filter((o) => o.caseId === c.id),
      ),
    );

  const alignment = alignCheckpoints(
    baseModelId,
    candidateModelId,
    FIXTURE_OUTPUTS,
  );

  const attributions = weakEvals
    .filter((e) => e.overall < 0.55)
    .slice(0, 4)
    .map((e) => {
      const out = FIXTURE_OUTPUTS.find(
        (o) => o.modelId === e.modelId && o.caseId === e.caseId,
      )!;
      return explainBehavior(out);
    });

  const candMean = mean(candEvals.map((e) => e.overall));
  const gate = trustGate({
    meanDelta: alignment.meanDelta,
    regressedShare: alignment.regressedShare,
    safetyRegress:
      alignment.axisSummary.find((a) => a.axis === "safety")?.regressCount ?? 0,
    groundedRegress:
      alignment.axisSummary.find((a) => a.axis === "groundedness")?.regressCount ??
      0,
    meanOverall: candMean,
  });

  return {
    catalog: {
      models: listFixtureModels(),
      nCases: listCases().length,
      nRubrics: listRubrics().length,
      versions: Array.from(new Set(listCases().map((c) => c.version))),
    },
    modelMeans: {
      [baseModelId]: mean(baseEvals.map((e) => e.overall)),
      [candidateModelId]: candMean,
      [weakModelId]: mean(weakEvals.map((e) => e.overall)),
    },
    crossModelSample: cross.map((c) => ({
      caseId: c.caseId,
      winner: c.winner,
      margin: Number(c.margin.toFixed(3)),
      agreement: Number(c.agreement.toFixed(3)),
    })),
    alignment,
    behaviorAttributions: attributions,
    gate,
  };
}
