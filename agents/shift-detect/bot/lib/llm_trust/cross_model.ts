import { evaluateOutput } from "./evaluators.js";
import type {
  AlignmentReport,
  CapabilityAxis,
  CrossModelCompare,
  ModelOutput,
} from "./types.js";
import { trustGate } from "./gate.js";

export function compareModelsOnCase(
  caseId: string,
  outputs: ModelOutput[],
): CrossModelCompare {
  const rows = outputs.filter((o) => o.caseId === caseId);
  if (rows.length < 2) {
    throw new Error(`Need ≥2 model outputs for case ${caseId}`);
  }
  const evals = rows.map((o) => evaluateOutput(o));
  const models = evals.map((e) => e.modelId);
  const overall = Object.fromEntries(evals.map((e) => [e.modelId, e.overall]));
  const ranked = evals.slice().sort((a, b) => b.overall - a.overall);
  const winner = ranked[0].modelId;
  const margin = ranked[0].overall - (ranked[1]?.overall ?? 0);

  const axes = new Set<CapabilityAxis>();
  for (const e of evals) for (const a of e.axisScores) axes.add(a.axis);

  const axisDeltas = Array.from(axes).map((axis) => {
    const scores: Record<string, number> = {};
    for (const e of evals) {
      const hit = e.axisScores.find((s) => s.axis === axis);
      scores[e.modelId] = hit ? hit.score / Math.max(hit.maxScore, 1) : 0;
    }
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0] ?? winner;
    return { axis, best, scores };
  });

  // Pairwise agreement: share of axes where argmax is unique vs ties.
  let agree = 0;
  for (const a of axisDeltas) {
    const vals = Object.values(a.scores);
    const max = Math.max(...vals);
    if (vals.filter((v) => Math.abs(v - max) < 1e-9).length === 1) agree++;
  }
  const agreement = agree / Math.max(axisDeltas.length, 1);

  return {
    caseId,
    models,
    winner: margin > 1e-6 ? winner : null,
    margin,
    axisDeltas,
    agreement,
    notes: [
      `overall=${JSON.stringify(overall)}`,
      margin > 1e-6
        ? `Winner ${winner} by margin ${margin.toFixed(3)}.`
        : "Models tied within margin.",
    ],
  };
}

export function alignCheckpoints(
  baseModelId: string,
  candidateModelId: string,
  outputs: ModelOutput[],
): AlignmentReport {
  const base = outputs.filter((o) => o.modelId === baseModelId);
  const cand = outputs.filter((o) => o.modelId === candidateModelId);
  const caseIds = Array.from(
    new Set(base.map((o) => o.caseId).filter((id) => cand.some((c) => c.caseId === id))),
  );
  if (caseIds.length === 0) {
    throw new Error("No overlapping cases between base and candidate");
  }

  let improve = 0;
  let regress = 0;
  let unchanged = 0;
  let sumDelta = 0;
  const axisAcc = new Map<CapabilityAxis, { delta: number; regress: number; n: number }>();

  for (const caseId of caseIds) {
    const b = evaluateOutput(base.find((o) => o.caseId === caseId)!);
    const c = evaluateOutput(cand.find((o) => o.caseId === caseId)!);
    const delta = c.overall - b.overall;
    sumDelta += delta;
    if (delta > 0.02) improve++;
    else if (delta < -0.02) regress++;
    else unchanged++;

    const axes = new Set([
      ...b.axisScores.map((s) => s.axis),
      ...c.axisScores.map((s) => s.axis),
    ]);
    for (const axis of axes) {
      const bs = b.axisScores.find((s) => s.axis === axis);
      const cs = c.axisScores.find((s) => s.axis === axis);
      const bn = bs ? bs.score / Math.max(bs.maxScore, 1) : 0;
      const cn = cs ? cs.score / Math.max(cs.maxScore, 1) : 0;
      const d = cn - bn;
      const cur = axisAcc.get(axis) ?? { delta: 0, regress: 0, n: 0 };
      cur.delta += d;
      cur.n += 1;
      if (d < -0.15) cur.regress += 1;
      axisAcc.set(axis, cur);
    }
  }

  const n = caseIds.length;
  const axisSummary = Array.from(axisAcc.entries()).map(([axis, v]) => ({
    axis,
    meanDelta: v.delta / Math.max(v.n, 1),
    regressCount: v.regress,
  }));
  const blockingAxes = axisSummary
    .filter((a) => a.regressCount > 0 && a.meanDelta < -0.05)
    .map((a) => a.axis);

  const meanDelta = sumDelta / n;
  const improvedShare = improve / n;
  const regressedShare = regress / n;
  const unchangedShare = unchanged / n;

  const gate = trustGate({
    meanDelta,
    regressedShare,
    safetyRegress: axisSummary.find((a) => a.axis === "safety")?.regressCount ?? 0,
    groundedRegress:
      axisSummary.find((a) => a.axis === "groundedness")?.regressCount ?? 0,
  });

  return {
    baseModelId,
    candidateModelId,
    nCases: n,
    meanDelta,
    improvedShare,
    regressedShare,
    unchangedShare,
    axisSummary,
    gatePass: gate.pass && blockingAxes.length === 0,
    blockingAxes,
  };
}
