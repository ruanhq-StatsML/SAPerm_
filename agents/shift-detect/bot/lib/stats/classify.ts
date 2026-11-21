import { mmdTest } from "./mmd.js";
import { conceptScores, labelShiftScore } from "./concept.js";
import type { ClassifyResult, SampleMatrix, ShiftLabel } from "./types.js";

export function classifyShift(ref: SampleMatrix, neu: SampleMatrix): ClassifyResult {
  const cov = mmdTest(ref.X, neu.X, 30, 7);
  const rationale: string[] = [];
  let conceptScore = 0;
  let labelScore = 0;

  if (ref.y && neu.y) {
    const scores = conceptScores(ref.X, ref.y, neu.X, neu.y);
    conceptScore = scores.deltaCf;
    labelScore = labelShiftScore(ref.y, neu.y);
    rationale.push(
      `covariate MMD p=${cov.pValue.toFixed(3)}, delta_cf=${conceptScore.toFixed(4)}, label_gap=${labelScore.toFixed(3)}`,
    );
  } else {
    rationale.push(
      `covariate MMD p=${cov.pValue.toFixed(3)}; no outcomes provided so concept/label scores are unavailable.`,
    );
  }

  const covReject = cov.pValue < 0.05;
  const conceptReject = conceptScore > 0.02;
  const labelReject = labelScore > 0.15;

  let label: ShiftLabel = "none";
  let confidence = 0.55;

  if (!covReject && !conceptReject && !labelReject) {
    label = "none";
    confidence = 1 - Math.min(0.4, cov.pValue);
    rationale.push("No strong covariate, concept, or label evidence.");
  } else if (covReject && conceptReject) {
    label = "mixed";
    confidence = 0.75;
    rationale.push("Both covariate and concept signals fire → mixed shift.");
  } else if (covReject && !conceptReject) {
    label = "covariate";
    confidence = 0.8;
    rationale.push("Covariate test rejects; concept score stays small.");
  } else if (!covReject && conceptReject && !labelReject) {
    label = "concept";
    confidence = 0.78;
    rationale.push("Concept score elevated without covariate rejection.");
  } else if (labelReject && !covReject) {
    label = "label";
    confidence = 0.7;
    rationale.push("Marginal label rate moved while covariates look stable.");
  } else {
    label = "unknown";
    confidence = 0.45;
    rationale.push("Signals conflict; mark unknown and inspect attribution.");
  }

  return {
    label,
    confidence,
    evidence: {
      covariateP: cov.pValue,
      conceptScore,
      labelShiftScore: labelScore,
    },
    rationale,
  };
}
