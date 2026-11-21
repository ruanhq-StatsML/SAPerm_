import { getCase } from "./datasets.js";
import { evaluateOutput } from "./evaluators.js";
import type {
  BehaviorAttribution,
  BehaviorFactor,
  CaseEvalResult,
  ModelOutput,
} from "./types.js";

const HINTS: Record<BehaviorFactor, string[]> = {
  model_capability: ["wrong answer", "incapable", "failed reasoning", "trap answer"],
  prompt_design: ["ambiguous prompt", "constraint", "format", "instruction"],
  context_window: ["truncated", "missing context", "long context", "forgot"],
  decoding: ["temperature", "sampling", "decode", "repetition"],
  rag_retrieval: ["retrieval", "wrong doc", "rag", "context miss"],
  tool_use: ["tool", "api", "function call", "plugin"],
  safety_policy: ["refusal", "unsafe", "policy", "overblock"],
  modality_encoder: ["image", "ocr", "vision", "audio", "misaligned", "media"],
  data_contamination: ["memorize", "contamination", "train leak"],
  annotation_noise: ["label noise", "bad reference", "annotator"],
};

function factorFromAxes(result: CaseEvalResult): BehaviorFactor {
  const weak = result.axisScores
    .slice()
    .sort(
      (a, b) =>
        a.score / Math.max(a.maxScore, 1) - b.score / Math.max(b.maxScore, 1),
    )[0];
  if (!weak) return "model_capability";
  switch (weak.axis) {
    case "safety":
    case "refusal_quality":
      return "safety_policy";
    case "groundedness":
      return "rag_retrieval";
    case "instruction_following":
      return "prompt_design";
    case "multimodal_alignment":
      return "modality_encoder";
    case "reasoning":
    case "factuality":
      return "model_capability";
    case "consistency":
      return "decoding";
    default:
      return "model_capability";
  }
}

function suggestedFix(factor: BehaviorFactor): string {
  switch (factor) {
    case "model_capability":
      return "Add hard negatives to the dataset; consider SFT/RL on this axis or a stronger base model.";
    case "prompt_design":
      return "Tighten constraints in the prompt template; add format examples; re-eval.";
    case "context_window":
      return "Shorten or re-rank context; verify truncation boundaries.";
    case "decoding":
      return "Pin temperature/top-p; run consistency pairs; avoid greedy-only claims.";
    case "rag_retrieval":
      return "Inspect retrieved chunks; fix chunking/recall before blaming the LLM.";
    case "tool_use":
      return "Log tool args/results; add tool-schema evals; fail closed on tool errors.";
    case "safety_policy":
      return "Calibrate refusal policy with paired allow/deny cases; check leakage.";
    case "modality_encoder":
      return "Add modality-binding cases (OCR/count/attribute); stress encoder resolution.";
    case "data_contamination":
      return "Rotate items; prefer private held-out; report contamination checks.";
    case "annotation_noise":
      return "Double-annotate references; publish inter-annotator agreement.";
  }
}

/** Explain a weak case from the evaluation lens (not weight visualization). */
export function explainBehavior(out: ModelOutput, symptom?: string): BehaviorAttribution {
  const c = getCase(out.caseId);
  const result = evaluateOutput(out, c);
  const primary = factorFromAxes(result);
  const blob = [
    symptom ?? "",
    out.text,
    ...result.axisScores.flatMap((s) => s.evidence),
    c.modality,
    ...c.tags,
  ]
    .join(" ")
    .toLowerCase();

  const secondary = (Object.keys(HINTS) as BehaviorFactor[])
    .map((f) => ({
      f,
      n: HINTS[f].reduce((acc, h) => (blob.includes(h) ? acc + 1 : acc), 0),
    }))
    .filter((x) => x.f !== primary && x.n > 0)
    .sort((a, b) => b.n - a.n)
    .slice(0, 2)
    .map((x) => x.f);

  const weakAxes = result.axisScores
    .filter((s) => s.score / Math.max(s.maxScore, 1) < 0.67)
    .map((s) => `${s.axis}=${s.score}/${s.maxScore}`);

  return {
    caseId: out.caseId,
    modelId: out.modelId,
    symptom:
      symptom ??
      (weakAxes.length
        ? `Weak axes: ${weakAxes.join(", ")}`
        : `Overall ${result.overall.toFixed(3)} with no severe axis miss`),
    primaryFactor: primary,
    secondaryFactors: secondary,
    evidence: [
      ...result.axisScores.flatMap((s) => s.evidence),
      `modality=${c.modality}`,
      `overall=${result.overall.toFixed(3)}`,
    ],
    suggestedFix: suggestedFix(primary),
  };
}
