import type { BadcaseRecord, ErrorFactor, ShiftLabel } from "./types.js";

const FACTOR_HINTS: Record<ErrorFactor, string[]> = {
  model: ["model", "rfperm", "c2st", "mmd", "learner", "estimator"],
  prompt: ["prompt", "instruction", "wording", "rubric wording"],
  context: ["context", "window", "feature", "embedding", "batch"],
  memory: ["memory", "kv", "history", "session state"],
  tool: ["tool", "timeout", "schema", "api", "attribution tool"],
  agent_architecture: ["router", "planner", "subagent", "orchestration", "architecture"],
  data_quality: ["missing", "nan", "label noise", "corrupt", "schema drift"],
  threshold: ["alpha", "threshold", "pvalue", "cutoff", "gate"],
  sample_size: ["n=", "sample size", "underpowered", "small n", "few rows"],
};

function scoreFactor(text: string, factor: ErrorFactor): number {
  const lower = text.toLowerCase();
  return FACTOR_HINTS[factor].reduce(
    (acc, hint) => (lower.includes(hint) ? acc + 1 : acc),
    0,
  );
}

/** Map a detection/attribution miss onto the eval-framework error taxonomy. */
export function taxonomizeBadcase(input: {
  id?: string;
  symptom: string;
  observedLabel: ShiftLabel;
  expectedLabel: ShiftLabel;
  notes?: string[];
}): BadcaseRecord {
  const blob = [input.symptom, ...(input.notes ?? [])].join(" | ");
  const ranked = (Object.keys(FACTOR_HINTS) as ErrorFactor[])
    .map((factor) => ({ factor, score: scoreFactor(blob, factor) }))
    .sort((a, b) => b.score - a.score);

  let primary = ranked[0]?.score ? ranked[0].factor : ("model" as ErrorFactor);
  const secondary = ranked
    .slice(1)
    .filter((r) => r.score > 0)
    .slice(0, 3)
    .map((r) => r.factor);

  // Heuristics when keyword scores are flat.
  if (!ranked[0]?.score) {
    if (input.observedLabel === "none" && input.expectedLabel !== "none") {
      primary = "threshold";
      secondary.push("sample_size", "model");
    } else if (input.observedLabel !== "none" && input.expectedLabel === "none") {
      primary = "threshold";
      secondary.push("data_quality");
    } else if (
      (input.observedLabel === "covariate" && input.expectedLabel === "concept") ||
      (input.observedLabel === "concept" && input.expectedLabel === "covariate")
    ) {
      primary = "model";
      secondary.push("context");
    }
  }

  const severity: BadcaseRecord["severity"] =
    input.expectedLabel === "none" || input.observedLabel === "none"
      ? "high"
      : input.observedLabel === input.expectedLabel
        ? "low"
        : "medium";

  return {
    id: input.id ?? `bc_${Date.now()}`,
    symptom: input.symptom,
    observedLabel: input.observedLabel,
    expectedLabel: input.expectedLabel,
    primaryFactor: primary,
    secondaryFactors: Array.from(new Set(secondary)).filter((f) => f !== primary),
    severity,
    promotionWorthy: severity !== "low",
    notes: [
      ...(input.notes ?? []),
      "Taxonomy aligns with Evaluation → Badcase Discovery → factor attribution.",
    ],
  };
}
