/** Shared types for distribution-shift detection. */

export type ShiftLabel =
  | "none"
  | "covariate"
  | "concept"
  | "label"
  | "mixed"
  | "unknown";

export type MethodId =
  | "mmd"
  | "c2st"
  | "mean_diff"
  | "po_risk"
  | "delta_cf"
  | "block_attribution";

export type SampleMatrix = {
  /** Row-major feature matrix. */
  X: number[][];
  /** Optional outcomes aligned with rows. */
  y?: number[];
  /** Optional column names. */
  featureNames?: string[];
};

export type DetectResult = {
  method: MethodId;
  statistic: number;
  pValue: number;
  rejectAt005: boolean;
  rejectAt001: boolean;
  nRef: number;
  nNew: number;
  nFeatures: number;
  notes: string[];
};

export type AttributionHit = {
  name: string;
  score: number;
  share: number;
  rank: number;
};

export type AttributeResult = {
  method:
    | "block_mean_diff"
    | "feature_mean_diff"
    | "loco_po_risk"
    | "pathway_counterfactual";
  ranked: AttributionHit[];
  top: string | null;
  topHit: boolean | null;
  groundTruth: string[];
};

export type ClassifyResult = {
  label: ShiftLabel;
  confidence: number;
  evidence: {
    covariateP: number;
    conceptScore: number;
    labelShiftScore: number;
  };
  rationale: string[];
};

export type FixtureId =
  | "null_shift"
  | "covariate_shift"
  | "concept_drift"
  | "mixed_shift"
  | "label_noise";

export type FixtureBundle = {
  id: FixtureId;
  description: string;
  groundTruth: ShiftLabel;
  driftedBlocks: string[];
  ref: SampleMatrix;
  neu: SampleMatrix;
  blocks: Record<string, number[]>;
};

export type ErrorFactor =
  | "model"
  | "prompt"
  | "context"
  | "memory"
  | "tool"
  | "agent_architecture"
  | "data_quality"
  | "threshold"
  | "sample_size";

export type BadcaseRecord = {
  id: string;
  symptom: string;
  observedLabel: ShiftLabel;
  expectedLabel: ShiftLabel;
  primaryFactor: ErrorFactor;
  secondaryFactors: ErrorFactor[];
  severity: "low" | "medium" | "high";
  promotionWorthy: boolean;
  notes: string[];
};

export type GateVerdict = {
  pass: boolean;
  checks: Array<{
    name: string;
    pass: boolean;
    observed: number;
    threshold: number;
    direction: "gte" | "lte";
  }>;
  summary: string;
};
