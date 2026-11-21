/** Trustworthy text / multimodal LLM evaluation types. */

export type Modality = "text" | "vision_language" | "speech_text" | "multimodal";

export type CapabilityAxis =
  | "factuality"
  | "instruction_following"
  | "reasoning"
  | "safety"
  | "groundedness"
  | "consistency"
  | "refusal_quality"
  | "multimodal_alignment";

export type RubricLevel = {
  score: number;
  label: string;
  criteria: string;
};

export type Rubric = {
  id: string;
  axis: CapabilityAxis;
  modality: Modality[];
  description: string;
  levels: RubricLevel[];
  weight: number;
};

export type EvalCase = {
  id: string;
  modality: Modality;
  axes: CapabilityAxis[];
  prompt: string;
  /** Optional reference / gold answer for grounded scoring. */
  reference?: string;
  /** Image/audio/document pointer for multimodal cases. */
  mediaRef?: string;
  tags: string[];
  version: string;
};

export type ModelOutput = {
  modelId: string;
  caseId: string;
  text: string;
  /** Optional per-token or step traces for interpretability. */
  rationale?: string;
  latencyMs?: number;
  meta?: Record<string, string | number | boolean>;
};

export type AxisScore = {
  axis: CapabilityAxis;
  score: number;
  maxScore: number;
  evidence: string[];
  confidence: number;
};

export type CaseEvalResult = {
  caseId: string;
  modelId: string;
  axisScores: AxisScore[];
  overall: number;
  grounded: boolean;
  notes: string[];
};

export type BehaviorFactor =
  | "model_capability"
  | "prompt_design"
  | "context_window"
  | "decoding"
  | "rag_retrieval"
  | "tool_use"
  | "safety_policy"
  | "modality_encoder"
  | "data_contamination"
  | "annotation_noise";

export type BehaviorAttribution = {
  caseId: string;
  modelId: string;
  symptom: string;
  primaryFactor: BehaviorFactor;
  secondaryFactors: BehaviorFactor[];
  evidence: string[];
  suggestedFix: string;
};

export type CrossModelCompare = {
  caseId: string;
  models: string[];
  winner: string | null;
  margin: number;
  axisDeltas: Array<{ axis: CapabilityAxis; best: string; scores: Record<string, number> }>;
  agreement: number;
  notes: string[];
};

export type AlignmentReport = {
  baseModelId: string;
  candidateModelId: string;
  nCases: number;
  meanDelta: number;
  improvedShare: number;
  regressedShare: number;
  unchangedShare: number;
  axisSummary: Array<{ axis: CapabilityAxis; meanDelta: number; regressCount: number }>;
  gatePass: boolean;
  blockingAxes: CapabilityAxis[];
};

export type TrustGateVerdict = {
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
