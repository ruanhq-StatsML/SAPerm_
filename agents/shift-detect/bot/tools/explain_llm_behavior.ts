import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { trustGlance } from "../lib/glance.js";
import { FIXTURE_OUTPUTS } from "../lib/llm_trust/datasets.js";
import { evaluateOutput } from "../lib/llm_trust/evaluators.js";
import { explainBehavior } from "../lib/llm_trust/interpret.js";

export default defineTool({
  description:
    "从评测角度解释模型行为: attribute a weak case to prompt/context/model/decoding/RAG/tool/safety/modality factors and suggest a fix.",
  effect: "read",
  inputSchema: z.object({
    modelId: z.string().default("weak-vl-v0"),
    caseId: z.string().default("vl_ocr_sign_v1"),
    text: z.string().optional(),
    symptom: z.string().optional(),
  }),
  async execute(input) {
    const text =
      input.text ??
      FIXTURE_OUTPUTS.find(
        (o) => o.modelId === input.modelId && o.caseId === input.caseId,
      )?.text;
    if (!text) throw new Error("Provide text or a known fixture model/case pair");
    const out = { modelId: input.modelId, caseId: input.caseId, text };
    const attr = explainBehavior(out, input.symptom);
    const scored = evaluateOutput(out);
    const weakAxes = scored.axisScores
      .filter((s) => s.score / Math.max(s.maxScore, 1) < 0.67)
      .map((s) => `${s.axis}=${s.score}/${s.maxScore}`);
    return {
      ...attr,
      glance: trustGlance({
        modelId: attr.modelId,
        caseId: attr.caseId,
        primaryFactor: attr.primaryFactor,
        overall: scored.overall,
        suggestedFix: attr.suggestedFix,
        weakAxes,
      }),
    };
  },
});

