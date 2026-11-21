import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { FIXTURE_OUTPUTS, getCase } from "../lib/llm_trust/datasets.js";
import { evaluateOutput } from "../lib/llm_trust/evaluators.js";

export default defineTool({
  description:
    "Score a text/multimodal model output against versioned rubrics (factuality, groundedness, safety, multimodal alignment, ...).",
  effect: "read",
  inputSchema: z.object({
    modelId: z.string().default("base-text-v1"),
    caseId: z.string().default("txt_fact_capitals_v1"),
    text: z
      .string()
      .optional()
      .describe("If omitted, uses the canned fixture output for modelId+caseId"),
  }),
  async execute(input) {
    const c = getCase(input.caseId);
    const text =
      input.text ??
      FIXTURE_OUTPUTS.find(
        (o) => o.modelId === input.modelId && o.caseId === input.caseId,
      )?.text;
    if (!text) {
      throw new Error(
        `No text provided and no fixture output for ${input.modelId}/${input.caseId}`,
      );
    }
    const result = evaluateOutput({
      modelId: input.modelId,
      caseId: input.caseId,
      text,
    });
    return {
      case: { id: c.id, modality: c.modality, axes: c.axes, version: c.version },
      result,
    };
  },
});
