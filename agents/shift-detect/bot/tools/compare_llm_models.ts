import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { compareModelsOnCase } from "../lib/llm_trust/cross_model.js";
import { FIXTURE_OUTPUTS } from "../lib/llm_trust/datasets.js";

export default defineTool({
  description:
    "Cross-model对照: compare two or more models on one eval case; report winner, margin, per-axis deltas, agreement.",
  effect: "read",
  inputSchema: z.object({
    caseId: z.string().default("txt_reason_parity_v1"),
    modelIds: z
      .array(z.string())
      .optional()
      .describe("Defaults to all fixture models that have this case"),
  }),
  async execute(input) {
    const outputs = FIXTURE_OUTPUTS.filter((o) => {
      if (o.caseId !== input.caseId) return false;
      if (input.modelIds && input.modelIds.length > 0) {
        return input.modelIds.includes(o.modelId);
      }
      return true;
    });
    return compareModelsOnCase(input.caseId, outputs);
  },
});
