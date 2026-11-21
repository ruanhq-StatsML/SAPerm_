import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { runTrustSuite } from "../lib/llm_trust/suite.js";

export default defineTool({
  description:
    "Run the offline trustworthy LLM eval suite: model means, cross-model sample, checkpoint alignment, behavior attributions, Trust Eval Gate.",
  effect: "read",
  inputSchema: z.object({
    baseModelId: z.string().optional(),
    candidateModelId: z.string().optional(),
    weakModelId: z.string().optional(),
  }),
  async execute(input) {
    return runTrustSuite(input);
  },
});
