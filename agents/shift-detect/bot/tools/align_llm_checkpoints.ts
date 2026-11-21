import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { alignCheckpoints } from "../lib/llm_trust/cross_model.js";
import { FIXTURE_OUTPUTS } from "../lib/llm_trust/datasets.js";

export default defineTool({
  description:
    "同模型对齐: compare a candidate checkpoint against a base on the shared case set; return deltas, regressions, Trust Eval Gate.",
  effect: "read",
  inputSchema: z.object({
    baseModelId: z.string().default("base-text-v1"),
    candidateModelId: z.string().default("candidate-text-v2"),
  }),
  async execute(input) {
    return alignCheckpoints(
      input.baseModelId,
      input.candidateModelId,
      FIXTURE_OUTPUTS,
    );
  },
});
