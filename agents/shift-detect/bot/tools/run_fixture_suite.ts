import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { runFixtureSuite } from "../lib/stats/suite.js";

export default defineTool({
  description:
    "Batch offline evaluation across synthetic fixtures: Type-I, power, attribution top-hit, type accuracy, and Eval Gate verdict.",
  effect: "read",
  inputSchema: z.object({
    fixtures: z
      .array(
        z.enum([
          "null_shift",
          "covariate_shift",
          "concept_drift",
          "mixed_shift",
          "label_noise",
        ]),
      )
      .optional(),
  }),
  async execute(input) {
    return runFixtureSuite(input.fixtures);
  },
});
