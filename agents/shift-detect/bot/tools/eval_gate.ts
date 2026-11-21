import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { evalGate } from "../lib/stats/gate.js";

export default defineTool({
  description:
    "Apply Eval Gate thresholds to suite metrics before promoting a method/prompt/architecture change.",
  effect: "read",
  inputSchema: z.object({
    typeI: z.number().min(0).max(1),
    powerCovariate: z.number().min(0).max(1),
    powerConcept: z.number().min(0).max(1),
    attributionTopHit: z.number().min(0).max(1),
    typeAccuracy: z.number().min(0).max(1),
    thresholds: z
      .object({
        typeI: z.number().min(0).max(1).optional(),
        powerCovariate: z.number().min(0).max(1).optional(),
        powerConcept: z.number().min(0).max(1).optional(),
        attributionTopHit: z.number().min(0).max(1).optional(),
        typeAccuracy: z.number().min(0).max(1).optional(),
      })
      .optional(),
  }),
  async execute(input) {
    const { thresholds, ...metrics } = input;
    return evalGate(metrics, thresholds);
  },
});
