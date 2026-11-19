import { prompt } from "@grok/agent-sdk";
import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

import { fsdsIncentives } from "../lib/fsds";

const armSchema = z.object({
  name: z.string(),
  mu: z.number().default(0),
  n: z.number().nonnegative().default(0),
  m2: z.number().nonnegative().default(0),
});

export default defineTool({
  description: prompt`
    Turn FSDS attribution shares into Gaussian Thompson incentives that
    keep arms working evenly.
    Draw formula: N(mu, se) - lambda * share + gamma / sqrt(n+1).
    Use after fsds_attribute. Prefer gamma around 0.35 and lambda around 1.
  `,
  effect: "read",
  inputSchema: z.object({
    names: z.array(z.string().min(1)).min(1),
    shares: z.array(z.number()).min(1).describe("FSDS shares from fsds_attribute."),
    arms: z.array(armSchema).optional().describe("Optional running reward stats per arm."),
    lambda: z.number().nonnegative().optional().describe("FSDS penalty weight (default 1)."),
    gamma: z
      .number()
      .nonnegative()
      .optional()
      .describe("Uniformity bonus weight (default 0.35). Raise to force even pulls."),
    seed: z.number().int().default(2026),
    draws: z.number().int().min(1).max(500).optional().describe("Monte Carlo draws for rates."),
  }),
  async execute(input) {
    return fsdsIncentives(input);
  },
});
