import { prompt } from "@grok/agent-sdk";
import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

import { attributeBlocks, fsdsIncentives, reviewBoard } from "../lib/fsds";

export default defineTool({
  description: prompt`
    One-shot audit: attribute scores, run FSDS Thompson incentives, and
    return a review verdict on whether effort is evenly distributed.
    Prefer this when the user pastes a scoreboard rather than calling
    fsds_attribute and fsds_incentives separately.
  `,
  effect: "read",
  inputSchema: z.object({
    names: z.array(z.string().min(1)).min(1),
    scores: z.array(z.number()).min(1),
    lambda: z.number().nonnegative().optional(),
    gamma: z.number().nonnegative().optional(),
    seed: z.number().int().default(2026),
    context: z.string().optional().describe("Optional free-text board context."),
  }),
  async execute({ names, scores, lambda, gamma, seed, context }) {
    const attribution = attributeBlocks({ names, scores });
    const incentives = fsdsIncentives({
      names,
      shares: attribution.shares,
      lambda,
      gamma,
      seed,
    });
    const review = reviewBoard({ attribution, incentives, context });
    return { attribution, incentives, review };
  },
});
