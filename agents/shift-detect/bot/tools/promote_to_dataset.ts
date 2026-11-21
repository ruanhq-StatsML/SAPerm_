import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

/**
 * Closed-loop promotion signal. In eval sessions we only return the draft
 * record; live sessions persist a pointer in host.kv for later dataset merge.
 */
export default defineTool({
  description:
    "Promote a badcase / attribution miss into a high-value dataset candidate (Evaluation → Data closed loop).",
  effect: "idempotent",
  dryRunResult: (input) => ({
    promoted: false,
    dryRun: true,
    record: input,
  }),
  inputSchema: z.object({
    source: z.enum(["trace", "user_feedback", "online_metric", "fixture_miss", "human_eval"]),
    title: z.string(),
    shiftLabel: z.enum(["none", "covariate", "concept", "label", "mixed", "unknown"]),
    primaryFactor: z.string(),
    payloadSummary: z.string(),
    tags: z.array(z.string()).default([]),
  }),
  async execute(input, ctx) {
    const key = `dataset:promo:${input.source}:${input.title}`;
    const record = {
      ...input,
      promotedAt: new Date().toISOString(),
      sessionId: ctx.session.id,
    };
    if (ctx.session.purpose === "eval") {
      return { promoted: false, evalMode: true, record };
    }
    await ctx.host.kv.put(key, record);
    return { promoted: true, key, record };
  },
});
