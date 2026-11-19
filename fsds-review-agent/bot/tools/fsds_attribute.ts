import { prompt } from "@grok/agent-sdk";
import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

import { attributeBlocks } from "../lib/fsds";

export default defineTool({
  description: prompt`
    Attribute contribution shares across arms / blocks / people from raw
    non-negative scores (distances, PO-risk drops, effort units).
    Returns FSDS-style share vector, ranking, entropy, and evenness.
    Call this before fsds_incentives.
  `,
  effect: "read",
  inputSchema: z.object({
    names: z.array(z.string().min(1)).min(1).describe("Arm / block / contributor names."),
    scores: z
      .array(z.number())
      .min(1)
      .describe("Aligned raw scores; negatives are clipped to 0 before sharing."),
  }),
  async execute({ names, scores }) {
    return attributeBlocks({ names, scores });
  },
});
