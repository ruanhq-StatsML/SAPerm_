import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { compareMethods } from "../lib/stats/suite.js";

export default defineTool({
  description:
    "Compare detector methods on one fixture for new-model / new-architecture validation.",
  effect: "read",
  inputSchema: z.object({
    fixture: z
      .enum([
        "null_shift",
        "covariate_shift",
        "concept_drift",
        "mixed_shift",
        "label_noise",
      ])
      .default("covariate_shift"),
    methods: z
      .array(z.enum(["mmd", "c2st", "mean_diff", "po_risk", "delta_cf"]))
      .default(["mmd", "c2st", "mean_diff", "po_risk", "delta_cf"]),
  }),
  async execute(input) {
    const rows = compareMethods(input.methods, input.fixture);
    return {
      fixture: input.fixture,
      ranking: rows
        .slice()
        .sort((a, b) => a.result.pValue - b.result.pValue)
        .map((r) => ({
          method: r.method,
          pValue: r.result.pValue,
          statistic: r.result.statistic,
          rejectAt005: r.result.rejectAt005,
        })),
      rows,
    };
  },
});
