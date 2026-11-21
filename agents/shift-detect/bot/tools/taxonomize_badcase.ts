import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { taxonomizeBadcase } from "../lib/stats/taxonomy.js";

const label = z.enum([
  "none",
  "covariate",
  "concept",
  "label",
  "mixed",
  "unknown",
]);

export default defineTool({
  description:
    "Cluster a miss into the error taxonomy (model/prompt/context/memory/tool/architecture/data_quality/threshold/sample_size) for Badcase Discovery.",
  effect: "read",
  inputSchema: z.object({
    id: z.string().optional(),
    symptom: z.string(),
    observedLabel: label,
    expectedLabel: label,
    notes: z.array(z.string()).optional(),
  }),
  async execute(input) {
    return taxonomizeBadcase(input);
  },
});
