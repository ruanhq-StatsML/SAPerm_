import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { listCatalog } from "../lib/stats/catalog.js";
import { listFixtures } from "../lib/stats/fixtures.js";

export default defineTool({
  description:
    "List the method catalog (detect/attribute/monitor) mapped to ruanhq-StatsML repos, plus available synthetic fixtures.",
  effect: "read",
  inputSchema: z.object({
    family: z.enum(["detect", "attribute", "monitor", "select"]).optional(),
  }),
  async execute(input) {
    return {
      methods: listCatalog(input.family),
      fixtures: listFixtures(),
    };
  },
});
