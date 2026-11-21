import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { c2stTest } from "../lib/stats/c2st.js";
import { deltaCfTest, poRiskTest } from "../lib/stats/concept.js";
import { loadFixture } from "../lib/stats/fixtures.js";
import { meanDiffTest } from "../lib/stats/mean_test.js";
import { mmdTest } from "../lib/stats/mmd.js";

const fixtureEnum = z.enum([
  "null_shift",
  "covariate_shift",
  "concept_drift",
  "mixed_shift",
  "label_noise",
]);

const methodEnum = z.enum(["mmd", "c2st", "mean_diff", "po_risk", "delta_cf", "all"]);

export default defineTool({
  description:
    "Run distribution-shift detectors (MMD, C2ST, mean_diff, PO-risk, delta_cf) on a fixture or raw matrices. Use before attribution.",
  effect: "read",
  inputSchema: z.object({
    fixture: fixtureEnum.optional(),
    method: methodEnum.default("all"),
    XRef: z.array(z.array(z.number())).optional(),
    XNew: z.array(z.array(z.number())).optional(),
    yRef: z.array(z.number()).optional(),
    yNew: z.array(z.number()).optional(),
    nPerm: z.number().int().min(5).max(200).default(24),
  }),
  async execute(input) {
    let XRef = input.XRef;
    let XNew = input.XNew;
    let yRef = input.yRef;
    let yNew = input.yNew;
    if (input.fixture) {
      const f = loadFixture(input.fixture);
      XRef = f.ref.X;
      XNew = f.neu.X;
      yRef = f.ref.y;
      yNew = f.neu.y;
    }
    if (!XRef || !XNew) throw new Error("Provide fixture or XRef/XNew");

    const want = input.method;
    const results = [];
    if (want === "all" || want === "mmd") results.push(mmdTest(XRef, XNew, input.nPerm));
    if (want === "all" || want === "c2st") results.push(c2stTest(XRef, XNew));
    if (want === "all" || want === "mean_diff") results.push(meanDiffTest(XRef, XNew));
    if ((want === "all" || want === "po_risk") && yRef && yNew) {
      results.push(poRiskTest(XRef, yRef, XNew, yNew));
    }
    if ((want === "all" || want === "delta_cf") && yRef && yNew) {
      results.push(deltaCfTest(XRef, yRef, XNew, yNew));
    }
    if (results.length === 0) {
      throw new Error("Selected method needs outcomes (yRef/yNew) which were not provided");
    }
    return {
      fixture: input.fixture ?? null,
      anyRejectAt005: results.some((r) => r.rejectAt005),
      results,
    };
  },
});
