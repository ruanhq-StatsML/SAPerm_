import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { classifyShift } from "../lib/stats/classify.js";
import { loadFixture } from "../lib/stats/fixtures.js";
import type { SampleMatrix } from "../lib/stats/types.js";

export default defineTool({
  description:
    "Classify shift type: none | covariate | concept | label | mixed | unknown, using MMD + delta_cf + label-gap evidence.",
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
      .optional(),
    XRef: z.array(z.array(z.number())).optional(),
    XNew: z.array(z.array(z.number())).optional(),
    yRef: z.array(z.number()).optional(),
    yNew: z.array(z.number()).optional(),
  }),
  async execute(input) {
    let ref: SampleMatrix | undefined = input.XRef
      ? { X: input.XRef, y: input.yRef }
      : undefined;
    let neu: SampleMatrix | undefined = input.XNew
      ? { X: input.XNew, y: input.yNew }
      : undefined;
    if (input.fixture) {
      const f = loadFixture(input.fixture);
      ref = f.ref;
      neu = f.neu;
    }
    if (!ref || !neu) throw new Error("Provide fixture or matrices");
    return { fixture: input.fixture ?? null, ...classifyShift(ref, neu) };
  },
});
