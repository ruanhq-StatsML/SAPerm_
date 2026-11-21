import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { classifyShift } from "../lib/stats/classify.js";
import { loadFixture } from "../lib/stats/fixtures.js";
import { pathwayBlockAttribution } from "../lib/stats/pathway.js";
import { mmdTest } from "../lib/stats/mmd.js";
import { deltaCfTest } from "../lib/stats/concept.js";

export default defineTool({
  description:
    "One-shot attribution report: detect evidence + shift type + pathway block ranking + grounded recommendation.",
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
  }),
  async execute(input) {
    const f = loadFixture(input.fixture);
    const mmd = mmdTest(f.ref.X, f.neu.X, 24, 3);
    const typed = classifyShift(f.ref, f.neu);
    const attr = pathwayBlockAttribution(
      f.ref.X,
      f.neu.X,
      f.blocks,
      f.driftedBlocks,
    );
    const concept =
      f.ref.y && f.neu.y
        ? deltaCfTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y)
        : null;

    const topBlocks = attr.ranked.slice(0, 3).map((h) => ({
      block: h.name,
      share: Number(h.share.toFixed(3)),
      score: Number(h.score.toFixed(4)),
    }));

    return {
      fixture: input.fixture,
      groundTruth: f.groundTruth,
      detection: {
        mmdP: mmd.pValue,
        deltaCfP: concept?.pValue ?? null,
        anyReject: mmd.rejectAt005 || Boolean(concept?.rejectAt005),
      },
      shiftType: typed,
      attribution: {
        method: attr.method,
        top: attr.top,
        topHit: attr.topHit,
        topBlocks,
      },
      recommendation: {
        nextAction:
          typed.label === "none"
            ? "No shift — keep monitoring; do not retrain."
            : typed.label === "covariate"
              ? `Retrain or reweight on drifted blocks: ${attr.top ?? "unknown"}; check FSDS feature selection.`
              : typed.label === "concept"
                ? "Concept drift — refresh outcome model; run CFPerm/LOCO for driver features."
                : typed.label === "mixed"
                  ? "Mixed shift — attribute blocks first, then re-estimate concept on residual."
                  : "Inspect taxonomy and promote a badcase if confidence is low.",
        researchHooks: [
          "tabular-drift-attribution",
          "CFPerm",
          "OnlineMonitoring_Out_of_Distribution",
          "FSDS-FeatureSelection_for_DistributionShift",
        ],
      },
    };
  },
});
