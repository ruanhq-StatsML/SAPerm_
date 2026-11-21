import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { shiftGlance } from "../lib/glance.js";
import { locoPoAttribution } from "../lib/stats/attribution.js";
import { classifyShift } from "../lib/stats/classify.js";
import { deltaCfTest } from "../lib/stats/concept.js";
import { loadFixture } from "../lib/stats/fixtures.js";
import { mmdTest } from "../lib/stats/mmd.js";
import { pathwayBlockAttribution } from "../lib/stats/pathway.js";

function nextAction(label: string, top: string | null): string {
  switch (label) {
    case "none":
      return "No shift — keep monitoring; do not retrain.";
    case "covariate":
      return `Reweight / refresh drifted block: ${top ?? "unknown"} (FSDS).`;
    case "concept":
      return `Refresh outcome model; top concept driver: ${top ?? "unknown"} (CFPerm/LOCO).`;
    case "mixed":
      return `Fix covariates (${top ?? "block"}) first, then residual concept.`;
    case "label":
      return "Inspect label pipeline / class prior before model changes.";
    default:
      return "Low confidence — taxonomize badcase and promote to dataset.";
  }
}

export default defineTool({
  description:
    "One-shot intuitive attribution report with share board. Auto-picks pathway (covariate/mixed) or LOCO (concept).",
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
    mode: z
      .enum(["auto", "pathway", "loco"])
      .default("auto")
      .describe("auto: pathway for covariate/mixed, loco for concept"),
  }),
  async execute(input) {
    const f = loadFixture(input.fixture);
    const mmd = mmdTest(f.ref.X, f.neu.X, 24, 3);
    const typed = classifyShift(f.ref, f.neu);
    const concept =
      f.ref.y && f.neu.y
        ? deltaCfTest(f.ref.X, f.ref.y, f.neu.X, f.neu.y)
        : null;

    const mode =
      input.mode === "auto"
        ? typed.label === "concept"
          ? "loco"
          : "pathway"
        : input.mode;

    let top: string | null = null;
    let topHit: boolean | null = null;
    let method: string = mode;
    let rows: Array<{ name: string; share: number; score: number }> = [];

    if (mode === "loco") {
      if (!f.ref.y || !f.neu.y) {
        throw new Error("loco mode needs outcomes; pick pathway or another fixture");
      }
      const gt =
        f.groundTruth === "concept" ? ["f0"] : f.driftedBlocks;
      const attr = locoPoAttribution(
        f.ref.X,
        f.ref.y,
        f.neu.X,
        f.neu.y,
        f.ref.featureNames,
        gt,
      );
      top = attr.top;
      topHit = attr.topHit;
      method = attr.method;
      rows = attr.ranked.slice(0, 5).map((h) => ({
        name: h.name,
        share: h.share,
        score: h.score,
      }));
    } else {
      const attr = pathwayBlockAttribution(
        f.ref.X,
        f.neu.X,
        f.blocks,
        f.driftedBlocks,
      );
      top = attr.top;
      topHit = attr.topHit;
      method = attr.method;
      rows = attr.ranked.slice(0, 5).map((h) => ({
        name: h.name,
        share: h.share,
        score: h.score,
      }));
    }

    const action = nextAction(typed.label, top);
    const glance = shiftGlance({
      fixture: input.fixture,
      shiftType: typed.label,
      confidence: typed.confidence,
      top,
      topHit,
      rows,
      action,
    });

    return {
      fixture: input.fixture,
      groundTruth: f.groundTruth,
      mode,
      glance,
      detection: {
        mmdP: Number(mmd.pValue.toFixed(4)),
        deltaCfP: concept ? Number(concept.pValue.toFixed(4)) : null,
        anyReject: mmd.rejectAt005 || Boolean(concept?.rejectAt005),
      },
      shiftType: typed,
      attribution: {
        method,
        top,
        topHit,
        topBlocks: rows.map((r) => ({
          name: r.name,
          share: Number(r.share.toFixed(3)),
          score: Number(r.score.toFixed(4)),
        })),
      },
      recommendation: {
        nextAction: action,
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
