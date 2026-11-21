import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { shareBoard } from "../lib/glance.js";
import {
  blockMeanAttribution,
  locoPoAttribution,
} from "../lib/stats/attribution.js";
import { loadFixture } from "../lib/stats/fixtures.js";
import { pathwayBlockAttribution } from "../lib/stats/pathway.js";

const fixtureEnum = z.enum([
  "null_shift",
  "covariate_shift",
  "concept_drift",
  "mixed_shift",
  "label_noise",
]);

export default defineTool({
  description:
    "Attribute a detected distribution shift to feature blocks (pathway counterfactual), mean-diff blocks, or LOCO PO-risk drivers. Prefer this after detect/classify.",
  effect: "read",
  inputSchema: z.object({
    fixture: fixtureEnum.optional().describe("Built-in synthetic fixture id"),
    mode: z
      .enum(["pathway", "block_mean", "loco"])
      .default("pathway")
      .describe("Attribution mode; pathway is the default for tabular drift"),
    XRef: z.array(z.array(z.number())).optional(),
    XNew: z.array(z.array(z.number())).optional(),
    yRef: z.array(z.number()).optional(),
    yNew: z.array(z.number()).optional(),
    blocks: z.record(z.string(), z.array(z.number())).optional(),
    groundTruthBlocks: z.array(z.string()).optional(),
    featureNames: z.array(z.string()).optional(),
  }),
  async execute(input) {
    let XRef = input.XRef;
    let XNew = input.XNew;
    let yRef = input.yRef;
    let yNew = input.yNew;
    let blocks = input.blocks;
    let groundTruth = input.groundTruthBlocks ?? [];
    let featureNames = input.featureNames;

    if (input.fixture) {
      const f = loadFixture(input.fixture);
      XRef = f.ref.X;
      XNew = f.neu.X;
      yRef = f.ref.y;
      yNew = f.neu.y;
      blocks = f.blocks;
      groundTruth = f.driftedBlocks;
      featureNames = f.ref.featureNames;
    }

    if (!XRef || !XNew) {
      throw new Error("Provide fixture or both XRef and XNew matrices");
    }
    if (!blocks || Object.keys(blocks).length === 0) {
      const p = XRef[0].length;
      blocks = {
        block_0: Array.from({ length: Math.ceil(p / 2) }, (_, i) => i),
        block_1: Array.from({ length: Math.floor(p / 2) }, (_, i) => i + Math.ceil(p / 2)),
      };
    }

    if (input.mode === "loco") {
      if (!yRef || !yNew) {
        throw new Error("loco mode requires yRef and yNew (or a labeled fixture)");
      }
      const attr = locoPoAttribution(
        XRef,
        yRef,
        XNew,
        yNew,
        featureNames,
        groundTruth,
      );
      return {
        ...attr,
        fixture: input.fixture ?? null,
        glance: {
          headline: `top=${attr.top ?? "—"} · hit=${attr.topHit}`,
          board: shareBoard(
            attr.ranked.map((h) => ({ name: h.name, share: h.share, score: h.score })),
            "LOCO drivers",
          ),
        },
      };
    }

    if (input.mode === "block_mean") {
      const attr = blockMeanAttribution(
        XRef,
        XNew,
        blocks,
        groundTruth,
        featureNames,
      );
      return {
        ...attr,
        fixture: input.fixture ?? null,
        glance: {
          headline: `top=${attr.top ?? "—"} · hit=${attr.topHit}`,
          board: shareBoard(
            attr.ranked.map((h) => ({ name: h.name, share: h.share, score: h.score })),
            "Block mean-diff",
          ),
        },
      };
    }

    const attr = pathwayBlockAttribution(XRef, XNew, blocks, groundTruth);
    return {
      ...attr,
      fixture: input.fixture ?? null,
      glance: {
        headline: `top=${attr.top ?? "—"} · hit=${attr.topHit}`,
        board: shareBoard(
          attr.ranked.map((h) => ({ name: h.name, share: h.share, score: h.score })),
          "Pathway blocks",
        ),
      },
    };
  },
});
