import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { shareBoard, shiftGlance } from "../lib/glance.js";
import {
  attributeTextMetaFixture,
  attributeTextWindows,
  listTextMetaFixtures,
} from "../lib/text_meta.js";

export default defineTool({
  description:
    "Attribute text distribution shift using spaCy + VADER meta-features (style / syntax / NER / sentiment blocks). Prefer fixtures sentiment_shift, entity_shift, style_shift.",
  effect: "read",
  inputSchema: z.object({
    fixture: z
      .enum(["sentiment_shift", "entity_shift", "style_shift"])
      .optional(),
    refTexts: z.array(z.string()).optional(),
    newTexts: z.array(z.string()).optional(),
    groundTruthBlocks: z
      .array(z.enum(["spacy_style", "spacy_syntax", "spacy_ner", "vader_sent"]))
      .optional(),
    listOnly: z
      .boolean()
      .default(false)
      .describe("If true, only list fixtures and meta-feature blocks"),
  }),
  async execute(input) {
    if (input.listOnly) {
      return {
        fixtures: listTextMetaFixtures(),
        blocks: {
          spacy_style: "length / punctuation / type-token (spaCy)",
          spacy_syntax: "POS mix NOUN/VERB/ADJ/ADV/PROPN (spaCy)",
          spacy_ner: "entity density + PERSON/ORG/GPE/DATE (spaCy)",
          vader_sent: "compound/pos/neu/neg (VADER)",
        },
        packages: ["spacy:en_core_web_sm", "vaderSentiment"],
      };
    }

    const attr = input.fixture
      ? attributeTextMetaFixture(input.fixture)
      : attributeTextWindows({
          ref: input.refTexts ?? [],
          new: input.newTexts ?? [],
          groundTruth: input.groundTruthBlocks,
        });

    if (attr.error) {
      return { ok: false, ...attr };
    }
    if (!input.fixture && (!input.refTexts?.length || !input.newTexts?.length)) {
      throw new Error("Provide fixture or both refTexts and newTexts");
    }

    const rows = attr.ranked.map((h) => ({
      name: h.name,
      share: h.share,
      score: h.score,
    }));
    const action =
      attr.top === "vader_sent"
        ? "Sentiment block drifted — check label/prior or VADER-sensitive prompts."
        : attr.top === "spacy_ner"
          ? "NER block drifted — entity mix changed; refresh NER-aware features."
          : attr.top === "spacy_style"
            ? "Style/length drifted — review verbosity and punctuation regime."
            : attr.top === "spacy_syntax"
              ? "POS mix drifted — syntactic register shifted."
              : "Inspect top meta-feature block.";

    const glance = shiftGlance({
      fixture: input.fixture ?? "custom_texts",
      shiftType: "covariate",
      confidence: attr.topHit === false ? 0.55 : 0.85,
      top: attr.top,
      topHit: attr.topHit,
      rows,
      action,
    });

    return {
      ok: true,
      packages: attr.packages,
      fixture: input.fixture ?? null,
      description: attr.description ?? null,
      glance: {
        ...glance,
        board: shareBoard(rows, "Meta-feature blocks (spaCy+VADER)"),
      },
      attribution: {
        method: attr.method,
        top: attr.top,
        topHit: attr.topHit,
        groundTruth: attr.groundTruth,
        ranked: attr.ranked,
        featureRanked: attr.featureRanked,
      },
      means: {
        ref: attr.meanRef ?? null,
        new: attr.meanNew ?? null,
      },
      recommendation: { nextAction: action },
    };
  },
});
