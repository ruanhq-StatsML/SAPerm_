import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { listCases, listFixtureModels } from "../lib/llm_trust/datasets.js";
import { listRubrics } from "../lib/llm_trust/rubrics.js";

export default defineTool({
  description:
    "List trustworthy-eval catalog: versioned cases, rubrics, fixture models. Supports modality/tag filters.",
  effect: "read",
  inputSchema: z.object({
    modality: z
      .enum(["text", "vision_language", "speech_text", "multimodal"])
      .optional(),
    tag: z.string().optional(),
  }),
  async execute(input) {
    return {
      models: listFixtureModels(),
      rubrics: listRubrics().map((r) => ({
        id: r.id,
        axis: r.axis,
        weight: r.weight,
        modality: r.modality,
        description: r.description,
        levels: r.levels.map((l) => `${l.score}:${l.label}`),
      })),
      cases: listCases({
        modality: input.modality,
        tag: input.tag,
      }).map((c) => ({
        id: c.id,
        modality: c.modality,
        axes: c.axes,
        tags: c.tags,
        version: c.version,
        mediaRef: c.mediaRef ?? null,
      })),
    };
  },
});
