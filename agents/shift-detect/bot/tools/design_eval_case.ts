import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

/**
 * Scaffold a new eval case with forced clarity fields.
 * Does not mutate the repo dataset; returns a reviewable draft for humans.
 */
export default defineTool({
  description:
    "Design a new rigorous eval case draft (id, modality, axes, prompt, reference, tags, version) for review before dataset merge.",
  effect: "read",
  inputSchema: z.object({
    intent: z
      .string()
      .describe("What capability gap this case is meant to catch"),
    modality: z
      .enum(["text", "vision_language", "speech_text", "multimodal"])
      .default("text"),
    axes: z
      .array(
        z.enum([
          "factuality",
          "instruction_following",
          "reasoning",
          "safety",
          "groundedness",
          "consistency",
          "refusal_quality",
          "multimodal_alignment",
        ]),
      )
      .min(1),
    prompt: z.string(),
    reference: z.string(),
    mediaRef: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
  async execute(input) {
    const slug = input.intent
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "")
      .slice(0, 40);
    const prefix =
      input.modality === "text"
        ? "txt"
        : input.modality === "vision_language"
          ? "vl"
          : input.modality === "speech_text"
            ? "sp"
            : "mm";
    const draft = {
      id: `${prefix}_${slug || "case"}_v1`,
      modality: input.modality,
      axes: input.axes,
      prompt: input.prompt,
      reference: input.reference,
      mediaRef: input.mediaRef,
      tags: Array.from(new Set(["draft", ...input.tags])),
      version: "2026.11.1-draft",
      reviewChecklist: [
        "Single capability intent stated",
        "Reference is checkable without the model",
        "Axes match the failure mode",
        "No ambiguous scoring language",
        "Version bump on any prompt/reference edit",
      ],
      intent: input.intent,
    };
    return { draft, status: "awaiting_human_review" };
  },
});
