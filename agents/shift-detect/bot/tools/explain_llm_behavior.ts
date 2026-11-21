import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";
import { FIXTURE_OUTPUTS } from "../lib/llm_trust/datasets.js";
import { explainBehavior } from "../lib/llm_trust/interpret.js";

export default defineTool({
  description:
    "从评测角度解释模型行为: attribute a weak case to prompt/context/model/decoding/RAG/tool/safety/modality factors and suggest a fix.",
  effect: "read",
  inputSchema: z.object({
    modelId: z.string().default("weak-vl-v0"),
    caseId: z.string().default("vl_ocr_sign_v1"),
    text: z.string().optional(),
    symptom: z.string().optional(),
  }),
  async execute(input) {
    const text =
      input.text ??
      FIXTURE_OUTPUTS.find(
        (o) => o.modelId === input.modelId && o.caseId === input.caseId,
      )?.text;
    if (!text) throw new Error("Provide text or a known fixture model/case pair");
    return explainBehavior(
      { modelId: input.modelId, caseId: input.caseId, text },
      input.symptom,
    );
  },
});
