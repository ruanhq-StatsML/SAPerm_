import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["trust", "interpretability"],
  description: "Behavior explanation tool used on a weak multimodal miss.",
  async test(t) {
    await t.send(
      "Explain why weak-vl-v0 fails on vl_ocr_sign_v1 from an evaluation perspective.",
    );
    t.succeeded();
    t.calledTool("explain_llm_behavior");
  },
});
