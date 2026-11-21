import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["smoke"],
  description: "Attribution report tool is called for a covariate fixture.",
  async test(t) {
    await t.send(
      "Run a one-shot attribution report on the covariate_shift fixture and tell me the top drifted block.",
    );
    t.succeeded();
    t.calledTool("score_report");
  },
});
