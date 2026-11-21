import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["attribution", "regression"],
  description: "Pathway attribution on covariate_shift should hit block_1.",
  async test(t) {
    await t.send(
      "Attribute the covariate_shift fixture with pathway mode. Which block drifted?",
    );
    t.succeeded();
    t.calledTool("attribute_shift");
    t.notCalledTool("promote_to_dataset");
  },
});
