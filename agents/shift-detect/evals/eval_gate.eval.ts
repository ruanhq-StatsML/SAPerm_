import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["gate", "regression"],
  description: "Agent runs the fixture suite and applies Eval Gate before promoting.",
  async test(t) {
    await t.send(
      "Run the offline fixture suite and tell me whether the Eval Gate passes. Do not promote anything.",
    );
    t.succeeded();
    t.calledTool("run_fixture_suite");
  },
});
