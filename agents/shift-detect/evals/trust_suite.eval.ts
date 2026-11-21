import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["trust", "llm-eval", "regression"],
  description: "Offline trustworthy LLM suite is run for alignment + gate.",
  async test(t) {
    await t.send(
      "Run the trustworthy LLM eval suite and tell me whether the candidate checkpoint passes the Trust Eval Gate versus base-text-v1.",
    );
    t.succeeded();
    t.calledTool("run_trust_suite");
  },
});
