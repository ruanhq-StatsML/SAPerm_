import { defineEval, includes } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["smoke"],
  description: "Reviews a concentrated four-arm board with review_board.",
  async test(t) {
    await t.send(
      [
        "审核这块归因板，看 FSDS 激励能不能让他们均匀发力。",
        "arms: block_0=0.1, block_1=1.5, block_2=0.2, block_3=0.15",
        "用 review_board，报告 verdict 和 under-selected arms。",
      ].join("\n"),
    );
    t.succeeded();
    t.calledTool("review_board");
    t.check(t.reply, includes(/balanced|needs_rebalance|concentrated|均匀|even/i));
  },
});
