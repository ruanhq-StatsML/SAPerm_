import { prompt } from "@grok/agent-sdk";
import { parseGitHubPrContinuationKey } from "@grok/agent-sdk/channels/github";
import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

export default defineTool({
  description: prompt`
    Post the FSDS attribution / incentive review on this session's pull
    request. Call once after review_board (or fsds_attribute +
    fsds_incentives). Skip when there is no GitHub PR session.
  `,
  effect: "write",
  dryRunResult: ({ verdict, summary }) => ({
    posted: false,
    dryRun: true,
    verdict,
    summary,
  }),
  inputSchema: z.object({
    verdict: z.enum(["approve", "comment", "request_changes"]),
    summary: z.string().min(1).describe("Markdown body for the PR review."),
  }),
  async execute({ verdict, summary }, ctx) {
    if (ctx.session.purpose === "eval") {
      return { posted: false, reason: "eval" };
    }
    const ref = parseGitHubPrContinuationKey(ctx.session.continuationKey ?? "");
    if (ref === undefined) {
      return {
        posted: false,
        reason: "no_github_pr_session",
        hint: "Return the review markdown in chat instead.",
        summary,
      };
    }

    const event =
      verdict === "approve"
        ? "APPROVE"
        : verdict === "request_changes"
          ? "REQUEST_CHANGES"
          : "COMMENT";

    const octokit = await ctx.host.github.getOctokit();
    await octokit.rest.pulls.createReview({
      owner: ref.owner,
      repo: ref.repo,
      pull_number: ref.number,
      event,
      body: summary,
    });
    return { posted: true, pr: `${ref.owner}/${ref.repo}#${ref.number}`, event };
  },
});
