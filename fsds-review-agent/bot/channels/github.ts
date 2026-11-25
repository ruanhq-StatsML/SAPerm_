import { defaultGitHubAuth, githubChannel } from "@grok/agent-sdk/channels/github";

const INTEREST =
  /fsds|attribution|thompson|drift|vimp|po[-_]?risk|tabred|incentive|均匀|归因|审核/i;

/**
 * Local / dry bring-up: no cursorAccount, so validate + tool call work
 * without a Cursor login. For hosted wakes, serve/deploy with
 * `--repo ruanhq-StatsML/SAPerm_` (and/or SAPerm) plus a Cursor credential
 * so the host can attach the SCM event stream.
 */
export default githubChannel({
  botName: "fsds-review-agent",
  onPullRequest: (ctx, pr) => {
    if (pr.draft) return null;
    if (pr.action !== "opened" && pr.action !== "ready_for_review" && pr.action !== "synchronize") {
      return null;
    }
    const blob = `${pr.title}\n${pr.body ?? ""}`;
    if (!INTEREST.test(blob)) return null;
    return { auth: defaultGitHubAuth(ctx) };
  },
  onComment: (ctx, comment) => {
    if (!/@fsds-review-agent\b/i.test(comment.body) && !/fsds[- ]review/i.test(comment.body)) {
      return null;
    }
    return { auth: defaultGitHubAuth(ctx) };
  },
});
