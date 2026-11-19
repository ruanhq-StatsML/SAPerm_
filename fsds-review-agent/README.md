# fsds-review-agent

审核 + 归因 Agent：用 FSDS 份额做 Gaussian Thompson 激励，推动各 arm / 特征块 / 贡献方均匀发力。

## What it does

- Attribute raw scores → FSDS-style contribution shares
- Draw incentives: `N(mu, se) - λ·share + γ/√(n+1)`
- Audit a board for concentration vs even work
- Optionally post the review on GitHub PRs that mention FSDS / attribution

## Local loop

```bash
cd fsds-review-agent
npx @grok/agent-sdk validate --dir .
npx tsc --noEmit -p .
npx @grok/agent-sdk tool call review_board --dir . --input '{
  "names": ["block_0","block_1","block_2","block_3"],
  "scores": [0.1, 1.5, 0.2, 0.15],
  "seed": 2026
}'
npx @grok/agent-sdk login   # once, for model turns
npx @grok/agent-sdk run --dir . --message "审核 block shares 0.1/1.5/0.2/0.15"
npx @grok/agent-sdk eval --dir .
npx @grok/agent-sdk dev     # playground
```

## Deploy

```bash
npx @grok/agent-sdk validate --dir .
npx @grok/agent-sdk deploy --dir . --dry-run
npx @grok/agent-sdk deploy --dir .   # run in your own terminal
```

GitHub: `bot/channels/github.ts` wakes on FSDS / attribution PRs and
`@fsds-review-agent` mentions. Local validate/tool-call does not need
`cursorAccount`. For hosted SCM wakes:

```bash
npx @grok/agent-sdk serve --dir . --repo ruanhq-StatsML/SAPerm_
# deploy:
npx @grok/agent-sdk deploy --dir . --cursor-events-repo ruanhq-StatsML/SAPerm_
```
