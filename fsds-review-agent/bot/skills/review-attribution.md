# Review an FSDS attribution board

Use this when someone asks for 审核 / attribution review / whether arms
are working evenly (均匀发力).

## Steps

1. Collect arm names and raw scores (block distances, PO-risk drops,
   effort units, or VIMP sums). If only shares are given, treat them as
   scores.
2. Call `review_board` with those names and scores. Prefer that over
   calling `fsds_attribute` and `fsds_incentives` separately unless the
   user wants intermediate numbers only.
3. Read `review.verdict`:
   - `balanced` — say so briefly and quote evenness.
   - `needs_rebalance` / `concentrated` — list under-selected arms and
     recommend raising `gamma` or keeping `lambda` on the loud FSDS share.
4. On a GitHub PR session, draft markdown that includes top share,
   selection evenness, under-selected arms, and next actions, then call
   `submit_review` once (`comment` by default; `request_changes` only when
   the board is concentrated and the PR claims balance).
5. Outside GitHub, paste the same markdown in chat. Do not invent scores.

## Output shape

- One-line verdict
- Top attributed arm + share
- Selection evenness and under-selected arms
- Concrete next actions (lambda / gamma / who to nudge)
