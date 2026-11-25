# RecSys OnlineRFPerm × CFPerm board

Synthetic rating stream. Reference = batch 0, burn-in = 4, planted
onset at batch `8`. FDR = ADDIS (α=0.05). Auto-refresh resets
state after each reject (Appendix A.4).

## Aggregate

| scenario | detect_rate | mean_delay | mean_false_alarms | top_block_mode |
| --- | --- | --- | --- | --- |
| none | 0.000 | nan | 0.000 |  |
| concept | 1.000 | 0.000 | 0.000 | item |
| covariate | 1.000 | 0.500 | 0.250 | user |
| both | 1.000 | 0.000 | 0.000 | user |

## Reading

- `none`: false alarms should stay near zero (FDR sanity).
- `concept`: item-taste flip — expect detection after onset; top block often `item` / `review`.
- `covariate`: user-demo shift — top block often `user`.
- `both`: either driver is acceptable; delay should stay small.

## Per-replicate

```json
[
  {
    "scenario": "none",
    "n_reject": 0,
    "false_alarms": 0,
    "detected": 0,
    "delay": NaN,
    "first_reject_t": NaN,
    "top_block_at_detect": "",
    "rep": 0,
    "ticket_top3": ""
  },
  {
    "scenario": "none",
    "n_reject": 0,
    "false_alarms": 0,
    "detected": 0,
    "delay": NaN,
    "first_reject_t": NaN,
    "top_block_at_detect": "",
    "rep": 1,
    "ticket_top3": ""
  },
  {
    "scenario": "none",
    "n_reject": 0,
    "false_alarms": 0,
    "detected": 0,
    "delay": NaN,
    "first_reject_t": NaN,
    "top_block_at_detect": "",
    "rep": 2,
    "ticket_top3": ""
  },
  {
    "scenario": "none",
    "n_reject": 0,
    "false_alarms": 0,
    "detected": 0,
    "delay": NaN,
    "first_reject_t": NaN,
    "top_block_at_detect": "",
    "rep": 3,
    "ticket_top3": ""
  },
  {
    "scenario": "concept",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "item",
    "rep": 0,
    "ticket_top3": "item,review,user"
  },
  {
    "scenario": "concept",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "item",
    "rep": 1,
    "ticket_top3": "item,review,user"
  },
  {
    "scenario": "concept",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "item",
    "rep": 2,
    "ticket_top3": "item,review,context"
  },
  {
    "scenario": "concept",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "item",
    "rep": 3,
    "ticket_top3": "item,review,context"
  },
  {
    "scenario": "covariate",
    "n_reject": 2,
    "false_alarms": 1,
    "detected": 1,
    "delay": 2.0,
    "first_reject_t": 10.0,
    "top_block_at_detect": "user",
    "rep": 0,
    "ticket_top3": "review,context,item"
  },
  {
    "scenario": "covariate",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 1,
    "ticket_top3": "user,review,context"
  },
  {
    "scenario": "covariate",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 2,
    "ticket_top3": "user,review,context"
  },
  {
    "scenario": "covariate",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 3,
    "ticket_top3": "user,review,context"
  },
  {
    "scenario": "both",
    "n_reject": 2,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 0,
    "ticket_top3": "user,review,item"
  },
  {
    "scenario": "both",
    "n_reject": 1,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 1,
    "ticket_top3": "user,review,context"
  },
  {
    "scenario": "both",
    "n_reject": 2,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 2,
    "ticket_top3": "user,review,item"
  },
  {
    "scenario": "both",
    "n_reject": 2,
    "false_alarms": 0,
    "detected": 1,
    "delay": 0.0,
    "first_reject_t": 8.0,
    "top_block_at_detect": "user",
    "rep": 3,
    "ticket_top3": "user,review,context"
  }
]
```
