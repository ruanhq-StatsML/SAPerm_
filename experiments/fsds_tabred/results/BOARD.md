# FSDS Thompson on TabReD (classification)

Streams are time-ordered. Regression targets (Weather, Sberbank) are
median-binarized on the first half of the stream. Reward is negative Brier.

Serve mode: keep only the selected block in the RF predictor (FSDS
avoids high-shift blocks via the λ·w penalty). Reward = negative Brier.

| dataset | tag | mean_reward | mean_accuracy | arm_counts | late_w | elapsed_sec |
| --- | --- | ---: | ---: | --- | --- | ---: |
| weather | fsds_vimp | -0.1552 | 0.774 | `[16, 2, 14, 4]` | `[0.141, 0.106, 0.093, 0.66]` | 19.9 |
| weather | fsds_logo | -0.1344 | 0.804 | `[3, 2, 2, 29]` | `[0.317, 0.216, 0.468, 0.0]` | 5.8 |
| weather | ts_flat | -0.1363 | 0.801 | `[10, 2, 2, 22]` | `[0.25, 0.25, 0.25, 0.25]` | 0.1 |
| weather | boot_fsds_vimp | -0.1522 | 0.783 | `[18, 1, 14, 3]` | `[0.141, 0.106, 0.093, 0.66]` | 19.8 |
| ecom-offers | fsds_vimp | -0.1927 | 0.731 | `[14, 10, 4, 8]` | `[0.273, 0.215, 0.238, 0.275]` | 21.2 |
| ecom-offers | fsds_logo | -0.1900 | 0.731 | `[26, 2, 2, 6]` | `[0.041, 0.543, 0.218, 0.197]` | 6.8 |
| ecom-offers | ts_flat | -0.1916 | 0.731 | `[2, 3, 2, 29]` | `[0.25, 0.25, 0.25, 0.25]` | 0.1 |
| ecom-offers | boot_fsds_vimp | -0.1925 | 0.731 | `[16, 12, 1, 7]` | `[0.273, 0.215, 0.238, 0.275]` | 21.2 |
| sberbank-housing | fsds_vimp | -0.1439 | 0.778 | `[10, 10, 7, 9]` | `[0.234, 0.275, 0.243, 0.249]` | 51.4 |
| sberbank-housing | fsds_logo | -0.1462 | 0.780 | `[4, 16, 8, 8]` | `[0.31, 0.2, 0.303, 0.187]` | 25.3 |
| sberbank-housing | ts_flat | -0.1513 | 0.778 | `[2, 29, 2, 3]` | `[0.25, 0.25, 0.25, 0.25]` | 0.1 |
| sberbank-housing | boot_fsds_vimp | -0.1431 | 0.780 | `[8, 10, 10, 8]` | `[0.234, 0.275, 0.243, 0.249]` | 50.4 |
| homesite-insurance | fsds_vimp | -0.1283 | 0.817 | `[8, 8, 10, 10]` | `[0.231, 0.246, 0.185, 0.337]` | 35.7 |
| homesite-insurance | fsds_logo | -0.1276 | 0.817 | `[11, 8, 12, 5]` | `[0.059, 0.0, 0.442, 0.499]` | 10.5 |
| homesite-insurance | ts_flat | -0.1284 | 0.817 | `[3, 3, 24, 6]` | `[0.25, 0.25, 0.25, 0.25]` | 0.1 |
| homesite-insurance | boot_fsds_vimp | -0.1281 | 0.817 | `[9, 8, 10, 9]` | `[0.231, 0.246, 0.185, 0.337]` | 37.2 |
