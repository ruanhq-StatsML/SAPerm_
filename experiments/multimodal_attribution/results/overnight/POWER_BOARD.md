# Concept-drift POWER BOARD — 2026-11-21 01:43 UTC

Priority dataset `msrvtt_concept_top30`. Methods CF/RF/DRPerm (accel: n_perm=40 after resume).

## text_audio (priority — B=120 done)

| method | B | power@0.05 | mean p |
|---|---:|---:|---:|
| cfperm | 120 | 0.600 | 0.282 |
| drperm | 120 | 0.400 | 0.257 |
| rfperm | 150 | 1.000 | 0.007 |

## All combos (in progress for CF/DR on non-TA)

| combo | method | B | power@0.05 |
|---|---|---:|---:|
| text_audio | cfperm | 120 | 0.600 |
| text_audio | drperm | 120 | 0.400 |
| text_audio | rfperm | 150 | 1.000 |
| text_audio_video | rfperm | 150 | 0.933 |
| text_video | cfperm | 5 | 0.400 |
| text_video | drperm | 8 | 0.625 |
| text_video | rfperm | 150 | 0.987 |
