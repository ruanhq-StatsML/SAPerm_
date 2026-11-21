# Concept-drift POWER BOARD — 2026-11-21 01:49 UTC

Dataset `msrvtt_concept_top30`. Accel: CF∥DR only (`n_perm=40`); MSR/classical **off**.

## Full top30 table

| combo | method | B | power@0.05 | mean p |
|---|---|---:|---:|---:|
| text_audio | cfperm | 120 | 0.600 | 0.282 |
| text_audio | drperm | 120 | 0.400 | 0.257 |
| text_audio | rfperm | 150 | 1.000 | 0.007 |
| text_audio_video | rfperm | 150 | 0.933 | 0.018 |
| text_video | cfperm | 31 | 0.355 | 0.462 |
| text_video | drperm | 22 | 0.409 | 0.294 |
| text_video | rfperm | 150 | 0.987 | 0.011 |

## Priority text_audio (B=120 done)

| method | B | power@0.05 |
|---|---:|---:|
| rfperm | 150 | 1.000 |
| cfperm | 120 | 0.600 |
| drperm | 120 | 0.400 |

CF∥DR filling `text_video` / `text_audio_video`. ETA ~DR-bound ~1h.
