# Concept-drift POWER BOARD — 2026-11-21 02:35 UTC

Dataset `msrvtt_concept_top30`. Streams: CF∥DR∥classical∥MSR (4×1 core).

## text_audio FULL table

| method | B | power@0.05 | mean p |
|---|---:|---:|---:|
| bai96 | 91 | 0.000 | 1.000 |
| c2st_X | 91 | 0.000 | 1.000 |
| c2st_XY | 91 | 0.110 | 0.807 |
|  **cfperm** | 120 | 0.600 | 0.282 |
| chen10 | 91 | 0.000 | 1.000 |
|  **drperm** | 120 | 0.400 | 0.257 |
| kmmd | 91 | 0.000 | 1.000 |
| miles11 | 91 | 0.000 | 1.000 |
| mmd_rbf | 91 | 0.000 | 1.000 |
| pan14 | 91 | 0.000 | 1.000 |
|  **rfperm** | 150 | 1.000 | 0.007 |
| skk13 | 91 | 1.000 | 0.000 |
| sri08 | 91 | 0.000 | 1.000 |
| xu16 | 91 | 0.000 | 1.000 |
| zwl | 91 | 1.000 | 0.000 |

## null_top30 text_audio (type-I contrast)

| method | B | reject@0.05 | mean p |
|---|---:|---:|---:|
| bai96 | 53 | 0.019 | 0.543 |
| c2st_X | 53 | 0.038 | 0.540 |
| c2st_XY | 53 | 0.075 | 0.533 |
| cfperm | 53 | 0.057 | 0.513 |
| chen10 | 53 | 0.000 | 0.887 |
| drperm | 53 | 0.038 | 0.509 |
| kmmd | 53 | 0.057 | 0.598 |
| miles11 | 53 | 0.019 | 0.503 |
| mmd_rbf | 53 | 0.019 | 0.570 |
| pan14 | 53 | 0.038 | 0.550 |
| skk13 | 53 | 1.000 | 0.000 |
| sri08 | 53 | 0.019 | 0.552 |
| xu16 | 53 | 0.057 | 0.564 |
| zwl | 53 | 0.132 | 0.432 |

## All concept_top30 combos

| combo | method | B | power@0.05 |
|---|---|---:|---:|
| text_audio | bai96 | 91 | 0.000 |
| text_audio | c2st_X | 91 | 0.000 |
| text_audio | c2st_XY | 91 | 0.110 |
| text_audio | cfperm | 120 | 0.600 |
| text_audio | chen10 | 91 | 0.000 |
| text_audio | drperm | 120 | 0.400 |
| text_audio | kmmd | 91 | 0.000 |
| text_audio | miles11 | 91 | 0.000 |
| text_audio | mmd_rbf | 91 | 0.000 |
| text_audio | pan14 | 91 | 0.000 |
| text_audio | rfperm | 150 | 1.000 |
| text_audio | skk13 | 91 | 1.000 |
| text_audio | sri08 | 91 | 0.000 |
| text_audio | xu16 | 91 | 0.000 |
| text_audio | zwl | 91 | 1.000 |
| text_audio_video | rfperm | 150 | 0.933 |
| text_video | cfperm | 115 | 0.330 |
| text_video | drperm | 68 | 0.250 |
| text_video | rfperm | 150 | 0.987 |

### Progress / ETA

- CF text_video ~114/120 → then TAV; DR TV ~68/120; classical TA ~90/120; MSR AV ~109/120

- Priority CF/RF/DR text_audio **done** at B≥120. Classical/MSR still filling.
