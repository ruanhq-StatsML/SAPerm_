# Concept-drift power board (CF/RF/DRPerm, RF, n_perm=150)

_rows=2608 after dedupe; α=0.05_

## `msrvtt_concept_pc31_100` / `text_audio` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 0.647 | 0.061 | 150 |

## `msrvtt_concept_pc31_100` / `text_audio_video` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 0.573 | 0.069 | 150 |

## `msrvtt_concept_pc31_100` / `text_video` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 0.693 | 0.046 | 150 |

## `msrvtt_concept_top30` / `text_audio` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| zwl | 1.000 | 0.000 | 69 |
| skk13 | 1.000 | 0.000 | 69 |
| **rfperm** | 1.000 | 0.007 | 150 |
| **cfperm** | 0.710 | 0.205 | 69 |
| **drperm** | 0.594 | 0.179 | 69 |
| c2st_XY | 0.145 | 0.746 | 69 |
| c2st_X | 0.000 | 1.000 | 69 |
| kmmd | 0.000 | 1.000 | 69 |
| chen10 | 0.000 | 1.000 | 69 |
| bai96 | 0.000 | 1.000 | 69 |
| miles11 | 0.000 | 1.000 | 69 |
| pan14 | 0.000 | 1.000 | 69 |
| mmd_rbf | 0.000 | 1.000 | 69 |
| sri08 | 0.000 | 1.000 | 69 |
| xu16 | 0.000 | 1.000 | 69 |

## `msrvtt_concept_top30` / `text_audio_video` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 0.933 | 0.018 | 150 |

## `msrvtt_concept_top30` / `text_video` / `concept_drift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| **rfperm** | 0.987 | 0.011 | 150 |

## `msrvtt_null_top30` / `text_audio` / `no_shift`

| method | power | mean p | n_rep |
|---|---:|---:|---:|
| skk13 | 1.000 | 0.000 | 53 |
| zwl | 0.132 | 0.432 | 53 |
| c2st_XY | 0.075 | 0.533 | 53 |
| kmmd | 0.057 | 0.598 | 53 |
| xu16 | 0.057 | 0.564 | 53 |
| **cfperm** | 0.057 | 0.513 | 53 |
| pan14 | 0.038 | 0.550 | 53 |
| **drperm** | 0.038 | 0.509 | 53 |
| c2st_X | 0.038 | 0.540 | 53 |
| bai96 | 0.019 | 0.543 | 53 |
| mmd_rbf | 0.019 | 0.570 | 53 |
| miles11 | 0.019 | 0.503 | 53 |
| sri08 | 0.019 | 0.552 | 53 |
| chen10 | 0.000 | 0.887 | 53 |
