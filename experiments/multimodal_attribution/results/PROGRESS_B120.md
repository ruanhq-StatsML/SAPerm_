# Progress snapshot → retarget B=120
_snapshotted 2026-11-20T17:31:49Z_

Checkpoint keeps finished cells. Subsequent runs use `--n-rep 120` (skip rep≥120).

## MSR_overnight (rows=5011)

| dataset | combo | scenario | min_rep | max_rep | #methods≥120 | still short |
|---|---|---|---:|---:|---:|---|
| msrvtt_null_top30 | audio_video | no_shift | 49 | 49 | 0 | bai96:71, c2st_X:71, c2st_XY:71, chen10:71, crosscorr_audio_video:71, interact_PO_audio_video:71, kmmd:71, miles11:71 …(+6) |
| msrvtt_null_top30 | text_audio | no_shift | 12 | 149 | 14 | cfperm:108, drperm:108, rfperm:108 |
| msrvtt_null_top30 | text_audio_video | no_shift | 3 | 3 | 0 | bai96:117, c2st_X:117, c2st_XY:117, chen10:117, crosscorr_audio_video:117, crosscorr_text_audio:117, crosscorr_text_video:117, interact_PO_audio_video:117 …(+10) |
| msrvtt_null_top30 | text_video | no_shift | 149 | 149 | 14 | — |

## concept_drift (rows=2090)

| dataset | combo | scenario | min_rep | max_rep | #methods≥120 | still short |
|---|---|---|---:|---:|---:|---|
| msrvtt_concept_pc31_100 | text_audio | concept_drift | 149 | 149 | 1 | — |
| msrvtt_concept_pc31_100 | text_audio_video | concept_drift | 149 | 149 | 1 | — |
| msrvtt_concept_pc31_100 | text_video | concept_drift | 149 | 149 | 1 | — |
| msrvtt_concept_top30 | text_audio | concept_drift | 31 | 149 | 1 | bai96:89, c2st_X:89, c2st_XY:89, cfperm:89, chen10:89, drperm:89, kmmd:89, miles11:89 …(+6) |
| msrvtt_concept_top30 | text_audio_video | concept_drift | 149 | 149 | 1 | — |
| msrvtt_concept_top30 | text_video | concept_drift | 149 | 149 | 1 | — |
| msrvtt_null_top30 | text_audio | no_shift | 52 | 52 | 0 | bai96:68, c2st_X:68, c2st_XY:68, cfperm:68, chen10:68, drperm:68, kmmd:68, miles11:68 …(+6) |

## emotion_overnight (rows=170)

| dataset | combo | scenario | min_rep | max_rep | #methods≥120 | still short |
|---|---|---|---:|---:|---:|---|
| mosei | text_audio |  | 9 | 9 | 0 | bai96:111, c2st_X:111, c2st_XY:111, cfperm:111, chen10:111, crosscorr_text_audio:111, drperm:111, interact_PO_text_audio:111 …(+9) |
