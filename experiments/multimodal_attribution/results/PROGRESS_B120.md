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

## Accel restart (2026-11-21 01:32 UTC)

- Mode: **cfperm ∥ drperm only** (MSR/classical still off)
- Knobs: `--n-perm 40`, `CFDR_PERM_JOBS=2`, CF `top_k=4`, DR `n_splits=3`
- Observed: ~14s/rep CF, ~25s/rep DR (was ~100s / up to 26min)
- Target: `msrvtt_concept_top30` text_audio → B=120

## text_audio B=120 complete (2026-11-21 01:43 UTC)

| method | power@0.05 |
|---|---:|
| rfperm | 1.000 |
| cfperm | 0.600 |
| drperm | 0.400 |

CF∥DR still filling text_video + text_audio_video under accel knobs.

## overnight-power-check (2026-11-21 01:49 UTC)

- text_audio: CF/DR/RF all ≥120 — **done**
- text_video: CF ~30/120, DR ~21/120 (healthy, ~14–25s/rep)
- text_audio_video: CF/DR not started; RF done
- classical/MSR: **not** restarted (user: keep off)
- ETA remaining combos ≈ 1h (DR-bound)

## Restored classical + MSR (2026-11-21 01:59 UTC)

Streams (4 cores, 1 job each):
- `concept-cfperm` / `concept-drperm` — continue top30 (CFDR_PERM_JOBS=1, n_perm=40)
- `concept-classical` — 12 TST methods resume from ~73–76 → B=120
- `msr-overnight` — `overnight_power` resume null_top30 audio_video 103/120 → rest

## post-all-power-tables (2026-11-21 02:35 UTC)

- Jobs healthy (raw mtime <1min); no restart
- concept TA: CF/RF/DR done; classical ~90/120
- CF TV~114, DR TV~68; MSR AV~109/120; emotion mosei still B=10 (queued)
- ETA classical TA ~1h; MSR AV ~30min then TAV

## classical-msr-restored-check (2026-11-21 02:45 UTC)

- All 4 streams healthy (raw age <3min)
- classical TA **100/120**; MSR AV **112/120**
- CF TAV~12; DR TV~81; CF TV done

## overnight-power-check (2026-11-21 03:00 UTC)

| stream | status |
|---|---|
| cfperm | TV 120 ✓ · TAV 37/120 |
| drperm | TV 101/120 |
| classical | TA 114/120 |
| MSR | null AV 117/120 |

All healthy. ETA: classical TA ~30min; MSR AV ~10min then TAV; DR TV ~30min then TAV; CF TAV ~1h.

## classical-msr-restored-check (2026-11-21 03:27 UTC)

| stream | status |
|---|---|
| classical | **TA B=120 done** · TV ~14/120 · n_jobs=8 |
| MSR | **null AV B=120 done** · TAV from 4 · n_jobs=8 |
| cfperm | TAV 71/120 · CFDR_PERM_JOBS=4 |
| drperm | TV 120 ✓ · TAV ~5/120 · CFDR_PERM_JOBS=4 |

All healthy. Focus remaining TV/TAV fill.

## overnight-power-check (2026-11-21 04:02 UTC)

| stream | status |
|---|---|
| cfperm | TAV 113/120 |
| drperm | TAV 26/120 |
| classical | TA 120 ✓ · TV 31/120 · n_jobs=8 |
| MSR | AV 120 ✓ · TAV restarted (was hung >20m) · n_jobs=8 |
