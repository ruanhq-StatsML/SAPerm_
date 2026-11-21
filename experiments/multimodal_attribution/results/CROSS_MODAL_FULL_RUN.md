# Cross-modality full run plan (口径 locked)

**n_sample=500 · B=120 · n_perm∈{120,150} · α=0.05**

## Streams (all on)

| stream | what | n_perm | status |
|---|---|---|---|
| **sensitivity** | CF/DR/RF × ±interaction (3 pairs) × ±mean (3 mods) × null; combos single/pair/TAV; signed $I_{cal}$ | **150** | running |
| **overnight cross** | classical + `interact_PO`/`crosscorr` on mean_shift + interaction_shift datasets (+ pc30_100) | **120** | launched |
| **overnight null** | null TAV + pair type-I stack to B=120 | 120 | running |
| **concept null CF/DR** | type-I top-up 53→120 | 150 | running |
| **concept power** | CF/DR/RF + classical T×A/T×V/TAV | 150 | **delivery done** |

## Combos covered

`text`, `audio`, `video`, `text_audio`, `text_video`, `audio_video`, `text_audio_video`

## Delivery rule

Cite power only if realized **B ≥ 100**. Pilot B=24 archived.
