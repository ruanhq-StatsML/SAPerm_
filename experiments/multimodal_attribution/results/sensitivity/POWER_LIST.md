# POWER_LIST (live)

_auto · raw=342 · signed=48_

## Unique insight (λ=−1.8)

- $I_{cal}≪0$ ⇒ antagonistic **decision** for sentiment label.
- Singles weak + pair CF>0 ⇒ joint structure still needed for **detection**.

## Signed PO (calibrated)

| dataset | combo | pair | λ | n | mean I_cal | frac I_cal>0 | power|I| | signed− |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | text_audio | -1.8 | 24 | -14.94 | 0.08 | 0.00 | 0.00 |
| sens_interact_text_audio_neg1p8 | text_audio_video | text_audio | -1.8 | 24 | -81.05 | 0.00 | 0.00 | 0.00 |

## CF / DR / RF

| dataset | combo | method | power | n |
|---|---|---|---:|---:|
| sens_interact_text_audio_neg1p8 | audio | cfperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | audio | drperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | audio | rfperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text | cfperm | 0.90 | 10 |
| sens_interact_text_audio_neg1p8 | text | drperm | 0.30 | 10 |
| sens_interact_text_audio_neg1p8 | text | rfperm | 0.00 | 10 |
| sens_interact_text_audio_neg1p8 | text_audio | cfperm | 0.54 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio | drperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio | rfperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | cfperm | 0.62 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | drperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | rfperm | 0.00 | 24 |

## Fusion lift

- **cfperm**: text=0.90 audio=0.00 pair=0.54 tav=0.62 · lift=-0.36
- **drperm**: text=0.30 audio=0.00 pair=0.00 tav=0.00 · lift=-0.30
- **rfperm**: text=0.00 audio=0.00 pair=0.00 tav=0.00 · lift=+0.00
