# POWER_LIST (live)

_auto · raw=300 · signed=48_

## Unique insight (λ=−1.8 text×audio, in progress)

- **$I_{cal}<0$** ⇒ sentiment decision is **antagonistic** (modalities conflict for $Y$).
- **Detection can still need the pair**: if singles ≈0 but `power_pair>0`, the shift lives in the *joint* structure even when that structure is antagonistic.
- So **sign($I_{cal}$)** = decision mechanism; **fusion lift** = detectability. They need not co-move under λ<0.

## Signed PO (calibrated)

| dataset | combo | pair | λ | n | mean I_cal | frac I_cal>0 | power|I| | signed− |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | text_audio | -1.8 | 24 | -14.94 | 0.08 | 0.00 | 0.00 |
| sens_interact_text_audio_neg1p8 | text_audio_video | text_audio | -1.8 | 24 | -81.05 | 0.00 | 0.00 | 0.00 |

## CF / DR / RF

| dataset | combo | method | power | n |
|---|---|---|---:|---:|
| sens_interact_text_audio_neg1p8 | audio | cfperm | 0.00 | 20 |
| sens_interact_text_audio_neg1p8 | audio | drperm | 0.00 | 20 |
| sens_interact_text_audio_neg1p8 | audio | rfperm | 0.00 | 20 |
| sens_interact_text_audio_neg1p8 | text_audio | cfperm | 0.54 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio | drperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio | rfperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | cfperm | 0.62 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | drperm | 0.00 | 24 |
| sens_interact_text_audio_neg1p8 | text_audio_video | rfperm | 0.00 | 24 |

## Fusion lift preview

- **cfperm**: text=nan audio=0.00 pair=0.54 tav=0.62 · lift(pair−max₁)=+0.54
- **drperm**: text=nan audio=0.00 pair=0.00 tav=0.00 · lift(pair−max₁)=+0.00
- **rfperm**: text=nan audio=0.00 pair=0.00 tav=0.00 · lift(pair−max₁)=+0.00
