# POWER_LIST (live)

_auto · α=0.05 · rows=55_

## Signed PO (calibrated)

| dataset | pair | λ | n | mean I | mean I_cal | frac I_cal>0 | power\|I\| | signed+ | signed− |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | -1.8 | 11 | +5.48 | -6.59 | 0.09 | 0.00 | 0.00 | 0.00 |

> Unique early insight: under λ=−1.8, **raw I stays >0** but **I_cal=I−PO(full) is mostly <0** → use calibrated sign for antagonism; matches the negative plant.

## CF / DR / RF power by combo

| dataset | combo | λ | method | power | n | mean p |
|---|---|---:|---|---:|---:|---:|
| sens_interact_text_audio_neg1p8 | text_audio | -1.8 | cfperm | 0.64 | 11 | 0.083 |
| sens_interact_text_audio_neg1p8 | text_audio | -1.8 | drperm | 0.00 | 11 | 1.000 |
| sens_interact_text_audio_neg1p8 | text_audio | -1.8 | rfperm | 0.00 | 11 | 0.320 |

## Sentiment-mechanism reading (so far)

- Cell under fill: `sens_interact_text_audio` with **λ=−1.8** (antagonistic plant).
- Calibrated $I_{cal}<0$ ⇒ text×audio **conflict** for the label decision.
- CFPerm pair power already non-trivial while DR/RF still quiet at small n — watch as singles/TAV land for fusion-lift contrast.

