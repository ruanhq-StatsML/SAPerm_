# ModaSense digestion

- dataset: `modasense_emotion`
- n=500, p=50, y_rate=0.664
- layout: `X=[M_text, M_image, M_audio]`
- modalities: text, image, audio

## Modality dims

| modality | dim |
| --- | ---: |
| text | 37 |
| image | 7 |
| audio | 6 |

## Blocks

- `text_meta` (spacy+vader): 37 feats
- `image_meta` (opencv): 7 feats
- `audio_meta` (wave+numpy): 6 feats

## Downstream

```bash
python power_mm_modasense.py --B 80
python case_modasense_business.py
python run_mmd_two_datasets.py
```

Scientific read: `results/modasense_scientific_case.md`.
