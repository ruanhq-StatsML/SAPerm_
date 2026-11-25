# ModaSense data digestion

One path: **read → multimodal meta → schema**. No embedding PCA on this dataset.

```
parquet (ModaSense/emotion)
  ├─ text_propositional  → spaCy + VADER          → M_text
  ├─ image_perceptual    → OpenCV brightness/…    → M_image
  └─ audio_perceptual    → WAV rms/zcr/…          → M_audio
                ↓
         X = [M_text, M_image, M_audio]
                ↓
         schema.modality / schema.blocks   (CF LOCO indices)
```

## Entrypoint

```bash
cd agents/shift-detect/python
# first time: download parquet under data/modasense/
python -m datasets.modasense_emotion --n-max 500 --audio --write
```

Writes:

| artifact | role |
| --- | --- |
| `results/mm3_cache_modasense_n500.npz` | feature cache (M, y, names) |
| `results/digest_modasense_mm3_n500.{json,md}` | schema + dims summary |

API used by power / MMD / localization scripts:

```python
from datasets.modasense_emotion import digest
d = digest(500, with_audio=True)
X, y, schema = d.M, d.y, d.schema   # Z is None (meta-only)
```

## Why this layout

| choice | reason |
| --- | --- |
| meta-only (`Z=None`) | ModaSense thesis is modality blocks, not BERT PCA |
| fixed column order | text → image → audio; CFPerm LOCO uses `schema.modality` |
| binary Y from `text_label` | neg = {fear, sad, angry, disgust}; designed label conflict P≠=1 |
| cache by `n` | rebuild only when n or extractors change |

## Feature packages

| block | package | examples |
| --- | --- | --- |
| `text_meta` | spaCy + VADER | `vader_neg`, `compound`, POS / NER |
| `image_meta` | OpenCV | `img_brightness`, `img_contrast`, `img_hue_std` |
| `audio_meta` | wave + numpy | `aud_rms_energy`, `aud_zcr`, `aud_silence_ratio` |

Exact names live in `extract_features.FEATURE_NAMES`, `extract_mm_features.IMAGE_NAMES` / `AUDIO_NAMES`.  
Schema builder: `extract_mm_features.schema_mm(with_audio=True)`.

## Downstream (same Digested object)

```
digest
  → RFPerm / DRPerm / CFPerm     (power_mm_modasense.py)
  → ρ_PO = PO([Mt,Mi]) / PO(Mt)
  → per-modality MMD share π_m   (mmd_modalities.py)
  → carrier subset localization  (case_modasense_business.py)
```

Closed-chain scientific read (numbers only when RF + CF + π + ρ agree):  
`results/modasense_scientific_case.md`.
