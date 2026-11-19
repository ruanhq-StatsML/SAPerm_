# Multimodal higher-order interaction attribution

Concatenate **text / audio / video** embeddings (MSR-VTT style), keep a PCA
band (`top30` or `pc30_100`), then attribute distribution shift with block
leave-out **pseudo-outcome risk** interactions from `OOD_vimp_wrapper`.

## Pipeline

1. Extract embeddings (notebooks → `embeddings.py`):
   - text: MiniLM / Qwen-style encoders
   - audio: WavLM sliding windows
   - video: ViViT on MSR-VTT clips
2. Per-modality PCA → keep first 30 PCs **or** PCs 30–100.
3. Horizontal concat; record modality column index blocks (~500 rows / batch).
4. Second-order: `interaction = po(-m1) + po(-m2) - po(-both)`
5. Third-order: inclusion–exclusion over the three modality blocks.

## Quick start (synthetic MSR-VTT-style, no GPU)

```bash
python -m experiments.multimodal_attribution --band top30 --n 500
python -m experiments.multimodal_attribution --band pc30_100 --n 500 --out /tmp/mm_interaction.json
```

## Precomputed embeddings

```bash
python -m experiments.multimodal_attribution \
  --text-ref text_ref.npy --audio-ref audio_ref.npy --video-ref video_ref.npy \
  --text-new text_new.npy --audio-new audio_new.npy --video-new video_new.npy \
  --y-ref y_ref.npy --y-new y_new.npy --band top30 --n 500
```

## API

```python
from OOD_vimp_wrapper import (
    higher_order_interaction_PO_risk,
    third_order_interaction_PO_risk,
)
from experiments.multimodal_attribution import build_msrvtt_style_batches

batch = build_msrvtt_style_batches(n_per_batch=500, band="top30")
idx = batch["modality_indices"]
pair = higher_order_interaction_PO_risk(
    batch["X"], batch["Y"], batch["T"], idx["text"], idx["video"]
)
triple = third_order_interaction_PO_risk(
    batch["X"], batch["Y"], batch["T"], idx["text"], idx["audio"], idx["video"]
)
```
