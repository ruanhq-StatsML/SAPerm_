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

## Power board vs classical two-sample tests

Uses the working methods in `benchmark_methods/benchmark_method_tst.py`
plus multimodal interaction detectors on PCA-concatenated embeddings
(`benchmark_config.MODEL_REGISTRY` / `model_registry_class`):

```bash
python -m experiments.multimodal_attribution.power_benchmark \
  --n 500 --band top30 --n-rep 40 --n-perm 40 --with-po-interact \
  --out experiments/multimodal_attribution/results/power_board.csv
```

### Overnight: 8 datasets × modality combos × n=500

Nested PO interaction is slow — that is expected. Checkpointed runner:

```bash
python -m experiments.multimodal_attribution.overnight_power \
  --n 500 --n-rep 40 --n-perm 40 --with-po-interact \
  --out-dir experiments/multimodal_attribution/results/overnight
```

Datasets: null (top30 / pc30_100), mean shift (video / audio), interaction
(text×video, text×audio, audio×video, text×video on pc30_100).

Modality combos: `text_audio`, `text_video`, `audio_video`, `text_audio_video`.

Progress: `results/overnight/overnight.log`, raw p-values in `power_raw.csv`,
board in `POWER_BOARD.md`.

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
