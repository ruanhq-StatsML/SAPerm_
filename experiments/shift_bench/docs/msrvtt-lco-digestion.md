# MSR-VTT / LCO-Omni digestion

Source (already on `cursor/multimodal-higher-order-interaction-e93c`):

- `experiments/multimodal_attribution/prepare_msrvtt_lco.py`
- `experiments/multimodal_attribution/msrvtt.py`
- `experiments/multimodal_attribution/concatenate.py`
- `experiments/multimodal_attribution/embeddings.py` (`LCO-Embedding-Omni-3B`)

No CLIP/WavLM frame mean-pool. No cross-modality alignment beyond per-modality PCA + horizontal concat (same spirit as shift_bench AlignHead→concat time windows).

## Pipeline

```
MSR-VTT captions (+ optional mp4/wav)
        │
        ├─ unique video_id (collapse multi-caption)
        │
        ├─ text  → caption string          ─┐
        ├─ audio → wav path | "[audio] …"  ─┼─ LCO-Embedding-Omni-3B  (one encode / modality)
        └─ video → mp4 path | text proxy   ─┘
                │
                ▼
        per-modality emb cache (.npz)
                │
                ├─ StandardScaler + PCA band
                │     top30     = PC 1–30
                │     pc31_100  = PC 31–100
                │
                └─ hstack [text | audio | video]   ← record modality_indices
                        │
                        ▼
                   Digested X, y, T
                   (ref train vs new test, ~500 / batch)
```

## Steps (concrete)

1. **Load** `friedrichor/MSR-VTT` (`train_9k` / `test_1k`); keep first caption per `video_id`.
2. **Resolve** video under `video_root` (`TrainValVideo/`, `TestVideo/`, …); `ffmpeg` → mono 16 kHz wav when present.
3. **Encode with LCO Omni** (shared space, once per modality):
   - text: caption
   - audio: `{"audio": wav}` or tagged text proxy if missing
   - video: `{"video": mp4, "text": caption}` or tagged text proxy
4. **Cache** `msrvtt_{split}_{modality}_LCO_Omni_3B.npz`.
5. **Digest**: per-modality PCA band → `concatenate_modalities` / `concatenate_modalities_pooled` → `stack_outcome` → `(X_ref, X_new, y, T)`.
6. **Test**: `modality_indices` feed block leave-out PO / Shapley / RF·DR·MMD (shift_bench `interactions` or multimodal `perm_methods`).

## Commands

```bash
# extract (needs LCO Omni weights + optional mp4 root)
python -m experiments.multimodal_attribution.prepare_msrvtt_lco \
  --max-rows 700 --batch-size 4 \
  --video-root experiments/multimodal_attribution/data/MSRVTT \
  --out-dir experiments/multimodal_attribution/results/msrvtt_lco_cache

# offline smoke (synthetic text/audio/video blocks, same concat digestion)
# build_msrvtt_style_batches(n_per_batch=500, band="top30"|"pc31_100")
```

## Why this digestion

Mean-pooling CLIP frames / WavLM windows at the video level collapses the cross-modal structure that block leave-out PO interaction is meant to read. LCO Omni encodes each modality once in one space; PCA band + concat keeps blocks named for CF/PO leave-out.
