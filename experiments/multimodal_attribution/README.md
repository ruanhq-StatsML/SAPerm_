## Checkpoint / stacking

- Power boards append each finished `(dataset, combo, method, rep)` immediately.
- Re-launch with a larger `--n-rep` to **stack**; finished cells are skipped (CSV deduped on resume).
- LCO embeddings checkpoint **per modality** under `results/emotion_cache/`.

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
  --n 500 --band top30 --n-rep 150 --n-perm 120 --with-po-interact \
  --out experiments/multimodal_attribution/results/power_board.csv
```

### Overnight: 8 datasets × modality combos × n=500

Nested PO interaction is slow — that is expected. Use **B=120** reps
(and 120 permutations). Checkpointed runner — if an earlier pass only did
40 reps, re-launch with ``--n-rep 150`` and it stacks the missing reps:

```bash
python -m experiments.multimodal_attribution.overnight_power \
  --n 500 --n-rep 150 --n-perm 120 --with-po-interact \
  --out-dir experiments/multimodal_attribution/results/overnight
```

Datasets: null (top30 / pc30_100), mean shift (video / audio), interaction
(text×video, text×audio, audio×video, text×video on pc30_100).

Modality combos: `text_audio`, `text_video`, `audio_video`, `text_audio_video`.

Progress: `results/overnight/overnight.log`, raw p-values in `power_raw.csv`,
board in `POWER_BOARD.md`.

### CMU-MOSEI + MELD (PC 31–100, MiniLM, RF outcome)

```bash
python -m experiments.multimodal_attribution.emotion_overnight \
  --n 500 --n-rep 150 --n-perm 120 --band pc30_100 \
  --out-dir experiments/multimodal_attribution/results/emotion_overnight
```

- **MOSEI**: `vintp/CMU-Mosei-text` (sentiment ∈ [-3,3])
- **MELD**: official utterance CSVs (~13k; sentiment / 7 emotions)
- Embeddings: MiniLM text + ASR/prompt channels for audio/video
- Digestion: `pc30_100` (PC 31–100); outcome / tau: **`rf_regressor`**
- B=150; checkpoint stacks if you raise `--n-rep` later

### Concept drift (manual `Y|X`) + CFPerm / RFPerm / DRPerm

No attribution — inject concept drift and compare power (outcome = RF):

```bash
python -m experiments.multimodal_attribution.concept_drift_overnight \
  --n 500 --n-rep 150 --n-perm 150 \
  --out-dir experiments/multimodal_attribution/results/concept_drift
```

Methods: classical TST + **`cfperm` / `rfperm` / `drperm`** (`ruanhq-StatsML/CFPerm`, `RFPerm`).
Outcome = **`rf_regressor`**. CFPerm/DRPerm **`n_perm=150`**. No attribution — power only.

### CMU-MOSEI + MELD (LCO-Embedding-Omni-3B, PC 31–100, RF, B=120)

```bash
python -m experiments.multimodal_attribution.emotion_overnight \
  --n 500 --n-rep 150 --n-perm 120 --band pc31_100 \
  --model-key LCO_Omni_3B --max-rows 800 \
  --out-dir experiments/multimodal_attribution/results/emotion_overnight
```

- Embeddings: `LCO-Embedding/LCO-Embedding-Omni-3B`
- Digestion: PCA **PC 31–100**, modality concat, batch **n=500**
- Outcome / tau / nu: **`rf_regressor`**
- Monte Carlo: **B=150** (stackable via checkpoint)

### Optional FSDS detectors

Digestion stays PCA-per-modality + concat. FSDS is opt-in only:

```bash
python -m experiments.multimodal_attribution.power_benchmark --n 500 --with-fsds
# or overnight:
python -m experiments.multimodal_attribution.overnight_power --n 500 --with-fsds
```

Adds `fsds_mmd`, `fsds_hsic`, `fsds_po`, `fsds_rrisk` (from `experiments/uplift_monitor/fsds_vimp.py` + PermuCATE PO).

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
