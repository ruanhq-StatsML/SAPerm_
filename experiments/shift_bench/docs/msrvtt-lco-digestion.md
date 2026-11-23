# MSR-VTT / LCO digestion

```
video_id (unique)
  → LCO Omni encode once each: text / audio / video
  → PCA band per modality (top30 | pc31–100)
  → hstack [text | audio | video]   # modality_indices
  → Digested (X, y, T)
```

No CLIP/WavLM mean-pool. No cross-modal fusion beyond concat.

```bash
python -m experiments.multimodal_attribution.prepare_msrvtt_lco --max-rows 700
```

Code: `prepare_msrvtt_lco` → `concatenate_modalities`  
(`cursor/multimodal-higher-order-interaction-e93c`)
