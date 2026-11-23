# MSR-VTT / LCO digestion (Python)

```
python -m experiments.multimodal_attribution.prepare_msrvtt_lco --max-rows 700
```

```
modalities = {text, audio, video}   # LCO Omni, once each
digest_concat(...)                  # PCA band → hstack
digest_aligned(...)                 # AlignHead → hstack
```

Extract: `experiments/multimodal_attribution/prepare_msrvtt_lco.py`  
Concat: `experiments/shift_bench/digest_mm.py`
