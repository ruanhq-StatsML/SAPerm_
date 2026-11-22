# Multimodal thesis — ModaSense/emotion

Thesis: cross-modality meta blocks give power + named localization;
$(\Delta$Power, $\rho_{PO})$ is interaction evidence.

B=80, N_perm=100, packages=spaCy+VADER+OpenCV.

| scenario | text RF/DR | MM RF/DR | ρ_PO | CF text | CF image | signature |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| image_brightness_shift | 0.00/0.00 | 0.05/0.00 | 0.62 | 0.00 | 0.01 | absorb_interaction |
| null_shift | 0.01/0.05 | 0.01/0.06 | 2.14 | 0.03 | 0.06 | null |
| text_sentiment_shift | 1.00/0.28 | 1.00/0.64 | 1.95 | 1.00 | 0.54 | text_localize |
