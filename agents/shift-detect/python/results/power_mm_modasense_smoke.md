# Multimodal thesis — ModaSense/emotion

Thesis: cross-modality meta blocks give power + named localization;
$(\Delta$Power, $\rho_{PO})$ is interaction evidence.

B=4, N_perm=30, packages=spaCy+VADER+OpenCV.

| scenario | text RF/DR | MM RF/DR | ρ_PO | CF text | CF image | signature |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| image_brightness_shift | 0.00/0.00 | 0.00/0.00 | 0.48 | 0.00 | 0.00 | absorb_interaction |
| null_shift | 0.00/0.00 | 0.00/0.00 | 1.30 | 0.00 | 0.00 | null |
| text_sentiment_shift | 1.00/0.00 | 1.00/0.00 | 2.05 | 1.00 | 1.00 | mixed |
