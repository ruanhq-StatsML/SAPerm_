# ModaSense: short writeup + BibTeX

## Paragraph

On ModaSense/emotion—a 500-example multimodal set with designed pairwise label conflict across text, image, and audio—the shift story is unusually readable. Digestion builds $X=[M_{\mathrm{text}},M_{\mathrm{image}},M_{\mathrm{audio}}]$ from portable meta blocks (VADER/spaCy, OpenCV, WAV stats). Routing a batch gap along one modality then reading RFPerm, CFPerm, MMD share $\pi_m$, and $\rho_{\mathrm{PO}}$ together yields a clean trichotomy: the text-sentiment window closes the chain ($\pi_{\mathrm{text}}{=}0.91$, RF${=}1$, CF$_{\mathrm{text}}{=}1$, VADER subset, $\rho{=}1.95$), so detection and localization name the copy channel; the image-brightness window is a silent absorb ($\pi_{\mathrm{image}}{=}0.95$, $\rho{=}0.62$, RF/CF near $\alpha$), so covariate mass moves without a rejectable alarm; the audio-energy window stays MMD-only under the current budget. The interpretation is operational rather than anonymous: when the chain closes, CFPerm points to executable knobs (e.g.\ `vader_neg`/`compound` vs `img_brightness`/`contrast`), not PCA dims—exactly the sense in which the multimodal meta layout remains interpretable even when feature quality is only “good enough.”

## BibTeX

```bibtex
@misc{modasense_emotion_2026,
  author       = {{Tri-PvP}},
  title        = {{ModaSense/emotion}: Multimodal Emotion with Cross-Modal Label Conflict},
  year         = {2026},
  howpublished = {Hugging Face Datasets},
  url          = {https://huggingface.co/datasets/ModaSense/emotion},
  note         = {500 test examples; per-modality emotion labels (text/image/audio) with designed pairwise conflict. Hub id also resolves as Tri-PvP/emotion.}
}
```
