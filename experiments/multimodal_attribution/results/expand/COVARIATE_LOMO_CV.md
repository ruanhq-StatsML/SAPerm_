# Covariate × LOMO cross-validation (running)

**Goal.** Full-modality covariate (`backtrans` / `inject` on text, audio, video) × singles/pairs/TAV power, then LOMO / dominant-modality read as a cross-check of the concept-drift interaction story.

| Under | Expectation |
|---|---|
| Concept drift | `vs_best`<0, $I_{\mathrm{cal}}^{(3)}<0$ (interaction not super-additive) |
| Covariate on modality $m$ | Power concentrates on combos containing $m$; LOMO drop of $m$ hurts most |

Caliber: $n=500$, $B\to120$, $n_{\mathrm{perm}}=150$, CF/DR/RF, all cores.

Live: `tmux mm-expand-cov` · `expand_shift_board --phases covariate --n-jobs 4`.
