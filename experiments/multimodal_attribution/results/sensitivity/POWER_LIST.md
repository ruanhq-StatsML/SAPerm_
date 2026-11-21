# POWER_LIST — CF / DR / RF × modality hierarchy

_α=0.05; regenerated each summarize_

Read left→right: **single a → single b → pair → TAV**, then PO signed.

## interaction_shift (planted pair)

### cfperm

| pair | λ | pow_a | pow_b | pow_pair | pow_TAV | lift=pair−max₁ | mean_I | matched signed-pow | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.8 | 0.75 | 0.00 | 0.54 | 0.62 | -0.21 | +6.02 | 0.00 | 24 |

### drperm

| pair | λ | pow_a | pow_b | pow_pair | pow_TAV | lift=pair−max₁ | mean_I | matched signed-pow | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.8 | 0.17 | 0.00 | 0.00 | 0.00 | -0.17 | +6.02 | 0.00 | 24 |

### rfperm

| pair | λ | pow_a | pow_b | pow_pair | pow_TAV | lift=pair−max₁ | mean_I | matched signed-pow | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_audio | -1.8 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | +6.02 | 0.00 | 24 |

## All cells (progress skeleton)

| scenario | dataset | combo | method | power | n |
|---|---|---|---|---:|---:|
| interaction_shift | sens_interact_text_audio_neg1p8 | audio | cfperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | audio | drperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | audio | rfperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text | cfperm | 0.75 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text | drperm | 0.17 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text | rfperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio | cfperm | 0.54 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio | drperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio | rfperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio_video | cfperm | 0.62 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio_video | drperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_neg1p8 | text_audio_video | rfperm | 0.00 | 24 |
| interaction_shift | sens_interact_text_audio_pos1p8 | text_audio | cfperm | 0.00 | 1 |
| interaction_shift | sens_interact_text_audio_pos1p8 | text_audio | drperm | 0.00 | 1 |
| interaction_shift | sens_interact_text_audio_pos1p8 | text_audio | rfperm | 0.00 | 1 |

