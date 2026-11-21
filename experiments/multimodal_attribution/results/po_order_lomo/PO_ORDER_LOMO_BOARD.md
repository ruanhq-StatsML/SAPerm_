# PO-risk 2nd/3rd-order (RF outcome) + LOMO power cross-check

Outcome / τ / ν models: `rf_regressor`.

Definitions (leave-out PO risk $R(S)$ = held-out PO after dropping block set $S$):

- $I^{(2)}(m_1,m_2)=R(\{m_1\})+R(\{m_2\})-R(\{m_1,m_2\})$; $I_{\mathrm{cal}}^{(2)}=I^{(2)}-\mathrm{PO}(\mathrm{full})$.
- $I^{(3)}=R(\{t\})+R(\{a\})+R(\{v\})-R(\{t,a\})-R(\{t,v\})-R(\{a,v\})+R(\{t,a,v\})$.
- LOMO power loss $=\mathrm{Power}(\mathrm{TAV})-\mathrm{Power}(\mathrm{TAV}\setminus\{m\})$.
  Loss $>0$: dropped modality helps the triple. Loss $<0$: over-complete / conflicting (dropping raises power) — cross-check for $I^{(3)}<0$.

## 1. Pairwise $I^{(2)}$ / $I^{(2)}_{\mathrm{cal}}$ (RF)

| dataset | scenario | pair | mean $I^{(2)}$ | mean $I^{(2)}_{\mathrm{cal}}$ | frac $I>0$ | frac $I_{\mathrm{cal}}>0$ | B |
|---|---|---|---:|---:|---:|---:|---:|
| `msrvtt_concept_pc31_100` | concept_drift | audio_video | 99.884 | -1020.427 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_pc31_100` | concept_drift | text_audio | 115.872 | -1004.438 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_pc31_100` | concept_drift | text_video | 113.452 | -1006.858 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_top30` | concept_drift | audio_video | 10.342 | -2.254 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_top30` | concept_drift | text_audio | 10.019 | -2.576 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_top30` | concept_drift | text_video | 9.779 | -2.816 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TA_neg` | interaction_shift | audio_video | 12.394 | -28.024 | 1.00 | 0.08 | 12 |
| `msrvtt_interact_TA_neg` | interaction_shift | text_audio | 9.161 | -31.257 | 1.00 | 0.08 | 12 |
| `msrvtt_interact_TA_neg` | interaction_shift | text_video | 13.854 | -26.564 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TA_pos` | interaction_shift | audio_video | 6.015 | -5.837 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TA_pos` | interaction_shift | text_audio | 3.857 | -7.996 | 1.00 | 0.08 | 12 |
| `msrvtt_interact_TA_pos` | interaction_shift | text_video | 6.528 | -5.325 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TV_neg` | interaction_shift | audio_video | 11.975 | -15.988 | 1.00 | 0.25 | 12 |
| `msrvtt_interact_TV_neg` | interaction_shift | text_audio | 11.435 | -16.528 | 1.00 | 0.08 | 12 |
| `msrvtt_interact_TV_neg` | interaction_shift | text_video | 11.025 | -16.938 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TV_pos` | interaction_shift | audio_video | 8.983 | -13.033 | 1.00 | 0.08 | 12 |
| `msrvtt_interact_TV_pos` | interaction_shift | text_audio | 8.490 | -13.526 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TV_pos` | interaction_shift | text_video | 6.240 | -15.775 | 1.00 | 0.08 | 12 |
| `msrvtt_null_top30` | no_shift | audio_video | 11.951 | -11.724 | 1.00 | 0.00 | 12 |
| `msrvtt_null_top30` | no_shift | text_audio | 13.822 | -9.854 | 1.00 | 0.08 | 12 |
| `msrvtt_null_top30` | no_shift | text_video | 12.759 | -10.917 | 1.00 | 0.00 | 12 |
| `mosei_train_test` | real_train_test | audio_video | 770.941 | -1515.725 | 1.00 | 0.00 | 12 |
| `mosei_train_test` | real_train_test | text_audio | 870.602 | -1416.064 | 1.00 | 0.00 | 12 |
| `mosei_train_test` | real_train_test | text_video | 1008.327 | -1278.339 | 1.00 | 0.00 | 12 |

## 2. Triple $I^{(3)}$ (RF)

| dataset | scenario | mean $I^{(3)}$ | mean $I^{(3)}_{\mathrm{cal}}$ | frac $I^{(3)}>0$ | frac $I^{(3)}_{\mathrm{cal}}>0$ | B |
|---|---|---:|---:|---:|---:|---:|
| `msrvtt_concept_pc31_100` | concept_drift | 110.956 | -778.977 | 1.00 | 0.00 | 12 |
| `msrvtt_concept_top30` | concept_drift | 10.890 | -2.224 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TA_neg` | interaction_shift | 9.824 | -23.000 | 1.00 | 0.17 | 12 |
| `msrvtt_interact_TA_pos` | interaction_shift | 11.127 | -8.087 | 1.00 | 0.17 | 12 |
| `msrvtt_interact_TV_neg` | interaction_shift | 11.825 | -17.518 | 1.00 | 0.00 | 12 |
| `msrvtt_interact_TV_pos` | interaction_shift | 8.384 | -10.911 | 1.00 | 0.00 | 12 |
| `msrvtt_null_top30` | no_shift | 15.386 | -5.773 | 1.00 | 0.08 | 12 |
| `mosei_train_test` | real_train_test | 1266.312 | -1056.018 | 1.00 | 0.00 | 12 |

## 3. LOMO power loss (cross-check)

| dataset | method | dropped | Power(TAV) | Power(LOO) | loss (TAV−LOO) | Δ% | B_TAV | B_LOO |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `msrvtt_concept_pc31_100` | rfperm | audio | 0.573 | 0.693 | -0.120 | -20.9% | 150 | 150 |
| `msrvtt_concept_pc31_100` | rfperm | video | 0.573 | 0.647 | -0.073 | -12.8% | 150 | 150 |
| `msrvtt_concept_pc31_100` | rfperm | vs_best_pair | 0.573 | 0.693 | -0.120 | -20.9% | 150 | 150 |
| `msrvtt_concept_top30` | cfperm | audio | 0.400 | 0.342 | +0.058 | +14.6% | 120 | 120 |
| `msrvtt_concept_top30` | cfperm | video | 0.400 | 0.600 | -0.200 | -50.0% | 120 | 120 |
| `msrvtt_concept_top30` | cfperm | vs_best_pair | 0.400 | 0.600 | -0.200 | -50.0% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | audio | 0.300 | 0.217 | +0.083 | +27.8% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | video | 0.300 | 0.400 | -0.100 | -33.3% | 120 | 120 |
| `msrvtt_concept_top30` | drperm | vs_best_pair | 0.300 | 0.400 | -0.100 | -33.3% | 120 | 120 |
| `msrvtt_concept_top30` | rfperm | audio | 0.933 | 0.987 | -0.053 | -5.7% | 150 | 150 |
| `msrvtt_concept_top30` | rfperm | video | 0.933 | 1.000 | -0.067 | -7.1% | 150 | 150 |
| `msrvtt_concept_top30` | rfperm | vs_best_pair | 0.933 | 1.000 | -0.067 | -7.1% | 150 | 150 |
| `msrvtt_null_top30` | rfperm | audio | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 120 |
| `msrvtt_null_top30` | rfperm | video | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 133 |
| `msrvtt_null_top30` | rfperm | vs_best_pair | 0.000 | 0.000 | +0.000 | +0.0% | 120 | 133 |

**Read rule.** If TAV loses to best pair (`dropped=vs_best_pair`, loss $<0$), triple fusion is inefficient — consistent with $I^{(3)}\le 0$ / over-complete. Per-modality loss $<0$ means that channel is antagonistic or pure noise in the concat.
