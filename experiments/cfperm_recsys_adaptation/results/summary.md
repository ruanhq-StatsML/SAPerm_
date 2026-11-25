# Synthetic CFPerm update-selection smoke

Existing CFPerm and RFPerm sources were not modified. CFPerm0 and CFPerm1 are not separate exports in the inspected entry points, so this file does not report those rejection rates.

## Run

- dataset: synthetic
- quick: True
- seeds: [202601, 202602, 202603]
- permutations: 99
- label budgets: [100]
- sizes: {'source_train': 2000, 'source_diagnostic': 500, 'target_diagnostic': 500, 'target_adaptation': 500, 'target_validation': 250, 'source_test': 1000, 'target_test': 1000}
- propensity clip: [0.05, 0.95]
- source replay: off
- adaptation protocol: matched epochs (20), full-batch Adam
- PO outcome and tau model: rf (FSDS forest, 150 trees, when rf)

PO-distance is mean(tau(X)^2) on held-out overlap rows. Selection uses the task-label PO-distance. The squared-loss PO-distance is reported beside it and is not a second selector. `po_squared_prediction_error` is the installed residual mean((psi - tau)^2) from `experiments/deep_active_sampling/ood_vimp.py`.

## Sample sizes and label rates

```
  quantity            subset       value
    domain source_diagnostic    0.000000
    domain       source_test    0.000000
    domain      source_train    0.000000
    domain target_adaptation    1.000000
    domain target_diagnostic    1.000000
    domain       target_test    1.000000
    domain target_validation    1.000000
label_rate source_diagnostic    0.484889
label_rate       source_test    0.488556
label_rate      source_train    0.486500
label_rate target_adaptation    0.507444
label_rate target_diagnostic    0.508222
label_rate       target_test    0.510667
label_rate target_validation    0.500889
         n source_diagnostic  500.000000
         n       source_test 1000.000000
         n      source_train 2000.000000
         n target_adaptation  500.000000
         n target_diagnostic  500.000000
         n       target_test 1000.000000
         n target_validation  250.000000
```

## Mean target-test Brier by policy

```
        scenario  cfperm_guided  cfperm_interaction  cfperm_localization  exhaustive_validation  mmd_only  periodic_full  predictive_risk_only  random_module
benign_covariate       0.205822            0.210962             0.207850               0.199375  0.211456       0.276365              0.190663       0.197685
     cross_modal       0.250909            0.248715             0.258629               0.240422  0.252872       0.296166              0.247428       0.241585
       localized       0.225574            0.225574             0.225574               0.213501  0.225574       0.294254              0.230610       0.213107
           mixed       0.242497            0.244383             0.257882               0.231807  0.243661       0.277954              0.247102       0.246345
modality_concept       0.259611            0.245711             0.259611               0.228193  0.326348       0.289939              0.259611       0.270860
            null       0.196314            0.196314             0.196314               0.197461  0.196314       0.295626              0.203779       0.206849
```

## Mean over scenarios and seeds

```
               policy  target_brier  source_brier  target_gain  source_forgetting
        cfperm_guided      0.230121      0.214129     0.009012           0.015970
   cfperm_interaction      0.228610      0.216389     0.010523           0.018230
  cfperm_localization      0.234310      0.218031     0.004823           0.019873
exhaustive_validation      0.218460      0.217932     0.020673           0.019773
             mmd_only      0.242704      0.205784    -0.003571           0.007625
        periodic_full      0.288384      0.285074    -0.049251           0.086915
 predictive_risk_only      0.229865      0.221434     0.009268           0.023275
        random_module      0.229405      0.221438     0.009728           0.023279
```

## Primary comparison (lower target Brier is better)

- cfperm_guided: 0.230121
- mmd_only: 0.242704
- predictive_risk_only: 0.229865
- cfperm_guided lower than mmd_only: True
- cfperm_guided lower than predictive_risk_only: False

## Share of runs whose selected arm matches the planted-action rule

```
               policy  hit_rate
        cfperm_guided  0.500000
   cfperm_interaction  0.500000
  cfperm_localization  0.388889
exhaustive_validation  0.388889
             mmd_only  0.500000
        periodic_full  0.166667
 predictive_risk_only  0.666667
        random_module  0.388889
```

## Selected arms

```
        scenario                policy           arm  n
benign_covariate         cfperm_guided     no_update  1
benign_covariate         cfperm_guided   visual_only  2
benign_covariate    cfperm_interaction   fusion_only  1
benign_covariate    cfperm_interaction     no_update  1
benign_covariate    cfperm_interaction       pair_TV  1
benign_covariate   cfperm_localization     no_update  1
benign_covariate   cfperm_localization targeted_data  2
benign_covariate exhaustive_validation    audio_only  1
benign_covariate exhaustive_validation     no_update  2
benign_covariate              mmd_only   visual_only  3
benign_covariate         periodic_full          full  3
benign_covariate  predictive_risk_only     no_update  3
benign_covariate         random_module     head_only  2
benign_covariate         random_module     text_only  1
     cross_modal         cfperm_guided     no_update  2
     cross_modal         cfperm_guided   visual_only  1
     cross_modal    cfperm_interaction   fusion_only  1
     cross_modal    cfperm_interaction     no_update  2
     cross_modal   cfperm_localization     no_update  2
     cross_modal   cfperm_localization targeted_data  1
     cross_modal exhaustive_validation   fusion_only  1
     cross_modal exhaustive_validation     head_only  1
     cross_modal exhaustive_validation   visual_only  1
     cross_modal              mmd_only     no_update  2
     cross_modal              mmd_only     text_only  1
     cross_modal         periodic_full          full  3
     cross_modal  predictive_risk_only     no_update  1
     cross_modal  predictive_risk_only     text_only  1
     cross_modal  predictive_risk_only   visual_only  1
     cross_modal         random_module   fusion_only  1
     cross_modal         random_module     head_only  2
       localized         cfperm_guided     no_update  3
       localized    cfperm_interaction     no_update  3
       localized   cfperm_localization     no_update  3
       localized exhaustive_validation    audio_only  1
       localized exhaustive_validation     head_only  1
       localized exhaustive_validation     text_only  1
       localized              mmd_only     no_update  3
       localized         periodic_full          full  3
       localized  predictive_risk_only    audio_only  2
       localized  predictive_risk_only   visual_only  1
       localized         random_module   fusion_only  2
       localized         random_module     head_only  1
           mixed         cfperm_guided     no_update  1
           mixed         cfperm_guided   visual_only  2
           mixed    cfperm_interaction   fusion_only  1
           mixed    cfperm_interaction     no_update  1
           mixed    cfperm_interaction   visual_only  1
           mixed   cfperm_localization     no_update  1
           mixed   cfperm_localization targeted_data  2
           mixed exhaustive_validation     head_only  2
           mixed exhaustive_validation targeted_data  1
           mixed              mmd_only   visual_only  3
           mixed         periodic_full          full  3
           mixed  predictive_risk_only     no_update  1
           mixed  predictive_risk_only   visual_only  2
           mixed         random_module   fusion_only  2
           mixed         random_module     text_only  1
modality_concept         cfperm_guided   visual_only  3
modality_concept    cfperm_interaction   fusion_only  1
modality_concept    cfperm_interaction   visual_only  2
modality_concept   cfperm_localization   visual_only  3
modality_concept exhaustive_validation   fusion_only  3
modality_concept              mmd_only     no_update  3
modality_concept         periodic_full          full  3
modality_concept  predictive_risk_only   visual_only  3
modality_concept         random_module     head_only  2
modality_concept         random_module     text_only  1
            null         cfperm_guided     no_update  3
            null    cfperm_interaction     no_update  3
            null   cfperm_localization     no_update  3
            null exhaustive_validation     head_only  2
            null exhaustive_validation     no_update  1
            null              mmd_only     no_update  3
            null         periodic_full          full  3
            null  predictive_risk_only   fusion_only  1
            null  predictive_risk_only     no_update  2
            null         random_module     head_only  2
            null         random_module     text_only  1
```

## Frozen-parameter check (1 means every run kept frozen weights)

```
          arm  frozen_unchanged
   audio_only               1.0
         full               1.0
  fusion_only               1.0
    head_only               1.0
    no_update               1.0
      pair_AV               1.0
      pair_TA               1.0
      pair_TV               1.0
targeted_data               1.0
    text_only               1.0
  visual_only               1.0
```
