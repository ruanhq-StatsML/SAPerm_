# Power comparison (smoke)

Outcome model: **RandomForest** (n_estimators=80).
B=80, n_doc=80, n_perm=100, α=0.05.

Core (RF): RFPerm, DRPerm, RRPerm, CFPerm, MMD_X/XY, C2ST_X/XY.
Classic two-sample on X (+ classic_c2st_XY): kernel_MMD, chen_qin, adaptive_test_high_dim, bai96_test, sri08_test, skk_test, random_projection_test.

Scenarios: `random_split`, `concept_drift` (Y_new←image_label), modality covariates.

| dataset           | scenario      |   B |   RFPerm_power |   DRPerm_power |   RRPerm_power |   MMD_X_power |   MMD_XY_power |   C2ST_X_power |   C2ST_XY_power |   CF_text_power |   CF_image_power |   CF_audio_power |   kernel_MMD_power |   chen_qin_power |   adaptive_test_high_dim_power |   bai96_test_power |   sri08_test_power |   skk_test_power |   random_projection_test_power |   classic_c2st_XY_power |   rho_po_mean |   po_risk_mean |   power_loss_mean |   r_risk_mean |
|:------------------|:--------------|----:|---------------:|---------------:|---------------:|--------------:|---------------:|---------------:|----------------:|----------------:|-----------------:|-----------------:|-------------------:|-----------------:|-------------------------------:|-------------------:|-------------------:|-----------------:|-------------------------------:|------------------------:|--------------:|---------------:|------------------:|--------------:|
| modasense_emotion | concept_drift |   3 |       1        |       0.333333 |       0.333333 |             0 |              0 |              0 |               0 |               0 |         0.333333 |         0.333333 |                  0 |                1 |                              0 |                  0 |                  0 |                0 |                       0        |                       0 |      0.855473 |     0.0130806  |         0.247031  |   0.00146546  |
| modasense_emotion | random_split  |   3 |       0.333333 |       0        |       0        |             0 |              0 |              0 |               0 |               0 |         0        |         0.333333 |                  0 |                1 |                              0 |                  0 |                  0 |                0 |                       0.666667 |                       0 |      1.9319   |     0.00175327 |         0.0137017 |   1.42844e-05 |
