---
description: Pick SAPerm / StatsML methods for detect vs attribute vs monitor
---

# Method selection

Call `list_method_catalog` for the live list. Shortcut:

| Goal | First tool | Research sibling |
| --- | --- | --- |
| Any shift on embeddings | `detect_shift` method=mmd/c2st | SAPerm KMMD / C2ST |
| Tabular location shift | `detect_shift` method=mean_diff | Bai/Chen-Qin/SKK |
| Concept / site ranking | `detect_shift` method=delta_cf/po_risk | OnlineMonitoring, primary_driver_10site |
| Block drivers | `attribute_shift` mode=pathway | tabular-drift-attribution |
| Feature drivers of concept | `attribute_shift` mode=loco | CFPerm, OOD_variable_importance |
| Joint (X,Y) permute-refit | catalog external RFPerm | RFPerm / OnlineRFPerm |
| Feature selection under shift | catalog FSDS | FSDS-FeatureSelection_for_DistributionShift |
| Recsys rating monitor + refresh | `experiments/recsys_monitor` | OnlineRFPerm → CFPerm blocks → refresh ticket |

Prefer portable agent tools first. Mention external repos when the user needs production-grade RF/XGB permute tests.
For recommender landing see `experiments/recsys_monitor/ROADMAP.md`.
