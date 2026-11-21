/** Method catalog mapping agent tools ↔ ruanhq-StatsML research stack. */

export type CatalogEntry = {
  id: string;
  family: "detect" | "attribute" | "monitor" | "select";
  agentMethod: string;
  researchRepos: string[];
  useWhen: string;
  caveat: string;
};

export const METHOD_CATALOG: CatalogEntry[] = [
  {
    id: "mmd",
    family: "detect",
    agentMethod: "mmd",
    researchRepos: ["SAPerm/kmmd", "Gretton KMMD"],
    useWhen: "Need a kernel two-sample signal on covariates or embeddings.",
    caveat: "Bandwidth sensitive; weaker on pure concept drift.",
  },
  {
    id: "c2st",
    family: "detect",
    agentMethod: "c2st",
    researchRepos: ["SAPerm/c2st", "Lopez-Paz C2ST"],
    useWhen: "High-dim embeddings where a classifier separates windows.",
    caveat: "Needs enough samples for stable CV accuracy.",
  },
  {
    id: "mean_diff",
    family: "detect",
    agentMethod: "mean_diff",
    researchRepos: ["Bai96", "Chen-Qin", "SKK"],
    useWhen: "Quick location-shift screen on tabular features.",
    caveat: "Misses same-mean covariance / concept changes.",
  },
  {
    id: "po_risk",
    family: "detect",
    agentMethod: "po_risk",
    researchRepos: [
      "OnlineMonitoring_Out_of_Distribution",
      "DRPerm",
      "RRPerm",
    ],
    useWhen: "Suspect concept drift; have outcomes on both windows.",
    caveat: "Requires labels/outcomes; nuisance models matter.",
  },
  {
    id: "delta_cf",
    family: "detect",
    agentMethod: "delta_cf",
    researchRepos: [
      "experiments/primary_driver_10site",
      "OnlineRFPerm",
    ],
    useWhen: "Rank sites/windows by skill-gated excess risk.",
    caveat: "Designed for concept primary-driver ranking.",
  },
  {
    id: "pathway",
    family: "attribute",
    agentMethod: "pathway_counterfactual",
    researchRepos: [
      "tabular-drift-attribution",
      "FSDS-FeatureSelection_for_DistributionShift",
    ],
    useWhen: "Need block-level 'who drifted' with counterfactual rewrite.",
    caveat: "Neighbor stub generator; swap in CTGAN/TabFORGE/DiffLTM later.",
  },
  {
    id: "loco",
    family: "attribute",
    agentMethod: "loco_po_risk",
    researchRepos: ["CFPerm loco", "OOD_variable_importance"],
    useWhen: "Attribute concept drift to features via leave-one-out risk drop.",
    caveat: "Computationally heavier; noisy at small n.",
  },
  {
    id: "rfperm",
    family: "detect",
    agentMethod: "external:RFPerm",
    researchRepos: ["RFPerm", "XGBPerm", "OnlineRFPerm"],
    useWhen: "Production-grade permute-then-refit tests on joint (X,Y).",
    caveat: "Full RFPerm lives in sibling repos; agent ships a portable subset.",
  },
  {
    id: "cfperm",
    family: "attribute",
    agentMethod: "external:CFPerm",
    researchRepos: ["CFPerm_loco", "CFPerm_PermuCATE", "CFPerm_grf"],
    useWhen: "Causal VIMP attribution after a shift is detected.",
    caveat: "Wire as Python sidecar when deep attribution is required.",
  },
];

export function listCatalog(family?: CatalogEntry["family"]): CatalogEntry[] {
  if (!family) return METHOD_CATALOG;
  return METHOD_CATALOG.filter((e) => e.family === family);
}
