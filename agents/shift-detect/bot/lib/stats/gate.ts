import type { GateVerdict } from "./types.js";

export type SuiteMetrics = {
  typeI: number;
  powerCovariate: number;
  powerConcept: number;
  attributionTopHit: number;
  typeAccuracy: number;
};

/** Eval Gate: hard thresholds before promoting a method/prompt change. */
export function evalGate(
  metrics: SuiteMetrics,
  thresholds: Partial<SuiteMetrics> = {},
): GateVerdict {
  const bars: SuiteMetrics = {
    typeI: thresholds.typeI ?? 0.1,
    powerCovariate: thresholds.powerCovariate ?? 0.7,
    powerConcept: thresholds.powerConcept ?? 0.6,
    attributionTopHit: thresholds.attributionTopHit ?? 0.7,
    typeAccuracy: thresholds.typeAccuracy ?? 0.6,
  };

  const checks: GateVerdict["checks"] = [
    {
      name: "type_I_control",
      pass: metrics.typeI <= bars.typeI,
      observed: metrics.typeI,
      threshold: bars.typeI,
      direction: "lte",
    },
    {
      name: "power_covariate",
      pass: metrics.powerCovariate >= bars.powerCovariate,
      observed: metrics.powerCovariate,
      threshold: bars.powerCovariate,
      direction: "gte",
    },
    {
      name: "power_concept",
      pass: metrics.powerConcept >= bars.powerConcept,
      observed: metrics.powerConcept,
      threshold: bars.powerConcept,
      direction: "gte",
    },
    {
      name: "attribution_top_hit",
      pass: metrics.attributionTopHit >= bars.attributionTopHit,
      observed: metrics.attributionTopHit,
      threshold: bars.attributionTopHit,
      direction: "gte",
    },
    {
      name: "shift_type_accuracy",
      pass: metrics.typeAccuracy >= bars.typeAccuracy,
      observed: metrics.typeAccuracy,
      threshold: bars.typeAccuracy,
      direction: "gte",
    },
  ];

  const pass = checks.every((c) => c.pass);
  const failed = checks.filter((c) => !c.pass).map((c) => c.name);
  return {
    pass,
    checks,
    summary: pass
      ? "Eval Gate PASS — safe to promote method/prompt change."
      : `Eval Gate FAIL — blocked by: ${failed.join(", ")}.`,
  };
}
