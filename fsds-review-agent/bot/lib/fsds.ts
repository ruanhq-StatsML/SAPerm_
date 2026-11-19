/**
 * FSDS attribution shares + Gaussian Thompson incentives.
 *
 * Matches the SAPerm streaming policy shape:
 *   draw_j = N(mu_j, se_j) - lambda * w_j
 * and adds a uniformity bonus so under-pulled arms keep working:
 *   draw_j += gamma / sqrt(n_j + 1)
 */

export type ArmStats = {
  name: string;
  mu: number;
  n: number;
  m2: number;
};

export type AttributionInput = {
  /** Arm / block / contributor ids. */
  names: string[];
  /** Non-negative raw scores (distances, PO-risk drops, effort units). */
  scores: number[];
};

export type AttributionResult = {
  names: string[];
  scores: number[];
  shares: number[];
  ranked: { name: string; score: number; share: number }[];
  top: string | null;
  entropy: number;
  maxShare: number;
  evenness: number;
};

export type IncentiveInput = {
  names: string[];
  /** FSDS shift / contribution shares (sum ~ 1). */
  shares: number[];
  /** Optional running reward stats; defaults to cold start. */
  arms?: ArmStats[];
  lambda?: number;
  /** Uniformity bonus weight. Higher => more even pulls. */
  gamma?: number;
  seed: number;
  /** How many independent draws to summarize selection rates. */
  draws?: number;
};

export type IncentiveResult = {
  names: string[];
  shares: number[];
  lambda: number;
  gamma: number;
  lastDraw: number[];
  selected: string;
  selectedIndex: number;
  selectionRates: number[];
  balance: {
    shareEntropy: number;
    selectionEntropy: number;
    maxSelectionRate: number;
    evenness: number;
    advice: string[];
  };
};

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller normal using a seeded unit generator. */
export function normal(rng: () => number, mean = 0, sd = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + sd * z;
}

export function asShare(raw: number[]): number[] {
  const clipped = raw.map((x) => Math.max(0, Number.isFinite(x) ? x : 0));
  const total = clipped.reduce((a, b) => a + b, 0);
  if (total <= 0) {
    const k = Math.max(clipped.length, 1);
    return clipped.map(() => 1 / k);
  }
  return clipped.map((x) => x / total);
}

export function entropy(p: number[]): number {
  let h = 0;
  for (const x of p) {
    if (x > 0) h -= x * Math.log(x);
  }
  return h;
}

/** 1 = perfectly even shares; 0 = one arm takes everything. */
export function evenness(p: number[]): number {
  const k = p.length;
  if (k <= 1) return 1;
  return entropy(p) / Math.log(k);
}

export function attributeBlocks(input: AttributionInput): AttributionResult {
  const { names, scores } = input;
  if (names.length === 0) {
    throw new Error("names must be non-empty");
  }
  if (names.length !== scores.length) {
    throw new Error("names and scores must have the same length");
  }
  const shares = asShare(scores);
  const ranked = names
    .map((name, i) => ({ name, score: scores[i]!, share: shares[i]! }))
    .sort((a, b) => b.share - a.share || b.score - a.score);
  return {
    names,
    scores: scores.map((x) => Number(x)),
    shares,
    ranked,
    top: ranked[0]?.name ?? null,
    entropy: entropy(shares),
    maxShare: Math.max(...shares, 0),
    evenness: evenness(shares),
  };
}

function seOf(arm: ArmStats): number {
  const count = arm.n;
  const variance = count >= 2 ? arm.m2 / (count - 1) : 1;
  return Math.sqrt(Math.max(variance, 1e-8) / Math.max(count, 1));
}

export function defaultArms(names: string[]): ArmStats[] {
  return names.map((name) => ({ name, mu: 0, n: 0, m2: 0 }));
}

export function drawIncentives(
  arms: ArmStats[],
  shares: number[],
  lambda: number,
  gamma: number,
  seed: number,
): { draws: number[]; selectedIndex: number } {
  const rng = mulberry32(seed);
  const draws = arms.map((arm, j) => {
    const sample = normal(rng, arm.mu, seOf(arm));
    const penalty = lambda * (shares[j] ?? 0);
    const bonus = gamma / Math.sqrt(arm.n + 1);
    return sample - penalty + bonus;
  });
  let selectedIndex = 0;
  for (let i = 1; i < draws.length; i++) {
    if (draws[i]! > draws[selectedIndex]!) selectedIndex = i;
  }
  return { draws, selectedIndex };
}

export function fsdsIncentives(input: IncentiveInput): IncentiveResult {
  const names = input.names;
  if (names.length === 0) throw new Error("names must be non-empty");
  const shares = asShare(input.shares.length === names.length ? input.shares : names.map(() => 1));
  if (input.shares.length !== names.length) {
    throw new Error("shares and names must have the same length");
  }
  const lambda = input.lambda ?? 1;
  const gamma = input.gamma ?? 0.35;
  const arms =
    input.arms && input.arms.length === names.length
      ? input.arms.map((a, i) => ({
          name: names[i]!,
          mu: a.mu,
          n: a.n,
          m2: a.m2,
        }))
      : defaultArms(names);
  const rounds = Math.max(1, input.draws ?? 64);
  const counts = names.map(() => 0);
  let lastDraw: number[] = [];
  let lastSelected = 0;
  for (let r = 0; r < rounds; r++) {
    const { draws, selectedIndex } = drawIncentives(
      arms,
      shares,
      lambda,
      gamma,
      input.seed + r * 9973,
    );
    lastDraw = draws;
    lastSelected = selectedIndex;
    counts[selectedIndex]! += 1;
  }
  const selectionRates = asShare(counts);
  const advice: string[] = [];
  const shareEven = evenness(shares);
  const selEven = evenness(selectionRates);
  if (shareEven < 0.75) {
    advice.push(
      `Attribution is concentrated (evenness=${shareEven.toFixed(2)}). Keep lambda>0 so FSDS dampens the loudest block.`,
    );
  }
  if (selEven < 0.8) {
    advice.push(
      `Selection is still uneven (evenness=${selEven.toFixed(2)}). Raise gamma or lower lambda so lagging arms keep pulling.`,
    );
  } else {
    advice.push(
      `Selection evenness=${selEven.toFixed(2)}: FSDS + uniformity bonus are keeping arms working evenly.`,
    );
  }
  const starved = names.filter((_, i) => (selectionRates[i] ?? 0) < 1 / (2 * names.length));
  if (starved.length) {
    advice.push(`Under-selected arms: ${starved.join(", ")}. Prefer these in the next review cycle.`);
  }

  return {
    names,
    shares,
    lambda,
    gamma,
    lastDraw,
    selected: names[lastSelected]!,
    selectedIndex: lastSelected,
    selectionRates,
    balance: {
      shareEntropy: entropy(shares),
      selectionEntropy: entropy(selectionRates),
      maxSelectionRate: Math.max(...selectionRates, 0),
      evenness: selEven,
      advice,
    },
  };
}

export type BoardReviewInput = {
  attribution: AttributionResult;
  incentives: IncentiveResult;
  context?: string;
};

export type BoardReviewResult = {
  verdict: "balanced" | "needs_rebalance" | "concentrated";
  summary: string;
  findings: string[];
  nextActions: string[];
};

export function reviewBoard(input: BoardReviewInput): BoardReviewResult {
  const { attribution, incentives } = input;
  const findings: string[] = [];
  const nextActions: string[] = [];

  findings.push(
    `Top attributed arm: ${attribution.top ?? "none"} (share=${(attribution.maxShare * 100).toFixed(1)}%).`,
  );
  findings.push(
    `Attribution evenness=${attribution.evenness.toFixed(2)}; selection evenness=${incentives.balance.evenness.toFixed(2)}.`,
  );
  findings.push(
    `Policy lambda=${incentives.lambda}, gamma=${incentives.gamma}; last selected=${incentives.selected}.`,
  );

  let verdict: BoardReviewResult["verdict"] = "balanced";
  if (attribution.maxShare >= 0.55 || incentives.balance.maxSelectionRate >= 0.55) {
    verdict = "concentrated";
    nextActions.push("Increase gamma (uniformity) or keep lambda on the dominant FSDS share.");
    nextActions.push("Ask owners of lagging arms for one concrete deliverable this cycle.");
  } else if (attribution.evenness < 0.8 || incentives.balance.evenness < 0.85) {
    verdict = "needs_rebalance";
    nextActions.push("Re-run fsds_incentives with a slightly higher gamma.");
    nextActions.push("Call out under-selected arms in the review comment.");
  } else {
    nextActions.push("Keep current lambda/gamma; re-audit after the next batch of rewards.");
  }
  nextActions.push(...incentives.balance.advice);

  const summary =
    verdict === "balanced"
      ? "FSDS shares and Thompson incentives look evenly worked."
      : verdict === "needs_rebalance"
        ? "Board needs a rebalance nudge so lagging arms keep contributing."
        : "Board is concentrated; apply stronger FSDS dampening and uniformity bonus.";

  return { verdict, summary, findings, nextActions };
}
