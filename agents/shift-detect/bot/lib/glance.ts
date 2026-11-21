/** One-glance cards — keep attribution visually obvious. */

export type ShareRow = { name: string; share: number; score?: number };

function bar(share: number, width = 20): string {
  const n = Math.max(0, Math.min(width, Math.round(share * width)));
  return "█".repeat(n) + "░".repeat(width - n);
}

export function shareBoard(rows: ShareRow[], title = "Attribution"): string {
  const top = rows.slice(0, 5);
  const lines = top.map((r, i) => {
    const pct = `${(r.share * 100).toFixed(1)}%`.padStart(6);
    const rank = `#${i + 1}`;
    return `${rank} ${r.name.padEnd(12)} ${bar(r.share)} ${pct}`;
  });
  return [`【${title}】`, ...lines].join("\n");
}

export function shiftGlance(input: {
  fixture: string;
  shiftType: string;
  confidence: number;
  top: string | null;
  topHit: boolean | null;
  rows: ShareRow[];
  action: string;
}): {
  headline: string;
  board: string;
  cards: Array<{ label: string; value: string }>;
} {
  const hit =
    input.topHit === null ? "n/a" : input.topHit ? "HIT ✓" : "MISS ✗";
  const headline = [
    `${input.fixture}`,
    `type=${input.shiftType}`,
    `driver=${input.top ?? "—"}`,
    `gt=${hit}`,
  ].join(" · ");

  return {
    headline,
    board: shareBoard(input.rows, "谁在漂 / Who drifted"),
    cards: [
      { label: "Shift type", value: input.shiftType },
      { label: "Confidence", value: input.confidence.toFixed(2) },
      { label: "Top driver", value: input.top ?? "—" },
      { label: "Ground-truth hit", value: hit },
      { label: "Next action", value: input.action },
    ],
  };
}

export function trustGlance(input: {
  modelId: string;
  caseId: string;
  primaryFactor: string;
  overall?: number;
  suggestedFix: string;
  weakAxes?: string[];
}): {
  headline: string;
  cards: Array<{ label: string; value: string }>;
} {
  return {
    headline: `${input.modelId} @ ${input.caseId} → ${input.primaryFactor}`,
    cards: [
      { label: "Primary factor", value: input.primaryFactor },
      {
        label: "Weak axes",
        value: (input.weakAxes ?? []).join(", ") || "—",
      },
      {
        label: "Overall",
        value:
          input.overall === undefined ? "—" : input.overall.toFixed(3),
      },
      { label: "Fix", value: input.suggestedFix },
    ],
  };
}
