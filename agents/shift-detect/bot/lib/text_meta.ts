import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const pyRoot = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../python",
);
const script = join(pyRoot, "meta_features.py");

export type TextMetaAttr = {
  method: string;
  packages: string[];
  fixture?: string;
  description?: string;
  ranked: Array<{ name: string; score: number; share: number; rank: number }>;
  featureRanked: Array<{
    name: string;
    score: number;
    share: number;
    rank: number;
  }>;
  top: string | null;
  topHit: boolean | null;
  groundTruth: string[];
  meanRef?: Record<string, number>;
  meanNew?: Record<string, number>;
  error?: string;
};

function runPython(args: string[], stdin?: string): TextMetaAttr {
  const r = spawnSync("python3", [script, ...args], {
    encoding: "utf8",
    input: stdin,
    maxBuffer: 8 * 1024 * 1024,
  });
  if (r.status !== 0) {
    const err = (r.stderr || r.stdout || "python meta_features failed").trim();
    return {
      method: "meta_feature_spacy_vader",
      packages: ["spacy:en_core_web_sm", "vaderSentiment"],
      ranked: [],
      featureRanked: [],
      top: null,
      topHit: null,
      groundTruth: [],
      error: err,
    };
  }
  return JSON.parse(r.stdout) as TextMetaAttr;
}

export function listTextMetaFixtures(): Record<
  string,
  { description: string; groundTruth: string[] }
> {
  const r = spawnSync("python3", [script, "list-fixtures"], {
    encoding: "utf8",
  });
  if (r.status !== 0) {
    throw new Error((r.stderr || r.stdout || "list-fixtures failed").trim());
  }
  return JSON.parse(r.stdout);
}

export function attributeTextMetaFixture(
  name: "sentiment_shift" | "entity_shift" | "style_shift",
): TextMetaAttr {
  return runPython(["fixture", "--name", name]);
}

export function attributeTextWindows(input: {
  ref: string[];
  new: string[];
  groundTruth?: string[];
}): TextMetaAttr {
  return runPython(
    ["attribute", "--input", "-"],
    JSON.stringify({
      ref: input.ref,
      new: input.new,
      groundTruth: input.groundTruth ?? [],
    }),
  );
}
