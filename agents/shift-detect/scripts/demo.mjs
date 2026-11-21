#!/usr/bin/env node
/**
 * Run the offline demos that make attribution obvious.
 * Usage: node scripts/demo.mjs
 */
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function call(tool, input) {
  const r = spawnSync(
    "npx",
    [
      "--yes",
      "@grok/agent-sdk@0.7.2",
      "tool",
      "call",
      tool,
      "--dir",
      root,
      "--json",
      "--input",
      JSON.stringify(input),
    ],
    { encoding: "utf8" },
  );
  if (r.status !== 0) {
    console.error(r.stderr || r.stdout);
    process.exit(r.status ?? 1);
  }
  const raw = r.stdout.trim();
  const start = raw.indexOf("{");
  return JSON.parse(raw.slice(start)).result;
}

function section(title) {
  console.log("\n" + "═".repeat(64));
  console.log(title);
  console.log("═".repeat(64));
}

section("1) Covariate shift — pathway attribution");
const cov = call("score_report", { fixture: "covariate_shift", mode: "auto" });
console.log(cov.glance.headline);
console.log(cov.glance.board);
console.log("action:", cov.recommendation.nextAction);

section("2) Concept drift — LOCO drivers");
const concept = call("score_report", { fixture: "concept_drift", mode: "auto" });
console.log(concept.glance.headline);
console.log(concept.glance.board);
console.log("action:", concept.recommendation.nextAction);

section("3) Mixed shift — block then concept");
const mixed = call("score_report", { fixture: "mixed_shift", mode: "pathway" });
console.log(mixed.glance.headline);
console.log(mixed.glance.board);

section("4) Drift Eval Gate");
const driftGate = call("run_fixture_suite", {});
console.log("metrics:", driftGate.metrics);
console.log(driftGate.gate.summary);

section("5) Trust eval — weak VL behavior");
const exp = call("explain_llm_behavior", {
  modelId: "weak-vl-v0",
  caseId: "vl_ocr_sign_v1",
});
console.log(exp.glance.headline);
for (const c of exp.glance.cards) console.log(`- ${c.label}: ${c.value}`);

section("6) Trust Eval Gate (candidate vs base)");
const trust = call("run_trust_suite", {});
console.log("means:", trust.modelMeans);
console.log(trust.gate.summary);
console.log(
  "blocking axes:",
  trust.alignment.blockingAxes.join(", ") || "(none)",
);

section("7) Text meta-features — spaCy + VADER");
for (const name of ["sentiment_shift", "entity_shift", "style_shift"]) {
  const meta = call("attribute_text_meta", { fixture: name });
  console.log("\n" + meta.glance.headline);
  console.log(meta.glance.board);
  console.log("action:", meta.recommendation.nextAction);
}

console.log("\nDone.\n");
