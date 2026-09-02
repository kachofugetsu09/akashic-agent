import assert from "node:assert/strict";
import test from "node:test";
import { compatibleEffort, groupModelRuntimes } from "./model-capsule-data.ts";

const runtime = (id, sourceName, efforts = ["low", "medium"]) => ({
  id, sourceName, supportedReasoningEfforts: efforts, reasoningEffort: "medium",
  provider: "fixture", model: id, sourceId: sourceName, roles: ["default"],
});

test("model groups retain stable global indexes without render-time rescans", () => {
  const groups = groupModelRuntimes([runtime("a", "one"), runtime("b", "two"), runtime("c", "one")]);
  assert.deepEqual(groups.map(([source, items]) => [source, items.map(({ index }) => index)]), [
    ["one", [0, 2]], ["two", [1]],
  ]);
});

test("model effort selection preserves compatible choice and owns fallback order", () => {
  assert.equal(compatibleEffort(runtime("a", "one"), "low"), "low");
  assert.equal(compatibleEffort(runtime("a", "one"), "unsupported"), "medium");
  assert.equal(compatibleEffort(runtime("a", "one", ["high"]), ""), "high");
});
