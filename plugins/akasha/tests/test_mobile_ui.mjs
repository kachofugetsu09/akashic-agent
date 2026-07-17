import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../mobile_ui.js", import.meta.url), "utf8");
const styles = await readFile(new URL("../mobile_ui.css", import.meta.url), "utf8");
const panel = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("Akasha keeps message recall and adds a task-first dashboard", () => {
  assert.equal(typeof panel.default.slots["turn.before_reasoning"].mount, "function");
  assert.equal(panel.default.navigation.label, "Akasha");
  assert.match(panel.default.navigation.description, /实际召回/);
  assert.equal(typeof panel.default.dashboard.mount, "function");
});

test("dashboard uses existing read-only inspector RPCs and semantic recall lanes", () => {
  assert.match(source, /context\.request\("inspector\.recent"\)/);
  assert.match(source, /context\.request\("inspector\.detail"/);
  assert.match(source, /本轮问题/);
  assert.match(source, /左脑 · 精确回忆/);
  assert.match(source, /右脑 · 联想记忆/);
  assert.match(styles, /--akasha-left:/);
  assert.match(styles, /--akasha-right:/);
  assert.match(styles, /\.akasha-inspector-state button,[\s\S]*min-height: 48px/);
  assert.doesNotMatch(styles, /linear-gradient|box-shadow/);
});

test("detail expansion preserves its parent row and reduced-motion behavior", () => {
  assert.match(styles, /\.akasha-inspection__detail[\s\S]*grid-template-rows: 0fr/);
  assert.match(styles, /\.akasha-inspection\.is-expanded \.akasha-inspection__detail[\s\S]*grid-template-rows: 1fr/);
  assert.match(styles, /\.akasha-inspector-memory__copy strong[\s\S]*overflow-wrap: anywhere/);
  assert.match(styles, /prefers-reduced-motion: reduce/);
});
