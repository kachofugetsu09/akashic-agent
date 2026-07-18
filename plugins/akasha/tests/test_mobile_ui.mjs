import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../mobile_ui.js", import.meta.url), "utf8");
const styles = await readFile(new URL("../mobile_ui.css", import.meta.url), "utf8");
const panel = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("Akasha only contributes the message recall slot", () => {
  assert.equal(typeof panel.default.slots["turn.before_reasoning"].mount, "function");
  assert.deepEqual(Object.keys(panel.default.slots), ["turn.before_reasoning"]);
  assert.equal("navigation" in panel.default, false);
  assert.equal("dashboard" in panel.default, false);
});

test("message recall stays compact and has no inspector code", () => {
  assert.match(source, /context\.query\("recall\.current"/);
  assert.match(source, /左脑 · 精确回忆/);
  assert.match(source, /右脑 · 联想记忆/);
  assert.doesNotMatch(source, /inspector\.(recent|detail)|akasha-inspector/);
  assert.match(styles, /--akasha-left:/);
  assert.match(styles, /--akasha-right:/);
  assert.doesNotMatch(styles, /akasha-inspector|linear-gradient|box-shadow/);
});
