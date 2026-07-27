import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../plugins/akasha/mobile_ui.js", import.meta.url),
  "utf8",
);
const styles = await readFile(
  new URL("../plugins/akasha/mobile_ui.css", import.meta.url),
  "utf8",
);
const module = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

test("Akasha contributes current-turn recall and a mobile Inspector", () => {
  assert.equal(
    typeof module.default.slots["turn.before_reasoning"].mount,
    "function",
  );
  assert.deepEqual(
    Object.keys(module.default.slots),
    ["turn.before_reasoning"],
  );
  assert.equal(typeof module.default.dashboard.mount, "function");
  assert.match(source, /context\.query\(\s*"recall\.current"/);
  assert.match(source, /context\.query\("inspector\.recent"\)/);
  assert.match(source, /context\.query\("inspector\.detail"/);
});

test("mobile UI keeps graph out and uses restrained interaction styles", () => {
  assert.doesNotMatch(source, /akasha-graph|graph\.(global|query|rebuild)/);
  assert.doesNotMatch(styles, /linear-gradient|radial-gradient|backdrop-filter/);
  assert.doesNotMatch(styles, /transition:\s*all|transition-property:\s*all/);
  assert.match(styles, /color-mix\(in oklch/);
  assert.match(styles, /min-height:\s*44px/);
  assert.match(styles, /scale:\s*0\.96/);
});

test("recall lanes use distinct Material tonal semantics", () => {
  assert.match(source, /left,\s*"precise"/);
  assert.match(source, /right,\s*"completion"/);
  assert.match(styles, /--akasha-mobile-precise:\s*var\(--m-primary\)/);
  assert.match(styles, /--akasha-mobile-completion:/);
  assert.match(styles, /var\(--m-trace,\s*oklch\(0\.56 0\.18 300\)\)/);
  assert.match(styles, /grid-template-columns:\s*4px minmax\(0, 1fr\) auto/);
});
