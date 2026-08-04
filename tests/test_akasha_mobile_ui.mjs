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

test("active recall does not cache a temporary empty projection", () => {
  assert.match(source, /activeMessage \? "none" : "immutable"/);
  assert.match(source, /if \(result\.pending === true\)[\s\S]*?continue;/);
  assert.match(source, /记忆生成中…/);
});

test("recall entries survive the Akasha user field migration", () => {
  assert.match(
    source,
    /item\.user_preview \|\| item\.user_text \|\| "（空消息）"/,
  );
});

test("mobile UI keeps graph out and uses restrained interaction styles", () => {
  assert.doesNotMatch(source, /akasha-graph|graph\.(global|query|rebuild)/);
  assert.doesNotMatch(source, /\.slice\(/);
  assert.doesNotMatch(styles, /linear-gradient|radial-gradient|backdrop-filter/);
  assert.doesNotMatch(styles, /transition:\s*all|transition-property:\s*all/);
  assert.match(styles, /color-mix\(in oklch/);
  assert.match(styles, /min-height:\s*44px/);
  assert.match(styles, /scale:\s*0\.96/);
  assert.match(styles, /content-visibility:\s*auto/);
  assert.match(styles, /contain-intrinsic-block-size:\s*auto 94px/);
});

test("recall lanes use distinct shared-theme tonal semantics", () => {
  assert.match(source, /left,\s*"precise"/);
  assert.match(source, /right,\s*"completion"/);
  assert.match(styles, /--akasha-mobile-precise:\s*var\(--ak-color-action-primary\)/);
  assert.match(styles, /--akasha-mobile-completion:\s*var\(--ak-color-status-trace\)/);
  assert.match(styles, /var\(--akasha-mobile-lane\) 10%/);
  assert.match(styles, /var\(--akasha-mobile-lane\) 28%/);
  assert.doesNotMatch(styles, /--m-primary|--m-trace/);
  assert.match(styles, /grid-template-columns:\s*4px minmax\(0, 1fr\) auto/);
});
