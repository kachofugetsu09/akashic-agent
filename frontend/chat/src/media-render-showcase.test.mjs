import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("./media-render-showcase.tsx", import.meta.url), "utf8");
const styles = readFileSync(new URL("./media-render-showcase.css", import.meta.url), "utf8");
const main = readFileSync(new URL("./main.tsx", import.meta.url), "utf8");

test("media render showcase compares now / memoh / waku / propose", () => {
  assert.match(source, /preview=media-render/);
  assert.match(source, /id: "now"/);
  assert.match(source, /id: "memoh"/);
  assert.match(source, /id: "waku"/);
  assert.match(source, /id: "propose"/);
  assert.match(source, /mrs-content-image/);
  assert.match(source, /mrs-chip/);
  assert.match(source, /mrs-code/);
  assert.match(main, /preview === "media-render"/);
  assert.match(styles, /\.mrs-content-image/);
  assert.match(styles, /max-height:\s*20rem/);
  assert.doesNotMatch(styles, /animate-pulse|glow|gradient-clip/);
});
