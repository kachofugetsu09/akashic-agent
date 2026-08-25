import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("./paper-shell-showcase.tsx", import.meta.url), "utf8");
const styles = readFileSync(new URL("./paper-shell-showcase.css", import.meta.url), "utf8");
const main = readFileSync(new URL("./main.tsx", import.meta.url), "utf8");

test("paper shell showcase covers dual-nav strategies", () => {
  assert.match(source, /preview=paper-shell/);
  assert.match(source, /id: "now"/);
  assert.match(source, /id: "lshape"/);
  assert.match(source, /id: "arena"/);
  assert.match(source, /id: "spokes"/);
  assert.match(source, /LShapeShell/);
  assert.match(source, /ArenaShell/);
  assert.match(source, /SpokesShell/);
  assert.match(main, /preview === "paper-shell"/);
  assert.match(styles, /--pss-parchment:\s*#f5f4ed/);
  assert.doesNotMatch(styles, /box-shadow:\s*0 8px|purple|indigo/);
});
