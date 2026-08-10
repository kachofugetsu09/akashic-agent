import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("./trace-motion-showcase.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("./trace-motion-showcase.css", import.meta.url), "utf8");

test("trace motion showcase contains the hybrid, five originals, and the full repeated sequence", () => {
  assert.equal((source.match(/id: "(?:hybrid|breathe|flow|echo|spring|scan)"/g) ?? []).length, 6);
  assert.deepEqual(
    [...source.matchAll(/kind: "(thinking|tool)"/g)].slice(0, 4).map((match) => match[1]),
    ["thinking", "tool", "thinking", "tool"],
  );
  assert.match(source, /aria-label="动画播放控制"/);
});

test("every candidate has motion styling and a reduced-motion path", () => {
  for (const candidate of ["hybrid", "breathe", "flow", "echo", "spring", "scan"]) {
    assert.match(styles, new RegExp(`motion-card--${candidate}`));
  }
  assert.match(styles, /@media \(prefers-reduced-motion: reduce\)/);
  assert.doesNotMatch(styles, /transition:\s*all/);
});
