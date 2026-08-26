import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const brand = await readFile(new URL("./brand-tokens.css", import.meta.url), "utf8");
const manuscript = await readFile(new URL("../../chat/src/mobile-manuscript.css", import.meta.url), "utf8");
const mobile = await readFile(new URL("../../chat/src/mobile-native.tsx", import.meta.url), "utf8");
const lab = await readFile(new URL("../../chat/src/mobile-lab.css", import.meta.url), "utf8");

test("paper brand exposes orthogonal semantic token axes", () => {
  for (const prefix of ["--ak-paper-", "--ak-ink-", "--ak-rule-", "--ak-type-", "--ak-annotation-"]) {
    assert.match(brand, new RegExp(prefix));
  }
  assert.doesNotMatch(brand, /--ak-(?:card|button|chip)-/);
});

test("production Mobile composes the paper brand without Material component tokens", () => {
  assert.match(mobile, /import "\.\/mobile-manuscript\.css"/);
  assert.match(mobile, /你的题记/);
  assert.match(mobile, /Akashic 手稿/);
  assert.match(manuscript, /var\(--ak-paper-/);
  assert.match(manuscript, /var\(--ak-ink-/);
  assert.match(manuscript, /var\(--ak-annotation-/);
  assert.doesNotMatch(manuscript, /--md-sys-|--m-shape-/);
});

test("focus preview gives the complete viewport to production Mobile", () => {
  assert.match(lab, /body\.is-focus-mode \.lab-header\s*\{\s*display: none;/);
  assert.match(lab, /block-size: 100svh;/);
});
