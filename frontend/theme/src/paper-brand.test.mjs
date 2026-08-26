import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const brand = await readFile(new URL("./brand-tokens.css", import.meta.url), "utf8");
const mobile = await readFile(new URL("../../chat/src/mobile-native.tsx", import.meta.url), "utf8");
const mobileStyles = await readFile(new URL("../../chat/src/mobile-native.css", import.meta.url), "utf8");
const lab = await readFile(new URL("../../chat/src/mobile-lab.css", import.meta.url), "utf8");

test("paper brand exposes orthogonal semantic token axes", () => {
  for (const prefix of ["--ak-paper-", "--ak-ink-", "--ak-rule-", "--ak-type-", "--ak-annotation-"]) {
    assert.match(brand, new RegExp(prefix));
  }
  assert.doesNotMatch(brand, /--ak-(?:card|button|chip)-/);
});

test("production Mobile follows the shared WebUI without decorative role copy", () => {
  assert.match(mobile, /import "\.\/message-view\.css"/);
  assert.doesNotMatch(mobile, /mobile-manuscript|你的题记|Akashic 手稿/);
  assert.match(mobileStyles, /same conversation language as the desktop WebUI/);
  assert.match(mobileStyles, /--chat-chip: var\(--ak-paper-sheet\)/);
  assert.match(mobileStyles, /\.mobile-message-anchor \.user-bubble/);
});

test("focus preview gives the complete viewport to production Mobile", () => {
  assert.match(lab, /body\.is-focus-mode \.lab-header\s*\{\s*display: none;/);
  assert.match(lab, /block-size: 100svh;/);
});
