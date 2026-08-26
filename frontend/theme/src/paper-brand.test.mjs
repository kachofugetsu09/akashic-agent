import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const brand = await readFile(new URL("./brand-tokens.css", import.meta.url), "utf8");
const desktopStyles = await readFile(new URL("../../chat/src/styles.css", import.meta.url), "utf8");
const mobile = await readFile(new URL("../../chat/src/mobile-native.tsx", import.meta.url), "utf8");
const mobileStyles = await readFile(new URL("../../chat/src/mobile-native.css", import.meta.url), "utf8");
const dashboardStyles = await readFile(new URL("../../dashboard/src/styles.css", import.meta.url), "utf8");
const paperSurface = await readFile(new URL("./paper-surface.css", import.meta.url), "utf8");
const lab = await readFile(new URL("../../chat/src/mobile-lab.css", import.meta.url), "utf8");

test("paper brand exposes orthogonal semantic token axes", () => {
  for (const prefix of ["--ak-paper-", "--ak-ink-", "--ak-rule-", "--ak-type-"]) {
    assert.match(brand, new RegExp(prefix));
  }
  assert.doesNotMatch(brand, /--ak-(?:annotation|card|button|chip)-/);
});

test("every public brand token has a real product consumer", () => {
  const consumers = [desktopStyles, mobileStyles, dashboardStyles, paperSurface].join("\n");
  const tokens = [...brand.matchAll(/^\s*(--ak-(?:paper|ink|rule|type)-[a-z0-9-]+):/gm)]
    .map((match) => match[1]);
  assert.ok(tokens.length > 0);
  for (const token of tokens) {
    assert.match(consumers, new RegExp(`var\\(${token}\\)`), `${token} has no product consumer`);
  }
});

test("Desktop and Mobile map shared chat roles to the same brand tokens", () => {
  for (const declaration of [
    "--chat-page: var(--ak-paper-canvas)",
    "--chat-ink: var(--ak-ink-primary)",
    "--chat-muted: var(--ak-ink-secondary)",
    "--chat-line: var(--ak-rule-subtle)",
    "--chat-chip: var(--ak-paper-sheet)",
    "--chat-lift: var(--ak-paper-editing)",
  ]) {
    assert.match(desktopStyles, new RegExp(declaration.replace(/[()]/g, "\\$&")));
    assert.match(mobileStyles, new RegExp(declaration.replace(/[()]/g, "\\$&")));
  }
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
