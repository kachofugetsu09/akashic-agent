import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("./drawer-island-showcase.tsx", import.meta.url), "utf8");
const css = await readFile(new URL("./drawer-island-showcase.css", import.meta.url), "utf8");
const main = await readFile(new URL("./main.tsx", import.meta.url), "utf8");

test("pre-unification drawer preview uses the real shared navigation component", () => {
  assert.match(source, /ConversationNavigation/);
  assert.match(source, /v0\.8\.15（315b4ba）/);
  assert.match(source, /label: "知识与运行",[\s\S]*?featured: true,/);
  assert.match(source, /label: "插件"/);
  assert.match(source, /className="legacy-memory-summary"/);
  assert.match(main, /preview === "drawer-islands"/);
  assert.doesNotMatch(source, /fetch\(|WebSocket|AkashicNative/);
});

test("preview preserves current sessions and every mobile action", () => {
  for (const id of ["heart-rate", "mobile-link", "drawer-study", "settings", "diagnostics", "resync", "pairing", "new-chat"]) {
    assert.match(source, new RegExp(`id: "${id}"`));
  }
  assert.match(source, /10 条待整理/);
});

test("preview adds no second navigation styling language", () => {
  assert.doesNotMatch(css, /drawer-capability-row|drawer-session-row|drawer-action/);
  assert.doesNotMatch(css, /transition\s*:\s*all/);
  assert.doesNotMatch(css, /gradient/i);
});
