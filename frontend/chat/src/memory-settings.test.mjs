import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const view = await readFile(new URL("./memory-settings.tsx", import.meta.url), "utf8");
const dialog = await readFile(new URL("./memory-embedding-dialog.tsx", import.meta.url), "utf8");
const controller = await readFile(new URL("./use-memory-settings.ts", import.meta.url), "utf8");
const data = await readFile(new URL("./memory-settings-data.ts", import.meta.url), "utf8");

test("memory view delegates persistence and embedding credentials", () => {
  assert.match(view, /useMemorySettings/);
  assert.match(view, /<MemoryEmbeddingDialog/);
  assert.doesNotMatch(view, /\bfetch\b|api_key|requestSettingsJson/);
  assert.match(data, /requestSettingsJson/);
});

test("memory and embedding mutations have independent single owners", () => {
  assert.match(controller, /if \(requestRef\.current\) return/);
  assert.match(dialog, /if \(requestRef\.current\) return/);
  assert.match(controller, /requestRef\.current\?\.abort\(\)/);
  assert.match(dialog, /requestRef\.current\?\.abort\(\)/);
});

test("embedding dialog explicitly restores focus to its opener", () => {
  assert.match(dialog, /onCloseAutoFocus/);
  assert.match(dialog, /returnFocusRef\.current\?\.focus\(\)/);
});
