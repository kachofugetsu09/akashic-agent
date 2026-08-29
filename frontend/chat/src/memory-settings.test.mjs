import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const view = await readFile(new URL("./embedding-settings.tsx", import.meta.url), "utf8");
const dialog = await readFile(new URL("./memory-embedding-dialog.tsx", import.meta.url), "utf8");
const data = await readFile(new URL("./memory-settings-data.ts", import.meta.url), "utf8");

test("embedding view delegates persistence and credentials to model control", () => {
  assert.match(view, /<MemoryEmbeddingDialog/);
  assert.doesNotMatch(view, /\bfetch\b|api_key|requestSettingsJson/);
  assert.match(data, /set_default/);
  assert.doesNotMatch(data, /api\/settings\/memory|embedding_model_id|enabled:/);
});

test("embedding dialog serializes its model mutation", () => {
  assert.match(dialog, /if \(requestRef\.current\) return/);
  assert.match(dialog, /requestRef\.current\?\.abort\(\)/);
});

test("embedding dialog explicitly restores focus to its opener", () => {
  assert.match(dialog, /onCloseAutoFocus/);
  assert.match(dialog, /returnFocusRef\.current\?\.focus\(\)/);
});
