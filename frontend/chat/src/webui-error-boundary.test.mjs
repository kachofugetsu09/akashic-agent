import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const entry = await readFile(new URL("./main.tsx", import.meta.url), "utf8");
const app = await readFile(new URL("./desktop-chat-view.tsx", import.meta.url), "utf8");
const boundary = await readFile(new URL("./webui-error-boundary.tsx", import.meta.url), "utf8");

test("entry lazy surfaces have an actionable fail-loud boundary", () => {
  assert.match(entry, /<WebUiErrorBoundary>/);
  assert.match(boundary, /role="alert"/);
  assert.match(boundary, /window\.location\.reload\(\)/);
  assert.match(boundary, /不会删除对话或设置/);
});

test("message rendering failure remains local and actionable", () => {
  assert.match(app, /message-renderer-error" role="alert"/);
  assert.match(app, />重新加载页面<\/button>/);
});
