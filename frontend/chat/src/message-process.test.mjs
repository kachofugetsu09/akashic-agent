import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";

const compiled = await build({
  entryPoints: [new URL("./message-view.tsx", import.meta.url).pathname],
  bundle: true, write: false, platform: "node", format: "cjs", packages: "external",
  alias: { "@": new URL(".", import.meta.url).pathname },
  loader: { ".css": "empty", ".svg": "dataurl" },
  plugins: [{ name: "fixture-plugin", setup(builder) {
    builder.onResolve({ filter: /(?:^|\/)message-response$/ }, () => ({ path: "fixture-markdown", namespace: "markdown" }));
    builder.onLoad({ filter: /.*/, namespace: "markdown" }, () => ({ loader: "js", contents: "export function MessageResponse({ children }) { return children; }" }));
    builder.onResolve({ filter: /mobile-plugin-runtime$/ }, () => ({ path: "fixture-plugin", namespace: "fixture" }));
    builder.onLoad({ filter: /.*/, namespace: "fixture" }, () => ({ resolveDir: new URL(".", import.meta.url).pathname, loader: "js", contents: `
      const React = require("react");
      export function MobilePluginSlot() {
        return React.createElement("div", { "data-recall": true }, "左脑 · 右脑");
      }` }));
  } }],
});
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputFiles[0].text)(createRequire(import.meta.url), module, module.exports);
const { ReplyActivityView } = module.exports;

test("recall remains in the same thinking panel from waiting to the first thinking chunk", async () => {
  const dom = new JSDOM("<div id='root'></div>", { url: "http://localhost/" });
  const globals = { window: dom.window, document: dom.window.document, HTMLElement: dom.window.HTMLElement,
    getComputedStyle: dom.window.getComputedStyle, requestAnimationFrame: (fn) => setTimeout(fn, 0),
    cancelAnimationFrame: clearTimeout, IS_REACT_ACT_ENVIRONMENT: true };
  const previous = new Map(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true });
  const root = createRoot(document.getElementById("root"));
  const activity = { handle: "h", session_id: "s", source: "conversation", active: true,
    preview: { message_id: "draft", call_record_id: "call", text: "", thinking: "" } };
  try {
    const render = async (thinking) => act(async () => root.render(React.createElement(ReplyActivityView,
      { activity: { ...activity, preview: { ...activity.preview, thinking } }, committed: new Set() })));
    await render("");
    const recall = document.querySelector("[data-recall]");
    assert.ok(recall?.closest(".process-trace"));
    assert.equal(document.querySelectorAll(".process-trace").length, 1);
    await render("正在核对记忆");
    assert.equal(document.querySelector("[data-recall]"), recall);
    assert.equal(document.querySelectorAll("[data-recall]").length, 1);
    assert.ok(recall.closest(".process-trace"));
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});
