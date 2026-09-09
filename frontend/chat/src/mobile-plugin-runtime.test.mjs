import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";

const compiled = await build({
  entryPoints: [new URL("./mobile-plugin-runtime.tsx", import.meta.url).pathname],
  bundle: true, write: false, platform: "node", format: "cjs", packages: "external",
});
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputFiles[0].text)(createRequire(import.meta.url), module, module.exports);
const { MobilePluginSlot, receiveMobilePluginCatalog, receiveMobilePluginResult } = module.exports;

test("queued cards get a full transport deadline while slow requests release their slots", async () => {
  const dom = new JSDOM("<div id='root'></div>", { url: "http://localhost/" });
  const globals = { window: dom.window, document: dom.window.document, HTMLElement: dom.window.HTMLElement,
    IntersectionObserver: class {
      constructor(callback) { this.callback = callback; }
      observe() { this.callback([{ isIntersecting: true }]); }
      disconnect() {}
    }, IS_REACT_ACT_ENVIRONMENT: true };
  const previous = new Map(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true });
  const root = createRoot(document.getElementById("root"));
  const sent = [];
  const cancelled = [];
  window.AkashicNative = {
    queryPluginUi: (requestId, ownerId, ...args) => sent.push({ requestId, ownerId, args }),
    cancelPluginUiOwner: (ownerId) => cancelled.push(ownerId),
  };
  window.queryOutcomes = [];
  try {
    const source = `export default { slots: { "turn.before_reasoning": { mount(host, context) {
      context.query("read", { message: context.messageId }).then(
        value => window.queryOutcomes.push({ message: context.messageId, value }),
        error => window.queryOutcomes.push({ message: context.messageId, error: error.message }),
      );
    } } } };`;
    await receiveMobilePluginCatalog({ catalogRevision: "a".repeat(64), updating: false,
      plugins: [{ id: "fixture", revision: "b".repeat(64), slots: ["turn.before_reasoning"],
        moduleUrl: `data:text/javascript,${encodeURIComponent(source)}` }] });

    // 用受控时钟走真实 renderer -> 排队 -> bridge -> 结果链，不等待墙上时间。
    let now = 0;
    let nextId = 0;
    const timers = new Map();
    window.setTimeout = (callback, delay) => {
      const id = ++nextId;
      timers.set(id, { callback, due: now + delay });
      return id;
    };
    window.clearTimeout = (id) => timers.delete(id);
    const advance = (until) => {
      while (true) {
        const next = [...timers].sort((a, b) => a[1].due - b[1].due)[0];
        if (!next || next[1].due > until) break;
        now = next[1].due;
        timers.delete(next[0]);
        next[1].callback();
      }
      now = until;
    };
    await act(async () => root.render(React.createElement(React.Fragment, null,
      ...["first", "second", "queued"].map(messageId => React.createElement(MobilePluginSlot,
        { key: messageId, name: "turn.before_reasoning", sessionId: "session", messageId })))));
    assert.equal(sent.length, 2);
    advance(29_999);
    assert.deepEqual(window.queryOutcomes, []);
    await act(async () => advance(30_000));
    assert.equal(sent.length, 3);
    assert.deepEqual(window.queryOutcomes.map(value => [value.message, value.error]),
      [["first", "插件请求超时"], ["second", "插件请求超时"]]);
    assert.deepEqual(cancelled, sent.slice(0, 2).map(value => value.ownerId));
    await act(async () => receiveMobilePluginResult({ requestId: sent[2].requestId, resultJson: '{"ready":true}' }));
    assert.deepEqual(window.queryOutcomes.at(-1), { message: "queued", value: { ready: true } });
    assert.equal(timers.size, 0);
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});
