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

/** 通过真实 Host 和受控 Native 回执验证缓存生命周期。 */
async function withMemoryRenderer(run) {
  const dom = new JSDOM("<div id='root'></div>", { url: "http://localhost/" });
  const globals = { window: dom.window, document: dom.window.document, HTMLElement: dom.window.HTMLElement,
    IntersectionObserver: class { constructor(callback) { this.callback = callback; }
      observe() { this.callback([{ isIntersecting: true }]); } disconnect() {} },
    IS_REACT_ACT_ENVIRONMENT: true };
  const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true });
  const root = createRoot(document.getElementById("root"));
  const sent = [], cancelled = [];
  window.AkashicNative = {
    queryPluginUi: (requestId, ownerId, ...args) => sent.push({ requestId, ownerId, args }),
    cancelPluginUiOwner: ownerId => cancelled.push(ownerId),
  };
  window.memoryOutcomes = [];
  const source = `const read = context => context.query("read", { message: context.messageId }, { cache: "memory" });
    export default { slots: { "turn.before_reasoning": {
      prefetch: async context => { await read(context); },
      mount(host, context) {
        let active = true;
        read(context).then(value => { if (active) { host.textContent = value.text; window.memoryOutcomes.push(value); } },
          error => { if (active) host.textContent = error.message; });
        return () => { active = false; };
      }
    } } };`;
  let revision = 0;
  const catalog = () => receiveMobilePluginCatalog({ catalogRevision: String(++revision).padStart(64, "c"), updating: false,
    plugins: [{ id: "memory-fixture", revision: String(revision).padStart(64, "d"), slots: ["turn.before_reasoning"],
      moduleUrl: `data:text/javascript,${encodeURIComponent(source)}` }] });
  const card = (messageId, prefetch = false, key = messageId) => React.createElement(MobilePluginSlot,
    { key, name: "turn.before_reasoning", sessionId: "session", messageId, prefetch });
  const render = async (...children) => act(async () => root.render(React.createElement(React.Fragment, null, ...children)));
  const reply = async (index, result) => act(async () => receiveMobilePluginResult({ requestId: sent[index].requestId,
    ...(result instanceof Error ? { error: result.message } : { resultJson: JSON.stringify(result) }) }));
  try {
    await catalog();
    await run({ sent, cancelled, card, render, reply, catalog });
  } finally {
    await act(async () => root.unmount());
    await act(async () => receiveMobilePluginCatalog({ catalogRevision: "", updating: true, plugins: [] }));
    dom.window.close();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

test("prefetch and open share one read that completes across collapse and warms reopen", async () => {
  await withMemoryRenderer(async ({ sent, cancelled, card, render, reply }) => {
    await render(card("one", true, "prefetch"), card("one", false, "open"));
    assert.equal(sent.length, 1);
    assert.equal(sent[0].args[6], "none", "old Native must never receive the memory cache enum");
    await render();
    assert.ok(!cancelled.includes(sent[0].ownerId), "collapse must not cancel the shared wire owner");
    await reply(0, { text: "recalled", pending: false });
    assert.deepEqual(window.memoryOutcomes, [], "detached renderer must stay detached");
    await render(card("one"));
    assert.equal(sent.length, 1);
    assert.equal(document.querySelector("[data-plugin='memory-fixture']").textContent, "recalled");
  });
});

test("pending and failed reads are retried instead of becoming final cache entries", async () => {
  await withMemoryRenderer(async ({ sent, card, render, reply }) => {
    await render(card("one"));
    await reply(0, { text: "partial", pending: true });
    await render();
    await render(card("one"));
    assert.equal(sent.length, 2);
    await reply(1, new Error("timeout"));
    await render();
    await render(card("one"));
    assert.equal(sent.length, 3);
    await reply(2, { text: "final", pending: false });
    await render();
    await render(card("one"));
    assert.equal(sent.length, 3);
  });
});

test("a visible reader promotes its queued prefetch and abandoned queued prefetches are removed", async () => {
  await withMemoryRenderer(async ({ sent, card, render, reply }) => {
    await render(card("first", true), card("second", true), card("queued", true));
    assert.equal(sent.length, 2);
    await render(card("first", true), card("second", true), card("queued", true), card("queued", false, "open"));
    assert.equal(sent.length, 3, "opening uses an available interactive slot");
    await reply(2, { text: "third", pending: false });
    await render(card("first", true), card("second", true), card("discard", true));
    assert.equal(sent.length, 3);
    await render(card("first", true), card("second", true));
    await reply(0, { text: "first", pending: false });
    await reply(1, { text: "second", pending: false });
    assert.equal(sent.length, 3, "offscreen queued read must never reach Native");
  });
});

test("catalog replacement cancels a retained wire read and discards its late result", async () => {
  await withMemoryRenderer(async ({ sent, cancelled, card, render, reply, catalog }) => {
    await render(card("one"));
    await render();
    await act(async () => catalog());
    assert.ok(cancelled.includes(sent[0].ownerId));
    await reply(0, { text: "old", pending: false });
    await render(card("one"));
    assert.equal(sent.length, 2);
    await reply(1, { text: "new", pending: false });
    assert.equal(window.memoryOutcomes.at(-1).text, "new");
  });
});
