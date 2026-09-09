import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";

const compiled = await build({ entryPoints: [new URL("./mobile.js", import.meta.url).pathname],
  bundle: true, write: false, platform: "node", format: "cjs", loader: { ".css": "empty" } });

/** 单一受控时钟覆盖加载提示、可见轮询和卸载后的停止。 */
function loadRenderer() {
  const timers = new Map();
  let now = 0, nextId = 0;
  const setTimer = (callback, delay) => { const id = ++nextId; timers.set(id, { callback, due: now + delay }); return id; };
  const module = { exports: {} };
  new Function("require", "module", "exports", "setTimeout", "clearTimeout", compiled.outputFiles[0].text)(
    createRequire(import.meta.url), module, module.exports, setTimer, id => timers.delete(id));
  const advance = async (duration) => {
    const end = now + duration;
    while (true) {
      const item = [...timers].sort((a, b) => a[1].due - b[1].due)[0];
      if (!item || item[1].due > end) break;
      now = item[1].due; timers.delete(item[0]); item[1].callback();
      await Promise.resolve();
    }
    now = end;
  };
  return { ...module.exports, advance, timers };
}

const result = (pending, preview = "已召回的原消息") => ({ pending, items: [{ hits: [
  { lane: "dense", messages: [{ preview }] }, { lane: "completion", messages: [{ preview: "另一条模式记忆" }] },
] }] });

test("refresh failure keeps visible recall and open lane, retry settles without hidden polling", async () => {
  const { mountRecall, advance, timers } = loadRenderer();
  const dom = new JSDOM("<div id='root'></div>");
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { value: dom.window.document, configurable: true });
  const host = document.getElementById("root");
  const calls = [];
  const context = { messageId: "message", capabilities: { queryCacheModes: ["memory"] },
    query: (method, payload, options) => new Promise((resolve, reject) => calls.push({ method, payload, options, resolve, reject })) };
  let dispose;
  try {
    dispose = mountRecall(host, context);
    assert.equal(calls[0].options.cache, "memory");
    assert.equal(host.textContent, "");
    await advance(150);
    assert.match(host.textContent, /正在读取召回记录/);
    calls[0].resolve(result(true));
    await Promise.resolve();
    host.querySelector("details").open = true;
    await advance(1000);
    assert.equal(calls.length, 2);
    calls[1].reject(new Error("插件请求超时"));
    await Promise.resolve();
    assert.match(host.textContent, /已召回的原消息/);
    assert.equal(host.querySelector("details").open, true);
    host.querySelector("button").click();
    assert.equal(calls.length, 3);
    calls[2].resolve(result(false, "完整召回结果"));
    await Promise.resolve();
    assert.equal(host.querySelector("details").open, true);
    assert.equal(host.querySelector("[role='status']").hidden, true);
    await advance(10_000);
    assert.equal(calls.length, 3);
    dispose();
    assert.equal(timers.size, 0);
  } finally {
    dispose?.(); dom.window.close();
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  }
});

test("prefetch reads once and an older Host keeps its supported cache mode", async () => {
  const { default: definition, advance, timers } = loadRenderer();
  const calls = [];
  await definition.slots["turn.before_reasoning"].prefetch({ messageId: "message",
    capabilities: { queryTransports: ["inline", "https"] },
    query: async (...args) => { calls.push(args); return result(true); } });
  await advance(10_000);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][2].cache, "none");
  assert.equal(timers.size, 0);
});
