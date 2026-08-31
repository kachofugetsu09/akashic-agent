import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../plugins/akasha/mobile_ui.js", import.meta.url),
  "utf8",
);
const pluginModule = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);
const plugin = pluginModule.default;
const recall = plugin.slots["turn.before_reasoning"];
const settle = () => new Promise((resolve) => setImmediate(resolve));

test("Akasha exposes one current-turn recall slot and one Inspector", () => {
  assert.deepEqual(Object.keys(plugin.slots), ["turn.before_reasoning"]);
  assert.equal(typeof recall.mount, "function");
  assert.equal(typeof plugin.dashboard.mount, "function");
});

test("current-turn recall uses the HTTPS transport without caching", async () => {
  const calls = [];
  const host = { innerHTML: "" };
  const cleanup = recall.mount(host, {
    capabilities: { queryTransports: ["https"] },
    messageId: "assistant:turn-1",
    query: async (...args) => {
      calls.push(args);
      return {
        left: [{ user_text: "旧字段仍可显示", ts: "2026-08-30T00:00:00Z", score: 0.9 }],
        right: [],
      };
    },
  });

  await settle();

  assert.deepEqual(calls, [[
    "recall.current",
    { message_id: "assistant:turn-1" },
    { cache: "none", transport: "https" },
  ]]);
  assert.match(host.innerHTML, /旧字段仍可显示/);
  cleanup();
});

test("active recall stays visible while an unpublished projection retries", async () => {
  const host = { innerHTML: "旧内容" };
  const cleanup = recall.mount(host, {
    capabilities: { queryTransports: ["https"] },
    messageId: "assistant:turn-1",
    query: async () => ({
      pending: true,
      recall_capture_available: true,
      left: [{ user_text: "已发布回忆", ts: "2026-08-30T00:00:00Z", score: 0.9 }],
      right: [],
    }),
  });

  await settle();

  assert.match(host.innerHTML, /已发布回忆/);
  cleanup();
});

test("settled recall projections use the immutable cache", async () => {
  const calls = [];
  const host = { innerHTML: "" };
  const cleanup = recall.mount(host, {
    capabilities: { queryTransports: ["https"] },
    messageId: "message-1",
    query: async (...args) => {
      calls.push(args);
      return { left: [], right: [] };
    },
  });

  await settle();

  assert.deepEqual(calls[0][2], { cache: "immutable", transport: "https" });
  cleanup();
});

test("Inspector loads recent retrievals through the plugin query boundary", async () => {
  const calls = [];
  const host = {
    innerHTML: "",
    querySelectorAll: () => [],
  };
  const cleanup = plugin.dashboard.mount(host, {
    query: async (...args) => {
      calls.push(args);
      return { items: [], total: 0 };
    },
  });

  await settle();

  assert.deepEqual(calls, [["inspector.recent"]]);
  assert.match(host.innerHTML, /还没有可检查的检索记录/);
  cleanup();
});

test("recall reports an unsupported client without issuing a query", () => {
  let queried = false;
  const host = { innerHTML: "" };
  const cleanup = recall.mount(host, {
    capabilities: { queryTransports: [] },
    messageId: "assistant:turn-1",
    query: async () => {
      queried = true;
      return {};
    },
  });

  assert.equal(cleanup, undefined);
  assert.equal(queried, false);
  assert.match(host.innerHTML, /不支持 Akasha 轻量数据通道/);
});
