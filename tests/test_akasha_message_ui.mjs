import assert from "node:assert/strict";
import test from "node:test";
import { JSDOM } from "jsdom";
import { build } from "esbuild";
const compiled = await build({ entryPoints: [new URL("../frontend/plugins/akasha/src/mobile.js", import.meta.url).pathname],
  bundle: true, write: false, format: "esm", loader: { ".css": "empty" } });
const { mountRecall, mount, renderDetail } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString("base64")}`);


const detail = {
  schema: "akasha.queries.v1", query_text: "Context", query_text_truncated: false,
  ts: "2026-09-05T15:00:00Z", source: { kind: "context", session_id: "s", source: "conversation", through_seq: 3 },
  hit_count: 1, presented_count: 1, graph_version: 1, pushes: 0, residual_l1: 0,
  hits: [{ lane: "dense", score: 1, sources: ["direct_dense"], messages: [
    { message_id: "u", preview: '<img src=x onerror="alert(1)">', presented: false, truncated: true },
    { message_id: "a", preview: "answer", presented: true, truncated: false },
  ] }],
};

test("actual recall display keeps each member and escapes stored text", () => {
  const html = renderDetail(detail);
  assert.equal((html.match(/<article>/g) ?? []).length, 2);
  assert.match(html, /u · 未提供/);
  assert.match(html, /a · 已提供/);
  assert.match(html, /正文预览/);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(html, /<img/);
});

test("an unsupported query schema is a visible failure, not empty memory", () => {
  assert.throws(() => renderDetail({ ...detail, schema: "unknown" }), /查询格式不受支持/);
});

test("closing Inspector prevents a late page response from replacing its host", async () => {
  let resolve;
  const response = new Promise((done) => { resolve = done; });
  const host = { innerHTML: "" };
  const close = mount(host, { query: () => response });
  close();
  host.innerHTML = "another page";
  resolve({ schema: "akasha.queries.v1", items: [], total: 0, page: 1, page_size: 30 });
  await new Promise((done) => setImmediate(done));
  assert.equal(host.innerHTML, "another page");
});


test("chat recall mounts the old two lanes, escapes memory text and stops after unmount", async () => {
  const dom = new JSDOM("<div id='host'></div>");
  const host = dom.window.document.getElementById("host");
  const calls = [];
  const close = mountRecall(host, { messageId: "draft", block: { source: "conversation" },
    query: async (...args) => { calls.push(args); return { items: [detail], pending: false }; } });
  await new Promise((done) => setImmediate(done));
  assert.equal(host.querySelectorAll(".akasha-mobile-recall").length, 2);
  assert.match(host.textContent, /左脑 · 精确回忆/);
  assert.match(host.textContent, /右脑 · 模式补全/);
  assert.equal(host.querySelector("img"), null);
  assert.doesNotMatch(host.textContent, /conversation/);
  assert.deepEqual(calls[0], ["recall.turn", { message_id: "draft", source: "conversation" },
    { cache: "none", transport: "https" }]);
  close();
  dom.window.close();
});
