import assert from "node:assert/strict";
import test from "node:test";

import { StreamProjectionStore } from "./stream-projection.ts";
import { publishWebStreamChanges } from "./web-stream-projection.ts";

function assistant(id, content, streaming = true) {
  return { id, role: "assistant", content, blocks: [], streaming };
}

test("publish lands the authoritative target immediately and notifies only the affected row", () => {
  const store = new StreamProjectionStore();
  const before = assistant("assistant:turn", "正在");
  const target = assistant("assistant:turn", "正在检查流式链路");
  let activeUpdates = 0;
  let historyUpdates = 0;
  store.subscribe("assistant:turn", () => { activeUpdates += 1; });
  store.subscribe("history", () => { historyUpdates += 1; });

  store.publish(before.id, target);

  assert.equal(store.read(before.id, before), target);
  assert.equal(activeUpdates, 1);
  assert.equal(historyUpdates, 0);
});

test("publish preserves an id migration alias for canonical terminal ids", () => {
  const store = new StreamProjectionStore();
  const before = assistant("assistant:turn", "正在");
  const terminal = assistant("message:canonical", "正在分析完成", false);

  store.publish(before.id, terminal);

  assert.equal(store.read(before.id, before), terminal);
  assert.equal(store.read(terminal.id, before), terminal);
});

test("reconcileBaseline drops projections already committed into the coarse snapshot", () => {
  const store = new StreamProjectionStore();
  const fallback = assistant("assistant:turn", "");
  const target = assistant("assistant:turn", "已提交");
  store.publish("assistant:turn", target);

  store.reconcileBaseline([target]);

  assert.equal(store.read("assistant:turn", fallback), fallback);
});

test("clear drops every projection", () => {
  const store = new StreamProjectionStore();
  const target = assistant("assistant:turn", "内容");
  store.publish("assistant:turn", target);
  store.clear();

  assert.equal(store.read("assistant:turn", target), target);
});

test("publishWebStreamChanges publishes active assistant mutations only", () => {
  const store = new StreamProjectionStore();
  const previous = [
    { id: "user:1", role: "user", content: "问", blocks: [], streaming: false },
    assistant("assistant:turn", "答"),
  ];
  const next = [
    previous[0],
    assistant("assistant:turn", "回答继续"),
  ];

  publishWebStreamChanges(previous, next, store);

  assert.equal(store.read("assistant:turn", previous[1]), next[1]);
  assert.equal(store.read("user:1", previous[0]), previous[0]);
});

test("publishWebStreamChanges skips unchanged and non-streaming rows", () => {
  const store = new StreamProjectionStore();
  const previous = [
    assistant("assistant:done", "历史", false),
    assistant("assistant:turn", "流式"),
  ];
  const terminal = assistant("assistant:turn", "流式结束", false);
  const next = [previous[0], terminal];

  publishWebStreamChanges(previous, next, store);

  assert.equal(store.read("assistant:turn", previous[1]), terminal);
  assert.equal(store.read("assistant:done", previous[0]), previous[0]);
});
