import assert from "node:assert/strict";
import test from "node:test";

import { StreamProjectionStore } from "./stream-projection.ts";
import { canProjectWebStreamWithoutRoot, publishWebStreamChanges } from "./web-stream-projection.ts";

function assistant(id, content, streaming = true) {
  return { id, role: "assistant", content, blocks: [], streaming };
}

function frameScheduler() {
  let nextHandle = 1;
  const callbacks = new Map();
  return {
    schedule(callback) {
      const handle = nextHandle;
      nextHandle += 1;
      callbacks.set(handle, callback);
      return handle;
    },
    cancel(handle) {
      callbacks.delete(handle);
    },
    flush() {
      const pending = [...callbacks.values()];
      callbacks.clear();
      for (const callback of pending) callback(0);
    },
    get pending() {
      return callbacks.size;
    },
  };
}

test("frame publication stores the latest target and notifies only the affected row once", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
  const before = assistant("assistant:turn", "正在");
  const middle = assistant("assistant:turn", "正在检查");
  const target = assistant("assistant:turn", "正在检查流式链路");
  let activeUpdates = 0;
  let historyUpdates = 0;
  store.subscribe("assistant:turn", () => { activeUpdates += 1; });
  store.subscribe("history", () => { historyUpdates += 1; });

  store.publishFrame(before.id, middle);
  store.publishFrame(before.id, target);

  assert.equal(store.read(before.id, before), target);
  assert.equal(frames.pending, 1);
  assert.equal(activeUpdates, 0);
  frames.flush();
  assert.equal(activeUpdates, 1);
  assert.equal(historyUpdates, 0);
});

test("immediate terminal cancels a stale frame and preserves the canonical id alias", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
  const before = assistant("assistant:turn", "正在");
  const terminal = assistant("message:canonical", "正在分析完成", false);
  let updates = 0;
  store.subscribe(before.id, () => { updates += 1; });

  store.publishFrame(before.id, assistant(before.id, "即将完成"));
  store.publishImmediate(before.id, terminal);

  assert.equal(store.read(before.id, before), terminal);
  assert.equal(store.read(terminal.id, before), terminal);
  assert.equal(frames.pending, 0);
  assert.equal(updates, 1);
  frames.flush();
  assert.equal(updates, 1);
});

test("reconcileBaseline drops projections already committed into the coarse snapshot", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
  const fallback = assistant("assistant:turn", "");
  const target = assistant("assistant:turn", "已提交");
  store.publishFrame("assistant:turn", target);

  store.reconcileBaseline([target]);

  assert.equal(store.read("assistant:turn", fallback), fallback);
  assert.equal(frames.pending, 0);
});

test("clear drops every projection and scheduled notification", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
  const target = assistant("assistant:turn", "内容");
  let updates = 0;
  store.subscribe(target.id, () => { updates += 1; });
  store.publishFrame("assistant:turn", target);
  store.clear();

  assert.equal(store.read("assistant:turn", target), target);
  assert.equal(frames.pending, 0);
  frames.flush();
  assert.equal(updates, 0);
});

test("publishWebStreamChanges schedules active assistant mutations only", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
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
  assert.equal(frames.pending, 1);
});

test("publishWebStreamChanges skips unchanged and non-streaming rows", () => {
  const frames = frameScheduler();
  const store = new StreamProjectionStore(frames.schedule, frames.cancel);
  const previous = [
    assistant("assistant:done", "历史", false),
    assistant("assistant:turn", "流式"),
  ];
  const terminal = assistant("assistant:turn", "流式结束", false);
  const next = [previous[0], terminal];

  publishWebStreamChanges(previous, next, store);

  assert.equal(store.read("assistant:turn", previous[1]), terminal);
  assert.equal(store.read("assistant:done", previous[0]), previous[0]);
  assert.equal(frames.pending, 0);
});

test("only an active last-row stream mutation can bypass the app root", () => {
  const history = assistant("assistant:history", "历史", false);
  const active = assistant("assistant:turn", "流式");
  const delta = assistant("assistant:turn", "流式继续");

  assert.equal(canProjectWebStreamWithoutRoot([history, active], [history, delta]), true);
  assert.equal(canProjectWebStreamWithoutRoot([history, active], [history, assistant("assistant:turn", "完成", false)]), false);
  assert.equal(canProjectWebStreamWithoutRoot([history, active], [assistant("assistant:history", "变化", false), delta]), false);
});
