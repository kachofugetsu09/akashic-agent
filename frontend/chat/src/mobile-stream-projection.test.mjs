import assert from "node:assert/strict";
import test from "node:test";

import {
  advanceMobileStreamPresentation,
  MobileStreamProjectionStore,
  mobileStreamFrameBudget,
} from "./mobile-stream-projection.ts";

class TestFrameScheduler {
  callback = null;
  timestamp = 0;

  request(callback) {
    assert.equal(this.callback, null);
    this.callback = callback;
    return 1;
  }

  cancel() {
    this.callback = null;
  }

  advance(elapsedMs = 16.67) {
    const callback = this.callback;
    assert.notEqual(callback, null);
    this.callback = null;
    this.timestamp += elapsedMs;
    callback(this.timestamp);
  }
}

function message(id, content, detail = "") {
  return { id, content, blocks: detail ? [{ id: "thinking", detail }] : [] };
}

test("stream projection wakes only the subscribed message row", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正");
  const target = message("assistant:turn", "正在检查流式链路");
  let activeUpdates = 0;
  let historyUpdates = 0;
  store.subscribe("assistant:turn", () => { activeUpdates += 1; });
  store.subscribe("history", () => { historyUpdates += 1; });

  store.publish(before.id, before, target, false);
  assert.equal(store.read(before.id, before), before);
  scheduler.advance();

  assert.equal(store.read(before.id, before).content, "正在检");
  assert.equal(activeUpdates, 1);
  assert.equal(historyUpdates, 0);
});

test("terminal projection bypasses smoothing and preserves an id migration alias", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正在");
  const terminal = message("message:canonical", "正在分析完成");

  store.publish(before.id, before, terminal, true);

  assert.equal(store.read(before.id, before), terminal);
  assert.equal(store.read(terminal.id, before), terminal);
  assert.equal(scheduler.callback, null);
});

test("presentation applies tool structure immediately and smooths thinking text", () => {
  const before = message("assistant:turn", "", "先");
  const target = {
    ...message("assistant:turn", "", "先检查调用链"),
    blocks: [
      { id: "thinking", detail: "先检查调用链" },
      { id: "tool", detail: "读取配置", state: "running" },
    ],
  };

  const next = advanceMobileStreamPresentation(before, target, 16.67);

  assert.equal(next.blocks[0].detail, "先检查");
  assert.equal(next.blocks[1], target.blocks[1]);
});

test("answer-only frames preserve the shared process block list", () => {
  const blocks = [{ id: "tool", detail: "完成", state: "completed" }];
  const before = { id: "assistant:turn", content: "回", blocks };
  const target = { id: "assistant:turn", content: "回答继续", blocks };

  const next = advanceMobileStreamPresentation(before, target, 16.67);

  assert.equal(next.blocks, blocks);
  assert.equal(next.content, "回答继");
});

test("resetting for a coarse snapshot does not wake streaming rows twice", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正");
  const target = message("assistant:turn", "正在检查");
  let updates = 0;
  store.subscribe(before.id, () => { updates += 1; });
  store.publish(before.id, before, target, false);
  scheduler.advance();

  store.clear();

  assert.equal(updates, 1);
  assert.equal(store.read(before.id, before), before);
});

test("a 100-character-per-second source is presented in frame-sized slices", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const content = "流式输出".repeat(25);
  let authoritative = message("assistant:turn", "");
  let lastLength = 0;
  let largestFrame = 0;
  store.subscribe(authoritative.id, () => {
    const visible = store.read(authoritative.id, authoritative);
    const length = Array.from(visible.content).length;
    largestFrame = Math.max(largestFrame, length - lastLength);
    lastLength = length;
  });

  for (let end = 10; end <= content.length; end += 10) {
    const target = message(authoritative.id, content.slice(0, end));
    store.publish(authoritative.id, authoritative, target, false);
    authoritative = target;
    for (let frame = 0; frame < 6 && scheduler.callback !== null; frame += 1) {
      scheduler.advance(16.67);
    }
  }

  assert.equal(store.read(authoritative.id, authoritative).content, content);
  assert.equal(largestFrame, 2);
});

test("frame budget targets about one or two characters per display frame", () => {
  assert.equal(mobileStreamFrameBudget(8, 20), 1);
  assert.equal(mobileStreamFrameBudget(16.67, 20), 2);
  assert.equal(mobileStreamFrameBudget(16.67, 100), 6);
});
