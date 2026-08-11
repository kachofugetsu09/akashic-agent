import assert from "node:assert/strict";
import test from "node:test";

import { StreamProjectionStore } from "./stream-projection.ts";
import {
  advanceWebStreamPresentation,
  publishWebStreamChanges,
} from "./web-stream-projection.ts";

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

function assistant(content, streaming = true, blocks = []) {
  return {
    id: "assistant:turn",
    role: "assistant",
    content,
    blocks,
    streaming,
  };
}

test("desktop stream reveals one code point and applies tool structure immediately", () => {
  const before = assistant("", true, []);
  const target = assistant("回答", true, [
    { kind: "thinking", content: "思考" },
    {
      kind: "tool",
      callId: "call:1",
      name: "read",
      status: "input-available",
      input: {},
      output: undefined,
      errorText: undefined,
    },
  ]);

  const next = advanceWebStreamPresentation(before, target, 16.67);

  assert.equal(next.blocks[0].content, "思");
  assert.equal(next.blocks[1], target.blocks[1]);
  assert.equal(next.content, "");
});

test("desktop projection starts next frame and terminal content appears immediately", () => {
  const scheduler = new TestFrameScheduler();
  const store = new StreamProjectionStore(scheduler, advanceWebStreamPresentation);
  const before = assistant("正");
  const chunk = assistant("正在检查流式链路");

  publishWebStreamChanges([before], [chunk], store, false);
  assert.equal(store.read(before.id, before).content, "正");
  scheduler.advance();
  assert.equal(store.read(before.id, before).content, "正在");

  const terminal = assistant("正在检查流式链路完成", false);
  publishWebStreamChanges([chunk], [terminal], store, false);
  assert.equal(store.read(before.id, before), terminal);
  assert.equal(scheduler.callback, null);
});

test("a completed push bypasses smoothing even while the target stays marked streaming", () => {
  const scheduler = new TestFrameScheduler();
  const store = new StreamProjectionStore(scheduler, advanceWebStreamPresentation);
  const before = assistant("正");
  const chunk = assistant("正在检查流式链路");
  publishWebStreamChanges([before], [chunk], store, false);
  scheduler.advance();

  const pushed = assistant("主动推送已完成", true);
  publishWebStreamChanges([chunk], [pushed], store, true);

  assert.equal(store.read(before.id, before), pushed);
  assert.equal(scheduler.callback, null);
});

test("one frame keeps a supplementary Unicode code point intact", () => {
  const before = assistant("");
  const target = assistant("😀好");

  const next = advanceWebStreamPresentation(before, target, 16.67);

  assert.equal(next.content, "😀");
  assert.equal(Array.from(next.content).length, 1);
});

test("a non-prefix correction replaces text instead of fabricating an append", () => {
  const before = assistant("旧前缀");
  const corrected = assistant("权威纠正");

  const next = advanceWebStreamPresentation(before, corrected, 16.67);

  assert.equal(next, corrected);
});
