import assert from "node:assert/strict";
import test from "node:test";

import {
  applyChatFrame,
  parseChatFrame,
  sendWhenOpen,
  traceKindForChatFrame,
} from "./web-chat-transport.ts";

test("WebSocket boundary validates every frame family and observable trace lane", () => {
  const thinking = parseChatFrame({ type: "react.thinking.delta", session_id: "one", turn_id: "turn", delta: "思" });
  const answer = parseChatFrame({ type: "answer.delta", session_id: "one", turn_id: "turn", delta: "答" });
  const terminal = parseChatFrame({ type: "message.final", session_id: "one", turn_id: "turn", content: "答案", duration_ms: 42 });
  assert.equal(traceKindForChatFrame(thinking), "thinking");
  assert.equal(traceKindForChatFrame(answer), "answer");
  assert.equal(traceKindForChatFrame(terminal), "terminal");
  assert.throws(() => parseChatFrame({ type: "answer.delta", session_id: "one" }), /缺少字符串字段: turn_id/u);
  assert.throws(() => parseChatFrame({ type: "message.final", session_id: "one", turn_id: "turn", content: "x", duration_ms: "42" }), /duration_ms 格式无效/u);
  assert.throws(() => parseChatFrame({ type: "future.frame" }), /未知消息类型/u);
});

test("frame controller preserves thinking, tool, answer, and terminal lifecycle", () => {
  let activeSessionId = "session";
  let messages = [];
  let status = "idle";
  let activeTurnId = null;
  let error = "";
  const loadedSessions = [];
  const loadedMessages = [];
  const context = {
    activeSessionId: () => activeSessionId,
    activateSession: (next) => { activeSessionId = next; },
    setError: (next) => { error = next; },
    setMessages: (updater) => {
      messages = updater(messages);
    },
    getStatus: () => status,
    setStatus: (next) => { status = next; },
    getActiveTurnId: () => activeTurnId,
    setActiveTurnId: (next) => { activeTurnId = next; },
    loadSessions: async () => { loadedSessions.push(activeSessionId); },
    loadMessages: async (sessionId) => { loadedMessages.push(sessionId); },
  };

  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn", content: "" }), context);
  applyChatFrame(parseChatFrame({ type: "react.thinking.delta", session_id: "session", turn_id: "turn", delta: "思考" }), context);
  applyChatFrame(parseChatFrame({ type: "react.tool.started", session_id: "session", turn_id: "turn", call_id: "call", tool_name: "shell", arguments: { cmd: "pwd" } }), context);
  applyChatFrame(parseChatFrame({ type: "react.tool.completed", session_id: "session", turn_id: "turn", call_id: "call", tool_name: "shell", status: "success", result_preview: "ok" }), context);
  applyChatFrame(parseChatFrame({ type: "answer.delta", session_id: "session", turn_id: "turn", delta: "答案" }), context);
  applyChatFrame(parseChatFrame({ type: "message.final", session_id: "session", turn_id: "turn", content: "答案", media: ["/tmp/a.png"], duration_ms: 50 }), context);

  assert.equal(status, "idle");
  assert.equal(error, "");
  assert.equal(messages.length, 1);
  assert.equal(messages[0].content, "答案");
  assert.equal(messages[0].streaming, false);
  assert.deepEqual(messages[0].blocks.map((block) => block.kind), ["thinking", "tool"]);
  assert.equal(messages[0].blocks[1].status, "output-available");
  assert.equal(messages[0].attachments[0].mediaType, "image/png");
  assert.deepEqual(loadedMessages, ["session"]);
  assert.deepEqual(loadedSessions, ["session"]);
});

test("foreign frames cannot mutate the active session and push terminal lands immediately", () => {
  let messages = [{ id: "turn", role: "assistant", content: "", blocks: [], streaming: true }];
  const context = {
    activeSessionId: () => "active",
    activateSession: () => {},
    setError: () => {},
    setMessages: (updater) => {
      messages = updater(messages);
    },
    getStatus: () => "streaming",
    setStatus: () => {},
    getActiveTurnId: () => "turn",
    setActiveTurnId: () => {},
    loadSessions: async () => {},
    loadMessages: async () => {},
  };
  applyChatFrame(parseChatFrame({ type: "answer.delta", session_id: "foreign", turn_id: "turn", delta: "wrong" }), context);
  assert.equal(messages[0].content, "");
  applyChatFrame(parseChatFrame({
    type: "message.final",
    session_id: "active",
    turn_id: "turn",
    content: "push",
    metadata: { source: "message_push" },
  }), context);
  assert.equal(messages[0].content, "push");
  assert.equal(messages[0].streaming, true);
});

test("output completed enters finalizing then terminal returns to idle", () => {
  let status = "idle";
  let activeTurnId = null;
  const context = {
    activeSessionId: () => "session",
    activateSession: () => {},
    setError: () => {},
    setMessages: () => {},
    getStatus: () => status,
    setStatus: (next) => { status = next; },
    getActiveTurnId: () => activeTurnId,
    setActiveTurnId: (next) => { activeTurnId = next; },
    loadSessions: async () => {},
    loadMessages: async () => {},
  };

  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn", content: "" }), context);
  assert.equal(status, "streaming");

  applyChatFrame(parseChatFrame({ type: "answer.delta", session_id: "session", turn_id: "turn", delta: "答案" }), context);
  assert.equal(status, "streaming");

  applyChatFrame(parseChatFrame({ type: "turn.output.completed", session_id: "session", turn_id: "turn", client_message_id: "cmid" }), context);
  assert.equal(status, "finalizing");

  applyChatFrame(parseChatFrame({ type: "message.final", session_id: "session", turn_id: "turn", content: "答案" }), context);
  assert.equal(status, "idle");
});

test("late output completed after terminal is ignored and keeps idle", () => {
  let status = "idle";
  let activeTurnId = null;
  const context = {
    activeSessionId: () => "session",
    activateSession: () => {},
    setError: () => {},
    setMessages: () => {},
    getStatus: () => status,
    setStatus: (next) => { status = next; },
    getActiveTurnId: () => activeTurnId,
    setActiveTurnId: (next) => { activeTurnId = next; },
    loadSessions: async () => {},
    loadMessages: async () => {},
  };

  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn", content: "" }), context);
  assert.equal(status, "streaming");

  // /stop terminal 先到，composer 回 idle
  applyChatFrame(parseChatFrame({ type: "turn.interrupted", request_id: "r", session_id: "session", status: "interrupted", message: "已中断" }), context);
  assert.equal(status, "idle");

  // 迟到的 output.completed 不得把 idle 改回 finalizing
  applyChatFrame(parseChatFrame({ type: "turn.output.completed", session_id: "session", turn_id: "turn", client_message_id: "cmid" }), context);
  assert.equal(status, "idle");
});

test("stale output completed from previous turn does not pollute next turn", () => {
  let status = "idle";
  let activeTurnId = null;
  const context = {
    activeSessionId: () => "session",
    activateSession: () => {},
    setError: () => {},
    setMessages: () => {},
    getStatus: () => status,
    setStatus: (next) => { status = next; },
    getActiveTurnId: () => activeTurnId,
    setActiveTurnId: (next) => { activeTurnId = next; },
    loadSessions: async () => {},
    loadMessages: async () => {},
  };

  // T1 开始 → 中断 → idle
  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn-1", content: "" }), context);
  assert.equal(status, "streaming");
  applyChatFrame(parseChatFrame({ type: "turn.interrupted", request_id: "r", session_id: "session", status: "interrupted", message: "已中断" }), context);
  assert.equal(status, "idle");

  // T2 开始（新 turn）
  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn-2", content: "" }), context);
  assert.equal(status, "streaming");
  assert.equal(activeTurnId, "turn-2");

  // T1 迟到的 output.completed 不得污染 T2
  applyChatFrame(parseChatFrame({ type: "turn.output.completed", session_id: "session", turn_id: "turn-1", client_message_id: "cmid" }), context);
  assert.equal(status, "streaming");
  assert.equal(activeTurnId, "turn-2");
});

test("stale final closes its own row without terminating the next turn", () => {
  let status = "idle";
  let activeTurnId = null;
  let messages = [];
  const context = {
    activeSessionId: () => "session",
    activateSession: () => {},
    setError: () => {},
    setMessages: (updater) => { messages = updater(messages); },
    getStatus: () => status,
    setStatus: (next) => { status = next; },
    getActiveTurnId: () => activeTurnId,
    setActiveTurnId: (next) => { activeTurnId = next; },
    loadSessions: async () => {},
    loadMessages: async () => {},
  };

  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn-1", content: "" }), context);
  applyChatFrame(parseChatFrame({ type: "answer.delta", session_id: "session", turn_id: "turn-1", delta: "T1 partial" }), context);
  applyChatFrame(parseChatFrame({ type: "turn.output.completed", session_id: "session", turn_id: "turn-1" }), context);
  applyChatFrame(parseChatFrame({ type: "turn.started", session_id: "session", turn_id: "turn-2", content: "" }), context);
  applyChatFrame(parseChatFrame({ type: "answer.delta", session_id: "session", turn_id: "turn-2", delta: "T2 partial" }), context);

  applyChatFrame(parseChatFrame({ type: "message.final", session_id: "session", turn_id: "turn-1", content: "T1 final" }), context);

  assert.equal(status, "streaming");
  assert.equal(activeTurnId, "turn-2");
  assert.deepEqual(
    messages.map(({ id, content, streaming }) => ({ id, content, streaming })),
    [
      { id: "turn-1", content: "T1 final", streaming: false },
      { id: "turn-2", content: "T2 partial", streaming: true },
    ],
  );
});

test("send transport serializes once, waits for open, and aborts before delivery", async () => {
  const originalWebSocket = globalThis.WebSocket;
  globalThis.WebSocket = { CONNECTING: 0, OPEN: 1 };
  class FakeSocket extends EventTarget {
    constructor(readyState) {
      super();
      this.readyState = readyState;
      this.sent = [];
    }
    send(value) { this.sent.push(value); }
    close() { this.readyState = 3; }
  }
  try {
    const open = new FakeSocket(1);
    await sendWhenOpen(open, { type: "turn.stop" });
    assert.deepEqual(open.sent, ['{"type":"turn.stop"}']);

    const connecting = new FakeSocket(0);
    const pending = sendWhenOpen(connecting, { type: "message.send", text: "hi" });
    connecting.readyState = 1;
    connecting.dispatchEvent(new Event("open"));
    await pending;
    assert.equal(connecting.sent.length, 1);

    const controller = new AbortController();
    const aborted = new FakeSocket(0);
    const rejected = sendWhenOpen(aborted, { type: "message.send" }, controller.signal);
    controller.abort();
    await assert.rejects(rejected, { name: "AbortError" });
    assert.equal(aborted.sent.length, 0);

    const stalled = new FakeSocket(0);
    await assert.rejects(
      sendWhenOpen(stalled, { type: "message.send" }, undefined, 1),
      /聊天连接超时/u,
    );
    assert.equal(stalled.sent.length, 0);
    assert.equal(stalled.readyState, 3);
  } finally {
    globalThis.WebSocket = originalWebSocket;
  }
});
