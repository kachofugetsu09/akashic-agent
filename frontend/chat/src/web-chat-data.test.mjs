import assert from "node:assert/strict";
import test from "node:test";

import {
  chatHistoryPage,
  chatModelState,
  fetchChatJson,
  messageRows,
  sessionRows,
  uploadFiles,
  webShellState,
} from "./web-chat-data.ts";
import {
  formatNavigationTime,
  isVisibleChatRow,
  rowToMessage,
  sessionLabel,
} from "./web-chat-message-data.ts";

test("desktop HTTP boundary accepts complete payloads and rejects malformed rows", () => {
  assert.deepEqual(sessionRows({ items: [{ key: "session-1", message_count: 2 }] }), [
    { key: "session-1", message_count: 2 },
  ]);
  assert.throws(() => sessionRows({ items: [{ key: "", message_count: 2 }] }), /无效 session 行/u);
  assert.throws(
    () => messageRows({ items: [{ id: 1, role: "user", content: "hi", reply_role: "assistant" }] }, "/messages"),
    /无效 message 行/u,
  );
  assert.deepEqual(webShellState({ status: "ready", configured: true, chatReady: true, settingsPath: "/settings" }), {
    status: "ready",
    configured: true,
    chatReady: true,
    settingsPath: "/settings",
  });
  assert.throws(() => webShellState({ status: "ready" }), /无效状态/u);
});

test("chat history boundary owns the stable seq cursor", () => {
  assert.deepEqual(chatHistoryPage({
    items: [{ id: 8, seq: 8, role: "user", content: "older" }],
    total: 12,
    has_more: true,
    before_seq: 8,
  }, "/messages"), {
    items: [{ id: 8, seq: 8, role: "user", content: "older" }],
    total: 12,
    hasMore: true,
    beforeSeq: 8,
  });
  assert.throws(() => chatHistoryPage({
    items: [{ id: 8, seq: 8, role: "user", content: "older" }],
    total: 12,
    has_more: true,
    before_seq: 7,
  }, "/messages"), /不一致的历史游标/u);
});

test("desktop model registry is validated once before presentation", () => {
  const state = chatModelState({
    generationId: 3,
    defaultRuntime: "fixture",
    sessionOverride: "",
    sessionSelection: { modelRef: "fixture:model", reasoningEffort: "medium" },
    runtimes: [{
      id: "fixture",
      provider: "fixture",
      model: "model",
      sourceId: "source",
      sourceName: "Fixture",
      reasoningEffort: "medium",
      supportedReasoningEfforts: ["low", "medium"],
      roles: ["main"],
    }],
  });
  assert.equal(state.runtimes[0].supportedReasoningEfforts[1], "medium");
  assert.throws(() => chatModelState({ ...state, generationId: 1.5 }), /无效模型注册表/u);
});

test("history projection owns reply, tools, media, filtering, and navigation labels", () => {
  const [row] = messageRows({ items: [{
    id: 9,
    role: "assistant",
    content: "完成",
    media: ["/tmp/result.png"],
    reasoning_content: "思考",
    turn_duration_ms: "42",
    reply_to_message_id: "7",
    reply_role: "user",
    reply_preview: "问题",
    tool_chain: [{ calls: [{ name: "shell", status: "success", result: "ok" }] }],
  }] }, "/messages");
  const message = rowToMessage(row);
  assert.equal(message.id, "9");
  assert.equal(message.durationMs, 42);
  assert.equal(message.attachments?.[0].mediaType, "image/png");
  assert.deepEqual(message.blocks.map((block) => block.kind), ["tool", "thinking"]);
  assert.equal(message.reply?.messageId, "7");
  assert.equal(isVisibleChatRow({ id: 1, role: "user", content: "[后台任务完成]内部", }), false);
  assert.equal(sessionLabel({ key: "one", first_message_content: "a".repeat(29) }), `${"a".repeat(28)}...`);
  assert.equal(formatNavigationTime("invalid"), undefined);
});

test("HTTP and upload failures stay explicit at the transport boundary", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response("not-json", { status: 200 });
    await assert.rejects(fetchChatJson("/invalid"), /无效 JSON/u);

    globalThis.fetch = async (input) => String(input).startsWith("blob:")
      ? new Response("file-body", { status: 200 })
      : new Response(JSON.stringify({ filename: "note.txt", upload_path: "/uploads/note.txt" }), { status: 200 });
    const uploaded = await uploadFiles(
      [{ filename: "note.txt", url: "blob:note" }],
      new AbortController().signal,
    );
    assert.deepEqual(uploaded, [{ filename: "note.txt", upload_path: "/uploads/note.txt" }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
