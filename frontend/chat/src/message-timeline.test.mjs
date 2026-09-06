import assert from "node:assert/strict";
import test from "node:test";
import { chatHistoryPage, sessionPage } from "./web-chat-data.ts";
import { mergeTimelineMessages, readMessageLogFrame, readTimelineMessage, timelineReply, timelineText } from "./message-timeline.ts";

const row = (seq, body, changes = {}) => ({
  id: `message-${seq}`, session_id: "akashic:fixture", seq,
  timestamp: "2026-09-06T12:00:00+08:00", author: "fixture-author", source: "fixture-source",
  attachments: [], body, ...changes,
});
const text = (value) => ({ kind: "text", value });
const page = (items, through, more = false) => ({ version: 2, items, through_seq: through,
  has_more: more, before_seq: more ? items[0].seq : null });

test("fixed history pages retain all four bodies, gaps, late results and raw archives", () => {
  const archive = { raw: '[ {"result":null, "arguments": "old"} ]', completeness: "unknown" };
  const records = [
    row(0, { kind: "input", parts: [text("[后台任务完成] 保留原消息")] }),
    row(3, { kind: "output", finish: "continue", parts: [text("先说"),
      { kind: "tool_call", name: "OriginalTool", binding_id: "old-binding", arguments: { path: "file" } }, text("后说")] }),
    row(8, { kind: "control", action: "abandon", through_seq: 3, reason: "用户取消" }),
    row(9, { kind: "tool_result", call_ref: { message_id: "message-3", part_index: 1 }, outcome: "unknown", parts: [text("晚到结果")] }),
    row(12, { kind: "output", finish: "quiet", parts: [{ kind: "history.transcript", archive }, { kind: "private.kind", display: "unavailable" }] }),
  ];
  const snapshot = structuredClone(records);
  const latest = chatHistoryPage(page(records.slice(2), 12, true), "fixture");
  const older = chatHistoryPage(page(records.slice(0, 2), 12), "fixture");
  const merged = mergeTimelineMessages(latest.items, older.items);
  assert.deepEqual(merged, records);
  assert.deepEqual(merged.map((item) => item.body.kind), ["input", "output", "control", "tool_result", "output"]);
  assert.deepEqual(mergeTimelineMessages(merged, older.items), records);
  assert.deepEqual(records, snapshot);
  assert.equal(timelineText(merged[0]), "[后台任务完成] 保留原消息");
  assert.equal(timelineReply(merged[1]).author, "fixture-author");
  assert.equal(timelineReply(merged[1]).preview, "先说 后说");
});

test("history rejects old projection, invalid ordering, wrong head and identity collisions", () => {
  const first = row(0, { kind: "input", parts: [text("one")] });
  const second = row(1, { kind: "output", parts: [text("two")], finish: "complete" });
  assert.throws(() => chatHistoryPage({ items: [{ id: 0, role: "assistant", content: "old" }], total: 1 }, "fixture"), /无效历史页/u);
  for (const payload of [page([second, first], 1), page([first, first], 1), page([second], 0),
    { ...page([first], 1, true), before_seq: 5 }, { ...page([first], 1), before_seq: 0 }]) {
    assert.throws(() => chatHistoryPage(payload, "fixture"), /历史游标/u);
  }
  assert.throws(() => mergeTimelineMessages([first], [{ ...second, id: first.id }]), /身份冲突/u);
  assert.throws(() => mergeTimelineMessages([first], [{ ...first, id: "duplicate-seq" }]), /身份冲突/u);
  assert.throws(() => mergeTimelineMessages([first], [{ ...second, session_id: "other" }]), /身份冲突/u);
  assert.throws(() => mergeTimelineMessages([first], [{ ...first, body: { kind: "input", parts: [text("changed")] } }]), /正文发生变化/u);
});

test("attachments and tool references fail visibly at the boundary instead of dropping content", () => {
  const input = row(0, { kind: "input", parts: [{ kind: "artifact_ref", value: "artifact" }] });
  assert.throws(() => readTimelineMessage(input), /附件引用/u);
  const attachment = { artifact_id: "artifact", kind: "image", filename: "image.png", media_type: "image/png", size_bytes: 10, sha256: "a".repeat(64) };
  assert.deepEqual(readTimelineMessage({ ...input, attachments: [attachment] }).attachments, [attachment]);
  assert.throws(() => readTimelineMessage(row(1, { kind: "input", parts: [{ kind: "tool_call", name: "tool", binding_id: "binding", arguments: {} }] })), /工具调用所在消息/u);
  assert.throws(() => readTimelineMessage(row(2, { kind: "tool_result", parts: [], call_ref: { message_id: "m", part_index: -1 }, outcome: "success" })), /引用或状态/u);
  assert.throws(() => readTimelineMessage(row(3, { kind: "output", finish: "complete", parts: [{ kind: "private.kind", value: "secret" }] })), /内容无效/u);
});

test("session directory keeps empty and attachment-only entries and validates continuation", () => {
  const items = [{ key: "empty", message_count: 0, first_message_content: "" },
    { key: "attachment-only", message_count: 1, first_message_content: "" }];
  const next = { updated_at: "2026-09-06T12:00:00+08:00", session_id: "attachment-only" };
  assert.deepEqual(sessionPage({ items, next_cursor: next }), { items, nextCursor: next });
  assert.deepEqual(sessionPage({ items: [], next_cursor: null }), { items: [], nextCursor: null });
  assert.throws(() => sessionPage({ items, next_cursor: { ...next, updated_at: "invalid" } }), /目录游标/u);
});

test("live messages retain fixed page progress and reject missing or cross-session facts", () => {
  const frame = { type: "messages.appended", version: 2, session_id: "akashic:fixture", after_seq: -1,
    through_seq: 5, next_after_seq: 3, has_more: true, items: [row(3, { kind: "input", parts: [] })] };
  assert.deepEqual(readMessageLogFrame(frame), frame);
  for (const changes of [{ version: 1 }, { after_seq: true }, { items: [] }, { next_after_seq: 2 },
    { has_more: false }, { through_seq: 1 }, { session_id: "other" }, { items: [...frame.items, ...frame.items] }]) {
    assert.throws(() => readMessageLogFrame({ ...frame, ...changes }));
  }
});

test("reply snapshots distinguish unavailable, idle, active preview and draining", () => {
  const item = { session_id: "akashic:fixture", source: "conversation", handle: "scope", active: true,
    preview: { message_id: "future-message", text: "当前草稿", thinking: "思考" } };
  const frame = { type: "reply.status", version: 2, session_id: item.session_id,
    snapshot_id: "current", available: true, items: [item] };
  assert.deepEqual(readMessageLogFrame(frame), frame);
  for (const changes of [{ items: [] }, { items: [{ ...item, preview: { ...item.preview, truncated: true } }] },
    { items: [{ ...item, active: false, preview: null }] },
    { available: false, snapshot_id: null, items: [] }]) {
    assert.deepEqual(readMessageLogFrame({ ...frame, ...changes }), { ...frame, ...changes });
  }
  for (const changes of [{ available: false }, { snapshot_id: null }, { items: [{ ...item, active: false }] },
    { items: [{ ...item, preview: { ...item.preview, truncated: "yes" } }] },
    { items: [item, item] }, { items: [{ ...item, session_id: "other" }] }, { items: [{ ...item, preview: {} }] }]) {
    assert.throws(() => readMessageLogFrame({ ...frame, ...changes }));
  }
});
