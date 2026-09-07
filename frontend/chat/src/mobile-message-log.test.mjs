import assert from "node:assert/strict";
import test from "node:test";
import { applyMobileMessageEvent, readMobileDownloads, readMobileMessageLog, readMobileStateSnapshot, mergeMobileMessageSnapshot } from "./mobile-message-log.ts";

const session = "akashic:mobile-fixture";
const message = (id, seq, body) => ({ id, seq, session_id: session, timestamp: "2026-09-06T01:00:00Z",
  author: "scheduler", source: "scheduled-program", attachments: [], body });
const text = (value) => ({ kind: "text", value });
const baseline = (messages = []) => ({ selectedSessionId: session, projectionGeneration: 4,
  messages, throughSeq: messages.at(-1)?.seq ?? -1, replyStatus: null });
const event = (event, projectionGeneration = 4) => ({ protocolVersion: 1, projectionGeneration, event });
const status = (items = []) => ({ type: "reply.status", version: 2, session_id: session,
  snapshot_id: "generation-a", available: true, items });
const activity = { session_id: session, source: "scheduled-program", handle: "scope-a", active: true,
  preview: { message_id: "answer", text: "草稿", thinking: "思考" } };

test("mobile snapshot preserves all bodies, long tool arguments and archive values", () => {
  const rows = [
    message("input", 0, { kind: "input", parts: [text("程序输入")] }),
    message("call", 2, { kind: "output", finish: "continue", parts: [
      { kind: "tool_call", binding_id: "b", name: "read", arguments: { query: "花月".repeat(50000) } },
    ] }),
    message("result", 4, { kind: "tool_result", call_ref: { message_id: "call", part_index: 0 }, outcome: "error",
      parts: [{ kind: "history.transcript", archive: { detail: ["原文", { value: "保留" }] } }] }),
    message("control", 6, { kind: "control", action: "pause", through_seq: 4, reason: null }),
  ];
  const raw = baseline(rows);
  const before = JSON.stringify(raw);
  assert.deepEqual(readMobileMessageLog(raw).messages, rows);
  assert.equal(JSON.stringify(raw), before);
  assert.throws(() => readMobileMessageLog({ ...raw, messages: [{ id: "ref", seq: 1, message_ref: {} }] }));
  assert.throws(() => readMobileMessageLog({ ...raw, messages: [rows[1], rows[0]] }));
  assert.throws(() => readMobileMessageLog({ ...raw, messages: [{ ...rows[0], session_id: "other" }] }));
  assert.throws(() => readMobileMessageLog({ ...raw, throughSeq: 4 }));
});

test("preview, commitment and draining share an ID without mutating messages", () => {
  const input = message("input", 0, { kind: "input", parts: [text("开始")] });
  const initial = readMobileMessageLog(baseline([input]));
  const preview = applyMobileMessageEvent(initial, event(status([activity])));
  assert.deepEqual(preview.messages, [input]);
  const answer = message("answer", 2, { kind: "output", finish: "complete", parts: [text("正式内容")] });
  const appended = { type: "messages.appended", version: 2, session_id: session,
    after_seq: 0, through_seq: 2, next_after_seq: 2, has_more: false, items: [answer] };
  const committed = applyMobileMessageEvent(preview, event(appended));
  assert.deepEqual(committed.messages.map((row) => row.id), ["input", "answer"]);
  assert.equal(committed.messages[0], input);
  const draining = applyMobileMessageEvent(committed, event(status([{ ...activity, active: false, preview: null }])));
  assert.equal(draining.messages, committed.messages);
  const done = applyMobileMessageEvent(draining, event(status()));
  assert.deepEqual(done.replyStatus.items, []);
  assert.equal(done.throughSeq, 2);
  assert.throws(() => applyMobileMessageEvent(done, event(appended)), /游标不连续/);
});

test("stale session and generation events cannot replace current activities", () => {
  const current = readMobileMessageLog(baseline());
  assert.equal(applyMobileMessageEvent(current, event(status([activity]), 3)), null);
  assert.equal(applyMobileMessageEvent(current, event({ ...status(), session_id: "other" })), null);
  assert.throws(() => applyMobileMessageEvent(current, { ...event(status()), protocolVersion: 3 }));
  assert.throws(() => applyMobileMessageEvent(current, event({ type: "answer.delta", delta: "old" })));
  assert.throws(() => readMobileMessageLog({ ...baseline(), replyStatus: status([{ ...activity, active: false }]) }));
  const unavailable = applyMobileMessageEvent(current, event({ ...status(), snapshot_id: null, available: false }));
  assert.equal(unavailable.replyStatus.available, false);
});

test("device download progress remains separate from attachment facts", () => {
  const artifact = { artifact_id: "a", kind: "file", filename: "report.txt", media_type: "text/plain",
    size_bytes: 200, sha256: "a".repeat(64) };
  const row = { ...message("input", 0, { kind: "input", parts: [{ kind: "artifact_ref", value: "a" }] }), attachments: [artifact] };
  const current = readMobileMessageLog(baseline([row]));
  const progress = { artifactId: "a", transferredBytes: 20, state: "downloading" };
  const downloaded = { ...progress, transferredBytes: 200, state: "cached", contentUrl: "https://appassets.androidplatform.net/media/a" };
  assert.deepEqual(readMobileDownloads([progress]), [progress]);
  assert.deepEqual(readMobileDownloads([downloaded]), [downloaded]);
  assert.deepEqual(current.messages[0].attachments, [artifact]);
  assert.throws(() => readMobileDownloads([progress, downloaded]));
  assert.throws(() => readMobileDownloads([{ ...progress, transferredBytes: -1 }]));
});

test("control patches reject message and preview fields before snapshot conversion", () => {
  const raw = { protocolVersion: 2, projectionGeneration: 4, selectedSessionId: session, downloads: [] };
  for (const field of ["messages", "throughSeq", "replyStatus", "messageId", "contentAppend"]) {
    assert.throws(() => readMobileStateSnapshot({ ...raw, [field]: null }), /字段无效/);
  }
  assert.throws(() => readMobileStateSnapshot({ ...raw, protocolVersion: 1 }));
  assert.deepEqual(readMobileStateSnapshot(raw), { ...raw, protocolVersion: 9, messages: [], throughSeq: -1, replyStatus: null });
});

test("late full snapshots cannot roll back a newer session or sync generation", () => {
  const current = { ...baseline([message("latest", 8, { kind: "input", parts: [text("新消息")] })]), projectionGeneration: 8 };
  const old = { ...baseline(), projectionGeneration: 7 };
  assert.equal(mergeMobileMessageSnapshot(current, old), null);
  assert.equal(mergeMobileMessageSnapshot(current, { ...old, selectedSessionId: "other" }), null);
  assert.throws(() => mergeMobileMessageSnapshot(current, { ...old, projectionGeneration: 8 }), /游标回退/);
  assert.throws(() => mergeMobileMessageSnapshot(current, { ...current, selectedSessionId: "other" }), /递增同步代际/);
  const next = { ...old, projectionGeneration: 9, selectedSessionId: "new-session" };
  assert.equal(mergeMobileMessageSnapshot(current, next), next);
  const olderPage = { ...current, messages: [message("earlier", 1, { kind: "input", parts: [text("历史")] })] };
  assert.deepEqual(mergeMobileMessageSnapshot(current, olderPage).messages.map((row) => row.id), ["earlier", "latest"]);
});

test("a delayed history or download snapshot cannot revive an ended preview", () => {
  const old = { ...baseline(), replyStatus: status([activity]) };
  const current = applyMobileMessageEvent(old, event(status([])));
  const refreshed = mergeMobileMessageSnapshot(current, old);
  assert.equal(refreshed.replyStatus, current.replyStatus);
  assert.deepEqual(refreshed.replyStatus.items, []);
  const nextEpoch = { ...old, projectionGeneration: 5 };
  assert.equal(mergeMobileMessageSnapshot(current, nextEpoch).replyStatus, nextEpoch.replyStatus);
});
