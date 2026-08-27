import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MOBILE_TURN_MISSING,
  MOBILE_TURN_TRACE_MAX_TRACKED,
  MobileTurnTraceRegistry,
  mobileTurnFirstVisibleKinds,
  mobileTurnTraceEmit,
  parseMobileTurnId,
} from "./mobile-turn-trace.ts";
const mobileSource = await readFile(
  new URL("./mobile-native.tsx", import.meta.url),
  "utf8",
);

function captureSink() {
  const records = [];
  return {
    records,
    sink: (record) => { records.push(record); },
  };
}

test("turn id parses only from the non-empty assistant:<turn> messageId contract", () => {
  assert.equal(parseMobileTurnId("assistant:turn-01J"), "turn-01J");
  assert.equal(parseMobileTurnId("assistant:"), undefined);
  assert.equal(parseMobileTurnId("user:abc"), undefined);
  assert.equal(parseMobileTurnId(""), undefined);
});

test("full identity is session + turn + client_message_id", () => {
  const registry = new MobileTurnTraceRegistry(() => {});
  const identity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.equal(identity.sessionId, "session-1");
  assert.equal(identity.turnId, "turn-1");
  assert.equal(identity.clientMessageId, "client-1");
  assert.equal(registry.identityFor("session-1", "turn-1").clientMessageId, "client-1");
});

test("missing identity parts are marked explicitly, never guessed", () => {
  const registry = new MobileTurnTraceRegistry(() => {});
  const identity = registry.registerTurnIdentity("session-1", undefined, undefined);
  assert.equal(identity.turnId, MOBILE_TURN_MISSING);
  assert.equal(identity.clientMessageId, MOBILE_TURN_MISSING);
  const noClientId = registry.registerTurnIdentity("session-1", "turn-1", undefined);
  assert.equal(noClientId.clientMessageId, MOBILE_TURN_MISSING);
});

test("missing client_message_id fills later; conflicting non-missing values degrade to one diagnostic", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const filled = registry.registerTurnIdentity("session-1", "turn-1", undefined);
  assert.equal(filled.clientMessageId, MOBILE_TURN_MISSING);
  const later = registry.registerTurnIdentity("session-1", "turn-1", "client-3");
  assert.equal(later.clientMessageId, "client-3");
  assert.equal(registry.identityFor("session-1", "turn-1").clientMessageId, "client-3");
  const conflicting = registry.registerTurnIdentity("session-1", "turn-1", "client-4");
  assert.equal(conflicting.clientMessageId, "client-3");
  assert.equal(captured.records.length, 1);
  assert.equal(captured.records[0].event, "webui.identity_conflict");
  assert.equal(captured.records[0].client_message_id, "client-3");
  assert.equal(captured.records[0].incoming_client_message_id, "client-4");
  const again = registry.registerTurnIdentity("session-1", "turn-1", "client-4");
  assert.equal(again.clientMessageId, "client-3");
  assert.equal(captured.records.length, 1);
});

test("conflicting non-missing client_message_id keeps the first identity, never throws, markFirst still works", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const identity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.doesNotThrow(() => {
    const conflicting = registry.registerTurnIdentity("session-1", "turn-1", "client-2");
    assert.equal(conflicting.key, identity.key);
    assert.equal(conflicting.clientMessageId, "client-1");
  });
  assert.equal(captured.records.length, 1);
  const diagnostic = captured.records[0];
  assert.equal(diagnostic.event, "webui.identity_conflict");
  assert.equal(diagnostic.session_id, "session-1");
  assert.equal(diagnostic.turn_id, "turn-1");
  assert.equal(diagnostic.client_message_id, "client-1");
  assert.equal(diagnostic.incoming_client_message_id, "client-2");
  // 同一 turn+incoming 组合只降级一次；原身份不变
  registry.registerTurnIdentity("session-1", "turn-1", "client-2");
  assert.equal(captured.records.length, 1);
  assert.equal(registry.registerTurnIdentity("session-1", "turn-1", "client-1").key, identity.key);
  // 随后 markFirst 正常：原身份仍可标记里程碑
  assert.equal(registry.markFirst(identity, "webui.patch_received", "thinking", "origin"), true);
  assert.equal(captured.records.length, 2);
  assert.equal(captured.records[1].event, "webui.patch_received");
  assert.equal(captured.records[1].client_message_id, "client-1");
});

test("first visible kinds: thinking precedes answer, same patch may introduce both, terminal last", () => {
  const empty = { content: "", thinking: [] };
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(undefined, { message: { content: "", thinking: ["思考"], streaming: true } }),
    ["thinking"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, { message: { content: "回答", thinking: [], streaming: true } }),
    ["answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, { message: { content: "回答", thinking: ["思考"], streaming: true } }),
    ["thinking", "answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(undefined, { thinkingAppend: { blockIndex: 0, delta: "思" } }),
    ["thinking"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, { contentAppend: "回" }),
    ["answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(undefined, { contentAppend: "回", thinkingAppend: { blockIndex: 0, delta: "思" } }),
    ["thinking", "answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, { message: { content: "", thinking: [], streaming: false }, terminal: true }),
    ["terminal"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, { message: { content: "终", thinking: [], streaming: false }, terminal: true }),
    ["answer", "terminal"],
  );
  assert.deepEqual(mobileTurnFirstVisibleKinds(empty, { contentAppend: "" }), []);
  assert.deepEqual(mobileTurnFirstVisibleKinds(empty, { thinkingAppend: { blockIndex: 0, delta: "" } }), []);
});

test("thinking then answer across patches: only the newly visible kind is reported", () => {
  const previous = { content: "", thinking: ["思考"] };
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(previous, { message: { content: "回答", thinking: ["思考"], streaming: true } }),
    ["answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(previous, { contentAppend: "回" }),
    ["answer"],
  );
  const answerPrevious = { content: "回答", thinking: [""] };
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(answerPrevious, { thinkingAppend: { blockIndex: 0, delta: "思" } }),
    ["thinking"],
  );
});

test("corrections and continuation patches never re-report an already visible kind", () => {
  const previous = { content: "回答", thinking: ["思考"] };
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(previous, { message: { content: "回答纠正", thinking: ["思考"], streaming: true } }),
    [],
  );
  assert.deepEqual(mobileTurnFirstVisibleKinds(previous, { contentAppend: "续写" }), []);
  assert.deepEqual(mobileTurnFirstVisibleKinds(previous, { thinkingAppend: { blockIndex: 0, delta: "续" } }), []);
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(previous, { message: { content: "回答", thinking: ["思考"], streaming: false } }),
    ["terminal"],
  );
});

test("each milestone fires at most once per event and kind per turn", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const identity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.equal(registry.markFirst(identity, "webui.patch_received", "thinking", "receive-stream-patch"), true);
  assert.equal(registry.markFirst(identity, "webui.patch_received", "thinking", "receive-stream-patch"), false);
  assert.equal(registry.markFirst(identity, "webui.patch_received", "answer", "receive-stream-patch"), true);
  assert.equal(registry.markFirst(identity, "webui.patch_received", "answer", "receive-stream-patch"), false);
  assert.equal(registry.markFirst(identity, "webui.patch_received", "terminal", "receive-stream-patch"), true);
  registry.markFirst(identity, "webui.patch_applied", "thinking", "receive-stream-patch");
  registry.markFirst(identity, "webui.patch_applied", "answer", "receive-stream-patch");
  registry.markFirst(identity, "webui.react_committed", "thinking", "message-row");
  registry.markFirst(identity, "webui.react_committed", "answer", "message-row");
  registry.markFirst(identity, "webui.next_frame_ready", "thinking", "message-row-frame");
  registry.markFirst(identity, "webui.next_frame_ready", "answer", "message-row-frame");
  assert.equal(captured.records.length, 9);
  assert.deepEqual(
    captured.records.map((record) => `${record.event}:${record.kind}`),
    [
      "webui.patch_received:thinking",
      "webui.patch_received:answer",
      "webui.patch_received:terminal",
      "webui.patch_applied:thinking",
      "webui.patch_applied:answer",
      "webui.react_committed:thinking",
      "webui.react_committed:answer",
      "webui.next_frame_ready:thinking",
      "webui.next_frame_ready:answer",
    ],
  );
  assert.ok(captured.records.every((record) => record.client_message_id === "client-1"));
  assert.ok(captured.records.every((record) => record.session_id === "session-1"));
  assert.ok(captured.records.every((record) => record.turn_id === "turn-1"));
  assert.ok(captured.records.every((record) => typeof record.wall_ms === "number" && record.wall_ms > 0));
  assert.ok(captured.records.every((record) => typeof record.performance_ms === "number"));
});

test("old turn never blocks or pollutes a new turn's milestones", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const oldIdentity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  const newIdentity = registry.registerTurnIdentity("session-1", "turn-2", "client-2");
  for (const event of ["webui.patch_received", "webui.patch_applied", "webui.react_committed", "webui.next_frame_ready"]) {
    registry.markFirst(oldIdentity, event, "thinking", "origin");
    registry.markFirst(newIdentity, event, "answer", "origin");
  }
  assert.equal(captured.records.length, 8);
  assert.equal(registry.markFirst(oldIdentity, "webui.patch_received", "thinking", "origin"), false);
  const newMilestones = captured.records.filter((record) => record.turn_id === "turn-2");
  assert.equal(newMilestones.length, 4);
  assert.ok(newMilestones.every((record) => record.client_message_id === "client-2"));
  const oldMilestones = captured.records.filter((record) => record.turn_id === "turn-1");
  assert.ok(oldMilestones.every((record) => record.kind === "thinking"));
});

test("registry stays bounded and evicts the oldest turn", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const first = registry.registerTurnIdentity("session-1", "turn-0", "client-0");
  for (let index = 1; index < MOBILE_TURN_TRACE_MAX_TRACKED + 5; index += 1) {
    registry.registerTurnIdentity("session-1", `turn-${index}`, `client-${index}`);
  }
  assert.equal(registry.identityFor("session-1", "turn-0"), undefined);
  assert.equal(registry.identityFor("session-1", "turn-4"), undefined);
  assert.equal(registry.markFirst(first, "webui.patch_received", "thinking", "origin"), false);
  assert.ok(registry.identityFor("session-1", "turn-5") !== undefined);
  const reRegistered = registry.registerTurnIdentity("session-1", "turn-0", "client-0");
  const recorded = registry.markFirst(reRegistered, "webui.patch_received", "thinking", "origin");
  assert.equal(recorded, true);
});

test("console payload is a single [akashic-trace] line with fixed content-free fields", () => {
  const logs = [];
  const original = console.log;
  console.log = (message) => { logs.push(message); };
  try {
    mobileTurnTraceEmit({
      event: "webui.patch_received",
      session_id: "session-1",
      turn_id: "turn-1",
      client_message_id: "client-1",
      wall_ms: 1234,
      performance_ms: 56.78,
      kind: "thinking",
      origin: "receive-stream-patch",
    });
  } finally {
    console.log = original;
  }
  assert.equal(logs.length, 1);
  assert.match(logs[0], /^\[akashic-trace\] \{/);
  const payload = JSON.parse(logs[0].slice("[akashic-trace] ".length));
  assert.deepEqual(
    Object.keys(payload).sort(),
    ["client_message_id", "event", "kind", "origin", "performance_ms", "session_id", "turn_id", "wall_ms"],
  );
  assert.deepEqual(
    Object.values(payload).map((value) => typeof value).sort(),
    ["number", "number", "string", "string", "string", "string", "string", "string"],
  );
});

test("identity conflict diagnostic is a single content-free [akashic-trace] line", () => {
  const logs = [];
  const original = console.log;
  console.log = (message) => { logs.push(message); };
  try {
    mobileTurnTraceEmit({
      event: "webui.identity_conflict",
      session_id: "session-1",
      turn_id: "turn-1",
      client_message_id: "client-1",
      incoming_client_message_id: "client-2",
      wall_ms: 1234,
      performance_ms: 56.78,
      kind: "identity",
      origin: "turn-trace-registry",
    });
  } finally {
    console.log = original;
  }
  assert.equal(logs.length, 1);
  assert.match(logs[0], /^\[akashic-trace\] \{/);
  const payload = JSON.parse(logs[0].slice("[akashic-trace] ".length));
  assert.equal(payload.event, "webui.identity_conflict");
  assert.equal(payload.client_message_id, "client-1");
  assert.equal(payload.incoming_client_message_id, "client-2");
  assert.ok(!JSON.stringify(payload).includes("content"), "diagnostic must stay content-free");
});

test("receiveStreamPatch registers identity before publish; conflict path never throws", async () => {
  const receiver = mobileSource.match(/receiveStreamPatch\(next\) \{[\s\S]*?\n[ ]{6}\},\n[ ]{6}receiveStatePatch/);
  assert.ok(receiver, "mobile stream receiver must remain discoverable");
  assert.match(receiver[0], /registerTurnIdentity\([\s\S]*?streamStore\.publish/, "identity registered before publish");
  const traceSource = await readFile(
    new URL("./mobile-turn-trace.ts", import.meta.url),
    "utf8",
  );
  const registerRegion = traceSource.match(/registerTurnIdentity\([\s\S]*?\n  \}/);
  assert.ok(registerRegion, "registerTurnIdentity must remain discoverable");
  assert.doesNotMatch(registerRegion[0], /\bthrow\b/, "identity conflict must not throw");
  assert.match(registerRegion[0], /webui\.identity_conflict/);
});

test("stream patch parser keeps protocolVersion 3 and parses optional clientMessageId", () => {
  const parseRegion = mobileSource.match(/function parseMobileStreamPatch\(value: unknown\): MobileStreamPatch \{[\s\S]*?\n\}/);
  assert.ok(parseRegion, "parseMobileStreamPatch must remain discoverable");
  assert.match(parseRegion[0], /return \{\s*\n\s*protocolVersion: 3,/);
  assert.match(parseRegion[0], /clientMessageId: optionalString\(raw\.clientMessageId, "streamPatch\.clientMessageId"\),/);
  assert.match(mobileSource, /interface MobileStreamPatch \{[\s\S]*?clientMessageId\?: string;/);
});

test("receiveStreamPatch records received per kind after parse and applied only after publish", () => {
  const receiver = mobileSource.match(/receiveStreamPatch\(next\) \{[\s\S]*?\n[ ]{6}\},\n[ ]{6}receiveStatePatch/);
  assert.ok(receiver, "mobile stream receiver must remain discoverable");
  assert.match(receiver[0], /parseMobileStreamPatch\(next\)[\s\S]*?webui\.patch_received/);
  const snapshotFallbacks = receiver[0].split("requestSnapshot();");
  assert.ok(snapshotFallbacks.length >= 2, "snapshot-request fallbacks must exist");
  for (const fallback of snapshotFallbacks.slice(0, -1)) {
    assert.doesNotMatch(fallback, /webui\.patch_applied/);
  }
  assert.match(receiver[0], /streamStore\.publishFrame\([\s\S]*?streamStore\.publishImmediate\([\s\S]*?webui\.patch_applied/);
  assert.match(receiver[0], /webui\.patch_received[\s\S]*?webui\.patch_applied/);
  // 每个 kind 分别 mark received 与 applied，变量是 kind 而非固定字符串
  assert.match(receiver[0], /markFirst\(traceIdentity, "webui\.patch_received", kind, "receive-stream-patch"\)/);
  assert.match(receiver[0], /markFirst\(traceIdentity, "webui\.patch_applied", kind, "receive-stream-patch"\)/);
  // 无 kind 不得写 patch 占位
  assert.doesNotMatch(receiver[0], /\?\? "patch"/);
});

test("message row records react_committed per kind via useLayoutEffect and one rAF per first milestone", () => {
  const rowRegion = mobileSource.match(/const MobileMessageRow = React\.memo\(function MobileMessageRow\([\s\S]*?const MobilePlainMessageView = React\.memo/);
  assert.ok(rowRegion, "mobile message row must remain discoverable");
  assert.match(rowRegion[0], /useSyncExternalStore\(subscribe, getSnapshot, getSnapshot\)[\s\S]*?useLayoutEffect/);
  assert.match(rowRegion[0], /mobileTurnDomVisibleKinds\(source\)/);
  assert.match(rowRegion[0], /webui\.react_committed/);
  assert.match(rowRegion[0], /webui\.next_frame_ready/);
  assert.match(rowRegion[0], /requestAnimationFrame/);
  assert.match(rowRegion[0], /cancelAnimationFrame/);
  assert.equal((rowRegion[0].match(/requestAnimationFrame/g) ?? []).length, 1);
  // 每个 kind 分别 mark react_committed 与 next_frame_ready
  assert.match(rowRegion[0], /markFirst\(traceIdentity, "webui\.react_committed", kind, "message-row"\)/);
  assert.match(rowRegion[0], /markFirst\(traceIdentity, "webui\.next_frame_ready", kind, "message-row-frame"\)/);
  // 同一次 commit 至多安排一个 rAF：已有挂起帧则并入挂起集合
  assert.match(rowRegion[0], /traceFrameRef\.current\.kinds\.push\(\.\.\.committedKinds\)/);
  // turn 切换取消上一 turn 的挂起帧，避免跨 turn 误报
  assert.match(rowRegion[0], /traceFrameRef\.current\.key !== traceIdentity\.key[\s\S]*?cancelAnimationFrame/);
  // 帧事件只叫 next_frame_ready，不宣称 paint
  assert.match(rowRegion[0], /requestAnimationFrame\([\s\S]*?webui\.next_frame_ready/);
  assert.doesNotMatch(rowRegion[0], /webui\.\w*paint\w*/i);
});

test("presentation reuse comparison includes attachment filename", () => {
  const matches = mobileSource.match(/function mobileMessagePresentationMatches\(previous: MobileMessage, next: MobileMessage\) \{[\s\S]*?\n\}/);
  assert.ok(matches, "mobileMessagePresentationMatches must remain discoverable");
  assert.match(matches[0], /previous\.attachments\.length !== next\.attachments\.length/);
  assert.match(matches[0], /attachment\.id === candidate\.id[\s\S]*?attachment\.filename === candidate\.filename/);
  assert.match(matches[0], /attachment\.contentType === candidate\.contentType/);
  assert.match(matches[0], /attachment\.sizeBytes === candidate\.sizeBytes/);
  assert.match(matches[0], /attachment\.transferredBytes === candidate\.transferredBytes/);
  assert.match(matches[0], /attachment\.state === candidate\.state/);
  assert.match(matches[0], /attachment\.canRemove === candidate\.canRemove/);
  assert.match(matches[0], /attachment\.contentUrl === candidate\.contentUrl/);
});

test("identityFor resolves a row rendered before registration as soon as the entry exists", () => {
  const registry = new MobileTurnTraceRegistry(() => {});
  // 先渲染后注册：行先以 source.id 查 identityFor 得到 undefined
  assert.equal(registry.identityFor("session-1", "turn-1"), undefined);
  registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  // 下一次 render 读当前 registry：同一 key 立即解析
  const resolved = registry.identityFor("session-1", "turn-1");
  assert.equal(resolved.clientMessageId, "client-1");
  assert.equal(resolved.key, registry.identityFor("session-1", "turn-1").key);
});

test("markFirst emits the entry's current client_message_id, never a stale captured identity", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  // missing 阶段注册并已上报一次：当时确无 id，诚实记 missing
  const early = registry.registerTurnIdentity("session-1", "turn-1", undefined);
  registry.markFirst(early, "webui.patch_received", "thinking", "receive-stream-patch");
  assert.equal(captured.records[0].client_message_id, MOBILE_TURN_MISSING);
  // 随后补齐真实 client id：后续里程碑（含旧 rAF 闭包持有的同一快照）都发当前 id
  registry.registerTurnIdentity("session-1", "turn-1", "client-9");
  assert.equal(registry.markFirst(early, "webui.react_committed", "thinking", "message-row"), true);
  assert.equal(registry.markFirst(early, "webui.next_frame_ready", "thinking", "message-row-frame"), true);
  assert.equal(captured.records[1].event, "webui.react_committed");
  assert.equal(captured.records[1].client_message_id, "client-9");
  assert.equal(captured.records[2].event, "webui.next_frame_ready");
  assert.equal(captured.records[2].client_message_id, "client-9");
  assert.equal(registry.identityFor("session-1", "turn-1").clientMessageId, "client-9");
});

test("terminal id migration binds the canonical id to the same entry so committed and frame-ready share the identity", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  // receiver 以临时 assistant id 注册 primary
  const receiverIdentity = registry.registerTurnIdentity(
    "session-1",
    parseMobileTurnId("assistant:turn-tmp"),
    "client-1",
  );
  assert.equal(receiverIdentity.turnId, "turn-tmp");
  // terminal publish 前：把 canonical messageId 绑定为同一 registry primary 的别名
  const bound = registry.bindMessageIdentity("session-1", "message:canonical", receiverIdentity);
  assert.equal(bound.clientMessageId, "client-1");
  // 行从新 source.id 解析仍命中同一 primary（同一 clientMessageId 与里程碑集合）
  const rowIdentity = registry.identityForMessage("session-1", "message:canonical");
  assert.ok(rowIdentity !== undefined, "canonical source.id must resolve through the bound alias");
  assert.equal(rowIdentity.clientMessageId, "client-1");
  assert.equal(rowIdentity.key, bound.key);
  // react_committed 与 next_frame_ready 均携同一 session turn clientMessageId
  registry.markFirst(rowIdentity, "webui.react_committed", "terminal", "message-row");
  registry.markFirst(rowIdentity, "webui.next_frame_ready", "terminal", "message-row-frame");
  assert.equal(captured.records.length, 2);
  assert.equal(captured.records[0].event, "webui.react_committed");
  assert.equal(captured.records[1].event, "webui.next_frame_ready");
  assert.equal(captured.records[0].client_message_id, "client-1");
  assert.equal(captured.records[1].client_message_id, "client-1");
  assert.equal(captured.records[0].turn_id, "turn-tmp");
  assert.equal(captured.records[1].turn_id, "turn-tmp");
  // 同一 entry：旧键下的里程碑与新键共享，terminal 不重复上报
  assert.equal(registry.markFirst(receiverIdentity, "webui.react_committed", "terminal", "message-row"), false);
  // 临时 assistant id 走 turn primary；未绑定的 canonical id 不猜测、不解析
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-tmp").key, receiverIdentity.key);
  assert.equal(registry.identityForMessage("session-1", "message:unbound"), undefined);
  // 另一 turn 的 primary 不受本别名影响
  const other = registry.registerTurnIdentity("session-1", "turn-own", "client-own");
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-own").key, other.key);
  assert.equal(registry.identityForMessage("session-1", "message:canonical").key, receiverIdentity.key);
});

test("same session two turns migrate to two distinct canonical ids, both live and isolated", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  // turn-1 流式 → terminal：绑定 canonical-1
  const first = registry.registerTurnIdentity("session-1", parseMobileTurnId("assistant:turn-1"), "client-1");
  registry.bindMessageIdentity("session-1", "message:canonical-1", first);
  // turn-2 流式 → terminal：绑定 canonical-2；两个 primary 同时存在
  const second = registry.registerTurnIdentity("session-1", parseMobileTurnId("assistant:turn-2"), "client-2");
  registry.bindMessageIdentity("session-1", "message:canonical-2", second);
  const row1 = registry.identityForMessage("session-1", "message:canonical-1");
  const row2 = registry.identityForMessage("session-1", "message:canonical-2");
  assert.ok(row1 !== undefined && row2 !== undefined, "both canonical rows must resolve simultaneously");
  assert.notEqual(row1.key, row2.key);
  assert.equal(row1.turnId, "turn-1");
  assert.equal(row2.turnId, "turn-2");
  assert.equal(row1.clientMessageId, "client-1");
  assert.equal(row2.clientMessageId, "client-2");
  // 旧行（temp assistant id）不得命中新 turn：各自解析回自己的 primary
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-1").key, row1.key);
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-2").key, row2.key);
  // 各自里程碑独立：turn-1 的 terminal 上报不占用 turn-2 的 entry
  registry.markFirst(row1, "webui.react_committed", "terminal", "message-row");
  registry.markFirst(row2, "webui.react_committed", "terminal", "message-row");
  registry.markFirst(row2, "webui.react_committed", "answer", "message-row");
  assert.equal(registry.markFirst(row1, "webui.react_committed", "terminal", "message-row"), false);
  assert.equal(registry.markFirst(row2, "webui.react_committed", "terminal", "message-row"), false);
  assert.equal(captured.records.length, 3);
  assert.deepEqual(
    captured.records.map((record) => `${record.turn_id}:${record.client_message_id}:${record.kind}`).sort(),
    ["turn-1:client-1:terminal", "turn-2:client-2:answer", "turn-2:client-2:terminal"],
  );
  // 旧行继续上报 terminal 后的 answer 里程碑不会串进 turn-2
  assert.equal(registry.markFirst(row1, "webui.react_committed", "answer", "message-row"), true);
  assert.equal(captured.records.filter((record) => record.turn_id === "turn-2").length, 2);
});

test("binding an alias already owned by another live primary fails loud and never silently rebinds", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const first = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  const second = registry.registerTurnIdentity("session-1", "turn-2", "client-2");
  const bound = registry.bindMessageIdentity("session-1", "message:canonical", first);
  assert.equal(bound.key, first.key);
  // 同一 alias 已指向仍存 primary：冲突诊断 + 拒绝改绑，返回既有绑定
  const refused = registry.bindMessageIdentity("session-1", "message:canonical", second);
  assert.equal(refused.key, first.key, "alias must keep the first live owner");
  assert.equal(registry.identityForMessage("session-1", "message:canonical").key, first.key);
  assert.equal(captured.records.length, 1);
  assert.equal(captured.records[0].event, "webui.identity_conflict");
  assert.equal(captured.records[0].kind, "alias");
  assert.equal(captured.records[0].session_id, "session-1");
  assert.equal(captured.records[0].turn_id, "turn-1");
  assert.equal(captured.records[0].client_message_id, "client-1");
  assert.equal(captured.records[0].incoming_client_message_id, "message:canonical");
  assert.ok(!JSON.stringify(captured.records[0]).includes("content"), "conflict must stay content-free");
  // 同一冲突只发一次诊断
  registry.bindMessageIdentity("session-1", "message:canonical", second);
  assert.equal(captured.records.length, 1);
  // 原 primary 不受影响，仍可标记里程碑；第二 turn 的 primary 用另一个 alias 正常绑定
  assert.equal(registry.markFirst(first, "webui.react_committed", "terminal", "message-row"), true);
  const secondBound = registry.bindMessageIdentity("session-1", "message:canonical-2", second);
  assert.equal(secondBound.key, second.key);
  assert.equal(registry.identityForMessage("session-1", "message:canonical-2").key, second.key);
});

test("evicting a primary drops its aliases; stale aliases resolve undefined and clear on read", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const doomed = registry.registerTurnIdentity("session-1", "turn-0", "client-0");
  registry.bindMessageIdentity("session-1", "message:canonical-0", doomed);
  for (let index = 1; index < MOBILE_TURN_TRACE_MAX_TRACKED + 2; index += 1) {
    registry.registerTurnIdentity("session-1", `turn-${index}`, `client-${index}`);
  }
  // turn-0 被有界淘汰：identityFor 与 identityForMessage 均返回 undefined
  assert.equal(registry.identityFor("session-1", "turn-0"), undefined);
  assert.equal(registry.identityForMessage("session-1", "message:canonical-0"), undefined);
  // stale alias 已清除：同 id 可重新绑定到新 primary，且不报冲突
  const fresh = registry.registerTurnIdentity("session-1", "turn-new", "client-new");
  const rebound = registry.bindMessageIdentity("session-1", "message:canonical-0", fresh);
  assert.equal(rebound.key, fresh.key);
  assert.equal(captured.records.length, 0, "rebind after eviction must not report conflict");
  assert.equal(registry.identityForMessage("session-1", "message:canonical-0").key, fresh.key);
  // 未淘汰的 primary（turn-63 仍在追踪）及其别名不受影响
  const survivor = registry.identityFor("session-1", "turn-63");
  assert.ok(survivor !== undefined);
  registry.bindMessageIdentity("session-1", "message:canonical-63", survivor);
  assert.equal(registry.identityForMessage("session-1", "message:canonical-63").key, survivor.key);
  assert.equal(registry.markFirst(survivor, "webui.react_committed", "terminal", "message-row"), true);
  // 别名本身不计入 tracked-turn 上限：大量绑定不淘汰任何 primary
  const alive = registry.registerTurnIdentity("session-1", "turn-alive", "client-alive");
  for (let index = 0; index < MOBILE_TURN_TRACE_MAX_TRACKED * 2; index += 1) {
    registry.bindMessageIdentity("session-1", `message:extra-${index}`, alive);
  }
  assert.ok(registry.tracks(alive.key), "aliases must not count toward the tracked-turn cap");
  assert.equal(registry.identityForMessage("session-1", "message:extra-0").key, alive.key);
  assert.equal(registry.identityForMessage("session-1", "message:extra-127").key, alive.key);
  assert.equal(registry.identityForMessage("session-1", "message:canonical-63").key, survivor.key);
});

test("binding an evicted source degrades observability without blocking terminal flow", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const doomed = registry.registerTurnIdentity("session-1", "turn-0", "client-0");
  registry.bindMessageIdentity("session-1", "message:canonical-0", doomed);
  // 有界淘汰真实移除 primary 与其 aliases（evictPrimary 是唯一删除 owner）
  for (let index = 1; index < MOBILE_TURN_TRACE_MAX_TRACKED + 1; index += 1) {
    registry.registerTurnIdentity("session-1", `turn-${index}`, `client-${index}`);
  }
  assert.equal(registry.identityForMessage("session-1", "message:canonical-0"), undefined);
  // 源 primary 已被淘汰：只降级观测；调用方仍可继续 terminal publish/render。
  let terminalContinued = false;
  const bound = registry.bindMessageIdentity("session-1", "message:dead-source", doomed);
  terminalContinued = true;
  assert.equal(bound, undefined);
  assert.equal(terminalContinued, true);
  assert.equal(captured.records.length, 1);
  assert.equal(captured.records[0].event, "webui.identity_conflict");
  assert.equal(captured.records[0].kind, "stale_source");
  assert.equal(captured.records[0].turn_id, "turn-0");
  assert.equal(captured.records[0].client_message_id, "client-0");
  assert.ok(!JSON.stringify(captured.records[0]).includes("content"));
  // 同一退化只发一次，避免纯观测噪声淹没 logcat。
  registry.bindMessageIdentity("session-1", "message:dead-source", doomed);
  assert.equal(captured.records.length, 1);
  assert.equal(registry.identityForMessage("session-1", "message:dead-source"), undefined);
});

test("a throwing trace sink cannot block identity binding or milestone delivery", () => {
  const diagnostics = [];
  const originalConsoleError = console.error;
  console.error = (line) => diagnostics.push(String(line));
  try {
    const registry = new MobileTurnTraceRegistry(() => {
      throw new Error("trace sink failed");
    });
    const first = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
    const second = registry.registerTurnIdentity("session-1", "turn-2", "client-2");

    // 1. milestone 与 live alias 冲突都只降级观测，业务 owner 仍继续。
    assert.doesNotThrow(() => {
      assert.equal(registry.markFirst(first, "webui.react_committed", "terminal", "message-row"), true);
      registry.bindMessageIdentity("session-1", "message:canonical", first);
      assert.equal(registry.bindMessageIdentity("session-1", "message:canonical", second).key, first.key);
    });

    // 2. source 被有界淘汰后的 terminal alias 绑定同样不得被 sink 反向中断。
    for (let index = 3; index < MOBILE_TURN_TRACE_MAX_TRACKED + 3; index += 1) {
      registry.registerTurnIdentity("session-1", `turn-${index}`, `client-${index}`);
    }
    assert.doesNotThrow(() => {
      assert.equal(registry.bindMessageIdentity("session-1", "message:stale", first), undefined);
    });
    assert.ok(diagnostics.length >= 3);
    for (const line of diagnostics) {
      const payload = JSON.parse(line.slice("[akashic-trace] ".length));
      assert.equal(payload.event, "webui.trace_sink_error");
      assert.equal(payload.error_type, "Error");
      assert.ok(!line.includes("content"));
      assert.ok(!line.includes("trace sink failed"));
    }
  } finally {
    console.error = originalConsoleError;
  }
});

test("identityForMessage resolves a row rendered before registration once entry or alias exists", () => {
  const registry = new MobileTurnTraceRegistry(() => {});
  // 先渲染后注册：临时 id 与 canonical id 初始均不可解析
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-1"), undefined);
  assert.equal(registry.identityForMessage("session-1", "message:canonical"), undefined);
  // 注册 primary 后：临时 assistant id 立即解析
  const identity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.equal(registry.identityForMessage("session-1", "assistant:turn-1").clientMessageId, "client-1");
  // 绑定别名后：canonical id 立即解析到同一 primary
  registry.bindMessageIdentity("session-1", "message:canonical", identity);
  assert.equal(registry.identityForMessage("session-1", "message:canonical").clientMessageId, "client-1");
  assert.equal(
    registry.identityForMessage("session-1", "message:canonical").key,
    registry.identityForMessage("session-1", "assistant:turn-1").key,
  );
});

test("receiveStreamPatch binds a migrated canonical id to the registered entry before publishing", async () => {
  const receiver = mobileSource.match(/receiveStreamPatch\(next\) \{[\s\S]*?\n[ ]{6}\},\n[ ]{6}receiveStatePatch/);
  assert.ok(receiver, "mobile stream receiver must remain discoverable");
  assert.match(receiver[0], /nextMessage\.id !== parsed\.messageId[\s\S]*?bindMessageIdentity\(/);
  assert.match(receiver[0], /bindMessageIdentity\([\s\S]*?streamStore\.publish/);
  // 假绿迁移路径必须整体移除：canonical id 不得再被 parse 成 undefined 后 bind
  assert.doesNotMatch(receiver[0], /bindTurnIdentity/);
  assert.doesNotMatch(receiver[0], /parseMobileTurnId\(nextMessage\.id\)/);
  const traceSource = await readFile(
    new URL("./mobile-turn-trace.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(traceSource, /bindTurnIdentity/, "registry must not keep bindTurnIdentity");
});

test("message row resolves the trace identity per render instead of caching it behind stable deps", () => {
  const rowRegion = mobileSource.match(/const MobileMessageRow = React\.memo\(function MobileMessageRow\([\s\S]*?const MobilePlainMessageView = React\.memo/);
  assert.ok(rowRegion, "mobile message row must remain discoverable");
  assert.match(rowRegion[0], /const traceIdentity = source\.role === "assistant"[\s\S]*?mobileTurnTrace\.identityForMessage\(source\.sessionId, source\.id\)/);
  assert.doesNotMatch(rowRegion[0], /useMemo\(\(\) => \([\s\S]*?identityFor/);
});
