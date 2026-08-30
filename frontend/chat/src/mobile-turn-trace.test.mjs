import assert from "node:assert/strict";
import test from "node:test";

import {
  MOBILE_TURN_MISSING,
  MOBILE_TURN_TRACE_MAX_TRACKED,
  MobileTurnTraceRegistry,
  mobileTurnFirstVisibleKinds,
  mobileTurnTraceEmit,
  parseMobileTurnId,
} from "./mobile-turn-trace.ts";

function captureSink() {
  const records = [];
  return {
    records,
    sink(record) {
      records.push(record);
    },
  };
}

test("turn ids parse only from the non-empty assistant message contract", () => {
  assert.equal(parseMobileTurnId("assistant:turn-01J"), "turn-01J");
  assert.equal(parseMobileTurnId("assistant:"), undefined);
  assert.equal(parseMobileTurnId("user:turn-01J"), undefined);
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
    mobileTurnFirstVisibleKinds(undefined, {
      message: { content: "回答", thinking: ["思考"], streaming: true },
    }),
    ["thinking", "answer"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, {
      message: { content: "回答", thinking: [], streaming: false },
    }),
    ["answer", "terminal"],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(
      { content: "回答", thinking: ["思考"] },
      { contentAppend: "续写", thinkingAppend: { blockIndex: 0, delta: "续" } },
    ),
    [],
  );
  assert.deepEqual(
    mobileTurnFirstVisibleKinds(empty, {
      contentAppend: "回",
      thinkingAppend: { blockIndex: 0, delta: "思" },
    }),
    ["thinking", "answer"],
  );
});

test("identity registration fills missing data and reports each conflict once", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);

  const missing = registry.registerTurnIdentity("session-1", "turn-1", undefined);
  assert.equal(missing.clientMessageId, MOBILE_TURN_MISSING);

  const filled = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.equal(filled.key, missing.key);
  assert.equal(filled.clientMessageId, "client-1");

  const conflicting = registry.registerTurnIdentity("session-1", "turn-1", "client-2");
  registry.registerTurnIdentity("session-1", "turn-1", "client-2");

  assert.equal(conflicting.clientMessageId, "client-1");
  assert.equal(captured.records.length, 1);
  assert.deepEqual(
    {
      event: captured.records[0].event,
      turn: captured.records[0].turn_id,
      current: captured.records[0].client_message_id,
      incoming: captured.records[0].incoming_client_message_id,
      kind: captured.records[0].kind,
    },
    {
      event: "webui.identity_conflict",
      turn: "turn-1",
      current: "client-1",
      incoming: "client-2",
      kind: "identity",
    },
  );
});

test("milestones are unique per event and kind and use the entry's current identity", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const early = registry.registerTurnIdentity("session-1", "turn-1", undefined);

  assert.equal(
    registry.markFirst(early, "webui.patch_received", "thinking", "receive-stream-patch"),
    true,
  );
  registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  assert.equal(
    registry.markFirst(early, "webui.react_committed", "thinking", "message-row"),
    true,
  );
  assert.equal(
    registry.markFirst(early, "webui.react_committed", "thinking", "message-row"),
    false,
  );
  assert.equal(
    registry.markFirst(early, "webui.react_committed", "answer", "message-row"),
    true,
  );

  const other = registry.registerTurnIdentity("session-1", "turn-2", "client-2");
  assert.equal(
    registry.markFirst(other, "webui.react_committed", "thinking", "message-row"),
    true,
  );

  assert.deepEqual(
    captured.records.map((record) => [record.turn_id, record.client_message_id, record.kind]),
    [
      ["turn-1", MOBILE_TURN_MISSING, "thinking"],
      ["turn-1", "client-1", "thinking"],
      ["turn-1", "client-1", "answer"],
      ["turn-2", "client-2", "thinking"],
    ],
  );
});

test("canonical message ids resolve through aliases without guessing", () => {
  const registry = new MobileTurnTraceRegistry(() => {});
  const identity = registry.registerTurnIdentity("session-1", "turn-1", "client-1");

  assert.equal(registry.identityForMessage("session-1", "message:canonical"), undefined);
  const bound = registry.bindMessageIdentity("session-1", "message:canonical", identity);

  assert.equal(bound.key, identity.key);
  assert.equal(
    registry.identityForMessage("session-1", "assistant:turn-1").key,
    identity.key,
  );
  assert.equal(
    registry.identityForMessage("session-1", "message:canonical").key,
    identity.key,
  );
  assert.equal(registry.identityForMessage("session-1", "message:unbound"), undefined);
});

test("a live alias keeps its first owner and reports a competing owner once", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const first = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
  const second = registry.registerTurnIdentity("session-1", "turn-2", "client-2");

  registry.bindMessageIdentity("session-1", "message:canonical", first);
  const refused = registry.bindMessageIdentity("session-1", "message:canonical", second);
  registry.bindMessageIdentity("session-1", "message:canonical", second);

  assert.equal(refused.key, first.key);
  assert.equal(
    registry.identityForMessage("session-1", "message:canonical").key,
    first.key,
  );
  assert.equal(captured.records.length, 1);
  assert.equal(captured.records[0].kind, "alias");
  assert.equal(captured.records[0].turn_id, "turn-1");

  const secondAlias = registry.bindMessageIdentity(
    "session-1",
    "message:canonical-2",
    second,
  );
  assert.equal(secondAlias.key, second.key);
});

test("bounded eviction removes aliases and stale sources degrade without blocking", () => {
  const captured = captureSink();
  const registry = new MobileTurnTraceRegistry(captured.sink);
  const doomed = registry.registerTurnIdentity("session-1", "turn-0", "client-0");
  registry.bindMessageIdentity("session-1", "message:canonical-0", doomed);

  for (let index = 1; index <= MOBILE_TURN_TRACE_MAX_TRACKED; index += 1) {
    registry.registerTurnIdentity("session-1", `turn-${index}`, `client-${index}`);
  }

  assert.equal(registry.identityFor("session-1", "turn-0"), undefined);
  assert.equal(registry.identityForMessage("session-1", "message:canonical-0"), undefined);
  assert.equal(
    registry.bindMessageIdentity("session-1", "message:stale", doomed),
    undefined,
  );
  assert.equal(captured.records.length, 1);
  assert.equal(captured.records[0].kind, "stale_source");

  const survivor = registry.identityFor("session-1", `turn-${MOBILE_TURN_TRACE_MAX_TRACKED}`);
  for (let index = 0; index < MOBILE_TURN_TRACE_MAX_TRACKED * 2; index += 1) {
    registry.bindMessageIdentity("session-1", `message:extra-${index}`, survivor);
  }
  assert.equal(registry.tracks(survivor.key), true);
});

test("a failing observation sink cannot block identity or milestone state", () => {
  const diagnostics = [];
  const originalConsoleError = console.error;
  console.error = (line) => diagnostics.push(String(line));
  try {
    const registry = new MobileTurnTraceRegistry(() => {
      throw new Error("sensitive failure text");
    });
    const first = registry.registerTurnIdentity("session-1", "turn-1", "client-1");
    const second = registry.registerTurnIdentity("session-1", "turn-2", "client-2");

    assert.equal(
      registry.markFirst(first, "webui.react_committed", "terminal", "message-row"),
      true,
    );
    registry.bindMessageIdentity("session-1", "message:canonical", first);
    assert.equal(
      registry.bindMessageIdentity("session-1", "message:canonical", second).key,
      first.key,
    );

    assert.ok(diagnostics.length >= 2);
    for (const line of diagnostics) {
      assert.match(line, /^\[akashic-trace\] \{/);
      const payload = JSON.parse(line.slice("[akashic-trace] ".length));
      assert.equal(payload.event, "webui.trace_sink_error");
      assert.equal(payload.error_type, "Error");
      assert.ok(!line.includes("sensitive failure text"));
      assert.ok(!line.includes("content"));
    }
  } finally {
    console.error = originalConsoleError;
  }
});

test("the default sink emits one content-free trace line", () => {
  const logs = [];
  const originalConsoleLog = console.log;
  console.log = (line) => logs.push(String(line));
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
    console.log = originalConsoleLog;
  }

  assert.equal(logs.length, 1);
  assert.match(logs[0], /^\[akashic-trace\] \{/);
  const payload = JSON.parse(logs[0].slice("[akashic-trace] ".length));
  assert.deepEqual(Object.keys(payload).sort(), [
    "client_message_id",
    "event",
    "kind",
    "origin",
    "performance_ms",
    "session_id",
    "turn_id",
    "wall_ms",
  ]);
});
