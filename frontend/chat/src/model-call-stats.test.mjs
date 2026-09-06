import assert from "node:assert/strict";
import test from "node:test";
import { formatModelCallStats, loadMobileModelCallStats, readModelCallStats, receiveMobileModelCallStats, selectModelCall } from "./model-call-stats.ts";

const stats = {
  call_record_id: "call", model: "fixture", state: "success", first_token_ms: 400, duration_ms: 1400,
  usage: { output_tokens: 12, request_count: 1, covered_request_count: 1, coverage: "exact" },
};
const output = (seq, call) => ({ id: `m${seq}`, seq, session_id: "s", body: {
  kind: "output", finish: "complete", parts: [{ kind: "model.facts", value: { call_record_id: call, thinking: null } }],
} });
const activity = (call) => ({ session_id: "s", source: "conversation", handle: "h", active: true,
  preview: { message_id: "draft", call_record_id: call, text: "", thinking: "" } });

test("reconnect uses persisted timing, partial or missing usage never produces an estimated rate", () => {
  assert.equal(formatModelCallStats(readModelCallStats(stats, "call"), false), "首 token 0.4s · 12.0 tok/s");
  for (const usage of [null, { ...stats.usage, coverage: "partial" }, { ...stats.usage, output_tokens: null }]) {
    assert.equal(formatModelCallStats({ ...stats, usage }, false), "首 token 0.4s · 耗时 1.4s");
  }
  assert.equal(formatModelCallStats({ ...stats, first_token_ms: null }, false), "耗时 1.4s");
  assert.equal(formatModelCallStats({ ...stats, first_token_ms: null, duration_ms: null, usage: null }, false), "暂无耗时数据");
  assert.equal(formatModelCallStats({ ...stats, state: "started", first_token_ms: null, duration_ms: null, usage: null }, true), "等待首 token…");
  assert.match(formatModelCallStats({ ...stats, state: "unknown", usage: null }, false), /用量未结算/u);
  assert.doesNotMatch(formatModelCallStats({ ...stats, duration_ms: 400 }, false), /tok\/s/u);
});

test("stats boundary rejects the wrong call, invalid time, counts, coverage and missing fields", () => {
  for (const value of [
    { ...stats, call_record_id: "old-call" }, { ...stats, first_token_ms: NaN },
    { ...stats, duration_ms: 100 }, { ...stats, duration_ms: undefined }, { ...stats, state: "done" },
    { ...stats, usage: {} }, { ...stats, usage: { ...stats.usage, covered_request_count: 2 } },
    { ...stats, usage: { ...stats.usage, output_tokens: -1 } },
    { ...stats, usage: { ...stats.usage, coverage: "estimated" } },
  ]) assert.throws(() => readModelCallStats(value, "call"), /无效/u);
});

test("active call replaces historical stats and a new empty draft cannot inherit the previous call", () => {
  const messages = [output(1, "old"), output(4, "call")];
  assert.deepEqual(selectModelCall(messages, []), { callId: "call", active: false });
  assert.deepEqual(selectModelCall(messages, [activity("live")]), { callId: "live", active: true });
  assert.deepEqual(selectModelCall(messages, [activity(null)]), { callId: null, active: true });
  assert.deepEqual(selectModelCall(messages, [{ ...activity("cancelled"), active: false }]), { callId: "call", active: false });
  assert.deepEqual(selectModelCall([], []), { callId: null, active: false });
});

test("native read correlates by request and ignores late replies after a session switch", async () => {
  const original = globalThis.window;
  const requests = [];
  globalThis.window = { AkashicNative: { readModelCallStats: (...args) => requests.push(args) } };
  try {
    const old = new AbortController();
    const first = loadMobileModelCallStats("old-call", old.signal);
    const aborted = assert.rejects(first, /page changed/u);
    old.abort(new Error("page changed"));
    await aborted;
    const current = loadMobileModelCallStats("call", new AbortController().signal);
    receiveMobileModelCallStats(requests[0][0], { ...stats, call_record_id: "old-call" });
    receiveMobileModelCallStats(requests[1][0], stats);
    assert.deepEqual(await current, stats);
    const invalid = loadMobileModelCallStats("call", new AbortController().signal);
    const rejected = assert.rejects(invalid, /无效/u);
    receiveMobileModelCallStats(requests[2][0], { ...stats, call_record_id: "wrong-call" });
    await rejected;
  } finally {
    globalThis.window = original;
  }
});
