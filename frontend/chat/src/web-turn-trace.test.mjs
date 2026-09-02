import assert from "node:assert/strict";
import test from "node:test";

import {
  WEB_TURN_TRACE_MAX_TRACKED,
  WebTurnTraceRegistry,
} from "./web-turn-trace.ts";

test("desktop trace joins transport, projection, React commit and next frame once", () => {
  const records = [];
  const trace = new WebTurnTraceRegistry((record) => records.push(record));

  trace.observeFrame("web:one", "turn-1", "answer");
  trace.observeFrame("web:one", "turn-1", "answer");
  trace.markProjection("turn-1");
  trace.markProjection("turn-1");
  const kinds = trace.markReactCommit("turn-1");
  trace.markNextFrame("turn-1", kinds);

  assert.deepEqual(records.map(({ event, kind }) => [event, kind]), [
    ["webui.frame_received", "answer"],
    ["webui.projection_published", "answer"],
    ["webui.react_committed", "answer"],
    ["webui.next_frame_ready", "answer"],
  ]);
  assert.equal(records.some((record) => "content" in record), false);
});

test("desktop trace keeps independent thinking, answer and terminal lanes", () => {
  const trace = new WebTurnTraceRegistry(() => {});
  for (const kind of ["thinking", "answer", "terminal"]) {
    trace.observeFrame("web:one", "turn-1", kind);
  }
  trace.markProjection("turn-1");
  const kinds = trace.markReactCommit("turn-1");
  trace.markNextFrame("turn-1", kinds);

  assert.equal(trace.snapshot().length, 12);
});

test("desktop trace registry evicts old turns at its fixed bound", () => {
  const trace = new WebTurnTraceRegistry(() => {});
  for (let index = 0; index <= WEB_TURN_TRACE_MAX_TRACKED; index += 1) {
    trace.observeFrame("web:one", `turn-${index}`, "answer");
  }

  trace.markProjection("turn-0");
  trace.markProjection(`turn-${WEB_TURN_TRACE_MAX_TRACKED}`);
  const projections = trace.snapshot().filter(({ event }) => event === "webui.projection_published");
  assert.deepEqual(projections.map(({ turn_id }) => turn_id), [`turn-${WEB_TURN_TRACE_MAX_TRACKED}`]);
  assert.ok(trace.snapshot().length <= WEB_TURN_TRACE_MAX_TRACKED * 12);
});
