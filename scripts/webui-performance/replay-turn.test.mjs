import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";

import { loadReplayTurn } from "./replay-turn.mjs";

test("replay turn loader validates exported stages and tool calls", () => {
  const root = mkdtempSync(resolve(tmpdir(), "akashic-replay-turn-"));
  const path = resolve(root, "turn.json");
  writeFileSync(path, JSON.stringify([
    { role: "user", content: "question" },
    {
      role: "assistant",
      content: "answer",
      tool_chain: [{
        text: "stage text",
        reasoning_content: "thinking",
        calls: [{
          call_id: "call-1", name: "probe", status: "success",
          arguments: { value: 1 }, final_arguments: { value: 1 }, result: "done",
        }],
      }],
    },
  ]));
  try {
    assert.deepEqual(loadReplayTurn(path), {
      content: "answer",
      stages: [{
        text: "stage text",
        reasoning: "thinking",
        calls: [{
          callId: "call-1", name: "probe", status: "success",
          arguments: { value: 1 }, finalArguments: { value: 1 }, result: "done",
        }],
      }],
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("replay turn loader rejects malformed tool arguments", () => {
  const root = mkdtempSync(resolve(tmpdir(), "akashic-replay-turn-invalid-"));
  const path = resolve(root, "turn.json");
  writeFileSync(path, JSON.stringify([{
    role: "assistant", content: "answer",
    tool_chain: [{ calls: [{
      call_id: "call-1", name: "probe", status: "success",
      arguments: [], final_arguments: {}, result: "done",
    }] }],
  }]));
  try {
    assert.throws(() => loadReplayTurn(path), /invalid arguments/u);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
