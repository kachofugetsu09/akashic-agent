import assert from "node:assert/strict";
import test from "node:test";

import {
  desktopMessages,
  mobileSnapshot,
  mobileStreamPatch,
  mobileTerminalPatch,
} from "./fixtures.mjs";

test("desktop fixture keeps the requested history size and rich-message cadence", () => {
  const messages = desktopMessages(100).items;

  assert.equal(messages.length, 100);
  assert.equal(messages.filter((message) => message.content.includes("```ts")).length, 10);
  assert.equal(messages.filter((message) => message.tool_chain.length > 0).length, 10);
});

test("mobile fixture and stream patches preserve protocol identity", () => {
  const snapshot = mobileSnapshot(300, { streaming: true });
  const delta = mobileStreamPatch(snapshot, 0, "片");
  const terminal = mobileTerminalPatch(snapshot, "片".repeat(600));

  assert.equal(snapshot.messages.length, 300);
  assert.equal(snapshot.messages.at(-1).streaming, true);
  assert.equal(delta.messageId, snapshot.messages.at(-1).id);
  assert.equal(terminal.message.streaming, false);
  assert.equal(terminal.state.protocolVersion, 1);
  assert.equal("messages" in terminal.state, false);
});
