import assert from "node:assert/strict";
import test from "node:test";

import { isGeneratingChatStatus } from "./web-chat-status.ts";

test("only an active turn owns the composer stop action", () => {
  assert.equal(isGeneratingChatStatus("submitted"), true);
  assert.equal(isGeneratingChatStatus("streaming"), true);
  assert.equal(isGeneratingChatStatus("finalizing"), false);
  assert.equal(isGeneratingChatStatus("idle"), false);
  assert.equal(isGeneratingChatStatus("error"), false);
});
