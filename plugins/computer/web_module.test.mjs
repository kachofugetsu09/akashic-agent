import assert from "node:assert/strict";
import test from "node:test";

import {
  BACKGROUND_HOLD_MS,
  reconnectDelay,
  shouldOpenForActivity,
} from "./web/connection.js";

test("Computer keeps a hidden desktop briefly for an instant reopen", () => {
  assert.equal(BACKGROUND_HOLD_MS, 30_000);
});

test("Computer reconnects quickly and caps repeated failure delay", () => {
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5, 20].map(reconnectDelay),
    [500, 1_000, 2_000, 4_000, 8_000, 10_000, 10_000],
  );
});

test("Computer opens for every new Agent action without replaying old activity", () => {
  assert.equal(shouldOpenForActivity(null, 8, false), false);
  assert.equal(shouldOpenForActivity(null, 8, true), true);
  assert.equal(shouldOpenForActivity(8, 8, true), false);
  assert.equal(shouldOpenForActivity(8, 9, false), true);
  assert.equal(shouldOpenForActivity(8, 9, true), true);
});
