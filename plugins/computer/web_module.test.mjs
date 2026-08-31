import assert from "node:assert/strict";
import test from "node:test";

import {
  BACKGROUND_HOLD_MS,
  reconnectDelay,
  shouldOpenForActivity,
} from "./web/connection.js";
import {
  clipboardShortcut,
  keysymForKey,
  pasteKeySequence,
} from "./web/remote-input.js";

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

test("Computer sends browser keys with the same explicit X11 mapping as the display", () => {
  assert.equal(keysymForKey("Enter", "Enter"), 0xff0d);
  assert.equal(keysymForKey("Control", "ControlRight"), 0xffe4);
  assert.equal(keysymForKey("F12", "F12"), 0xffc9);
  assert.equal(keysymForKey("a", "KeyA"), 0x61);
  assert.equal(keysymForKey("花", "KeyH"), 0x010082b1);
  assert.equal(keysymForKey("Unidentified", ""), null);
});

test("Computer separates host clipboard shortcuts from ordinary remote keys", () => {
  assert.equal(clipboardShortcut("c", true, false, false), "copy");
  assert.equal(clipboardShortcut("V", false, true, false), "paste");
  assert.equal(clipboardShortcut("v", true, false, true), null);
  assert.equal(clipboardShortcut("x", true, false, false), null);
  assert.equal(clipboardShortcut("v", false, false, false), null);
});

test("Computer turns host Command+V into a clean Linux Control+V chord", () => {
  assert.deepEqual(
    pasteKeySequence(false, ["MetaLeft"]),
    [
      { keysym: 0xffeb, code: "MetaLeft", down: false },
      { keysym: 0xffe3, code: "ControlLeft", down: true },
      { keysym: 0x76, code: "KeyV", down: true },
      { keysym: 0x76, code: "KeyV", down: false },
      { keysym: 0xffe3, code: "ControlLeft", down: false },
      { keysym: 0xffeb, code: "MetaLeft", down: true },
    ],
  );
});

test("Computer reuses a held remote Control key for Control+V", () => {
  assert.deepEqual(
    pasteKeySequence(true),
    [
      { keysym: 0x76, code: "KeyV", down: true },
      { keysym: 0x76, code: "KeyV", down: false },
    ],
  );
});
